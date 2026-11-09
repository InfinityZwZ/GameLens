from pathlib import Path
from fastapi.testclient import TestClient
from gamelens.core.runtime import InvestigationAgent


def report_state(tmp_path):
    directory=tmp_path/"reports";directory.mkdir()
    previous=directory/"730_patch_impact_report.md"
    previous.write_text("# 基础报告\n完整管线分析",encoding="utf-8")
    return {"appid":730,"thread_id":"summary-test","goal":"有哪些性能问题？",
        "final_answer":"主要反馈是卡顿和闪退，尚不能证明由更新导致。",
        "final_detailed_answer":"## 逐版本分析\n这里记录详细统计和证据解释。",
        "artifacts":{"final_report":str(previous),"evidence_assessed":"true"},
        "observations":[{"step":1,"tool":"query_reviews","ok":True,"summary":"查询完成",
                         "data":{"stats":{"count":1},"reviews":[{"review_id":"r1","text":"开枪卡顿"}]}}],
        "final_claims":[{"label":"匹配评论数","source_step":1,"fact_path":["stats","count"],"value":1}],
        "final_citations":[{"source_step":1,"review_id":"r1","quote":"开枪卡顿"}],"final_news_citations":[]}


def test_chat_summary_is_separate_from_full_report(tmp_path):
    state=report_state(tmp_path)
    result=InvestigationAgent._finalize(state)
    assert result["status"]=="completed"
    assert result["final_answer"]==state["final_answer"]
    assert "核验事实" not in result["final_answer"] and "r1" not in result["final_answer"]
    report=Path(result["artifacts"]["final_report"]).read_text(encoding="utf-8")
    for expected in ("逐版本分析","匹配评论数","r1","开枪卡顿","结论边界与进一步验证","完整管线分析"):
        assert expected in report
    assert "````json" not in report and "字段路径" not in report
    assert Path(state["artifacts"]["final_report"]).read_text(encoding="utf-8")=="# 基础报告\n完整管线分析"


def test_resumed_complete_report_does_not_embed_superseded_partial_record(tmp_path):
    state=report_state(tmp_path)
    previous=tmp_path/'reports/730_summary-test_partial_report.md'
    old='# 未完成调查的部分记录\n\n模型综合分析暂时中断。'
    previous.write_text(old,encoding='utf-8')
    state['artifacts']['final_report']=str(previous)
    state['artifacts']['partial_report']=str(previous)
    state.update(generation_pending=False,answer_origin='llm')
    result=InvestigationAgent._finalize(state)
    assert result['status']=='completed'
    report=Path(result['artifacts']['final_report']).read_text(encoding='utf-8')
    assert state['final_answer'] in report and '逐版本分析' in report
    assert '开枪卡顿' in report and 'r1' in report
    assert '模型综合分析暂时中断' not in report
    assert '未完成调查的部分记录' not in report
    assert previous.read_text(encoding='utf-8')==old


def test_short_answer_does_not_bypass_citation_validation(tmp_path):
    state=report_state(tmp_path)
    state["final_citations"][0]["quote"]="伪造原文"
    result=InvestigationAgent._finalize(state)
    assert result["status"]=="failed"
    assert not (tmp_path/"reports/730_summary-test_complete_report.md").exists()


def test_detailed_answer_cannot_turn_zero_semantic_evidence_into_zero_population(tmp_path):
    state = report_state(tmp_path)
    state["observations"].append({"step": 2, "tool": "search_game_evidence", "ok": True, "data": {
        "population": {"eligible_review_count": 113},
        "retrieval": {"candidate_count": 20, "selected_evidence_count": 0},
        "returned_count": 0,
    }})
    state["final_answer"] = "硬条件评论与语义证据数量不同，详细信息请查看报告。"
    state["final_detailed_answer"] = "本次没有符合条件的玩家评论。"

    result = InvestigationAgent._finalize(state)

    assert result["status"] == "failed"
    assert "语义证据筛选为0" in result["error"]


def test_full_report_can_be_read_and_downloaded(tmp_path,monkeypatch):
    from gamelens.api import app as module
    state=report_state(tmp_path);result=InvestigationAgent._finalize(state)
    run=module.RunState(run_id="summary-test",request=module.RunRequest(appid=730),status="completed",artifacts=result["artifacts"])
    monkeypatch.setattr(module,"RUNS",{"summary-test":run})
    monkeypatch.setattr(module,"REPORT_ROOT",tmp_path/"reports")
    client=TestClient(module.app)
    response=client.get("/api/runs/summary-test/report")
    assert response.status_code==200 and "逐版本分析" in response.text
    download=client.get("/api/runs/summary-test/report?download=true")
    assert download.status_code==200 and "attachment" in download.headers["content-disposition"]
    assert download.content==response.content


def test_permanent_delete_removes_complete_report_only_for_owned_run(tmp_path):
    from gamelens.storage.runs import RunStore
    from gamelens.storage.cleanup import delete_conversation
    root=tmp_path/"data";root.mkdir()
    records=root/"runs";records.mkdir()
    reports=tmp_path/"reports";reports.mkdir()
    store=RunStore(root/"web.sqlite")
    for run in ("a","b"):
        store.save({"run_id":run,"conversation_id":f"session-{run}","turn_index":1,"status":"completed",
            "request":{"appid":730,"goal":"调查"},"started_at":"2026-10-08T00:00:00Z","updated_at":"2026-10-08T00:00:00Z",
            "final_answer":"摘要","artifacts":{},"events":[]})
        (reports/f"730_{run}_complete_report.md").write_text("完整报告",encoding="utf-8")
        (reports/f"730_{run}_partial_report.md").write_text("未完成记录",encoding="utf-8")
    delete_conversation(store,"session-a",root/"checkpoint.sqlite",root/"telemetry.sqlite",records)
    assert not (reports/"730_a_complete_report.md").exists()
    assert (reports/"730_b_complete_report.md").is_file()
    assert not (reports/"730_a_partial_report.md").exists()
    assert (reports/"730_b_partial_report.md").is_file()


def test_report_omits_machine_payloads_and_explains_zero_results(tmp_path):
    import hashlib
    import json
    state=report_state(tmp_path)
    snapshot = tmp_path/"private-path"
    snapshot.write_text(json.dumps([{"review_id":f"r{i}","text":"开枪卡顿"} for i in range(100)],ensure_ascii=False),encoding="utf-8")
    state["final_detailed_answer"]='### 分析\n现有证据有限。\n\n```json\n{"source":{"path":"private-path"}}\n```\n\n{"raw":"serialized"}'
    state["observations"].append({"step":2,"tool":"query_reviews","ok":True,"data":{
        "stats":{"count":0,"positive_count":0,"negative_count":0,"positive_rate":None},
        "filters":{"period":"after","keywords":["崩溃","闪退"],"keyword_mode":"all"},"returned_count":0,
        "source":{"path":str(snapshot),"sha256":hashlib.sha256(snapshot.read_bytes()).hexdigest(),"cached_review_count":100},"execution":{"elapsed_ms":99}}})
    state["observations"].append({"step":3,"tool":"verify_investigation","ok":True,"data":{}})
    result=InvestigationAgent._finalize(state)
    report=Path(result["artifacts"]["final_report"]).read_text(encoding="utf-8")
    assert "同时包含关键词：崩溃、闪退" in report
    assert "这不能证明没有相关问题" in report
    for forbidden in ("```json","private-path","private-hash",'"serialized"',"elapsed_ms","source_step","fact_path"):
        assert forbidden not in report


def test_version_statistics_are_readable_and_do_not_claim_persistence(tmp_path):
    state=report_state(tmp_path)
    state["observations"].append({"step":2,"tool":"investigate_recent_versions","ok":True,"data":{
        "requested_count":5,"available_count":2,"versions_with_reviews":1,
        "versions":[{"title":"更新A","start_inclusive":1800000000,"end_exclusive":1800086400,"stats":{"count":0,"positive_rate":None}}]}})
    result=InvestigationAgent._finalize(state)
    report=Path(result["artifacts"]["final_report"]).read_text(encoding="utf-8")
    assert "计划检查最近5次更新，实际有2个补丁候选" in report
    assert "同一缺陷在每个版本都存在" in report
    assert "| 更新A |" in report
    assert "| --- | --- | --- | --- |\n| 更新A |" in report


def test_repeated_window_checks_use_latest_statistics(tmp_path):
    state=report_state(tmp_path)
    state["observations"] += [{"step":2,"tool":"analyze_patch","ok":True,"data":{"news_id":"patch1","patch_title":"更新A","before_reviews":0,"after_reviews":0}},
        {"step":3,"tool":"analyze_patch","ok":True,"data":{"news_id":"patch1","patch_title":"更新A","before_reviews":38,"after_reviews":36}}]
    result=InvestigationAgent._finalize(state)
    report=Path(result["artifacts"]["final_report"]).read_text(encoding="utf-8")
    assert "更新前评论有38条，更新后有36条" in report
    assert "更新前评论有0条" not in report
