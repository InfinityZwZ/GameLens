"""Reproducible retrieval-only evaluation; never calls a paid LLM API."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gamelens.core.paths import PROJECT_ROOT
from gamelens.rag.embedding import CONFIG, LocalEmbedding
from gamelens.rag.index import (
    BM25_B,
    BM25_K1,
    CANDIDATE_POOL_FACTOR,
    MIN_CANDIDATE_POOL,
    RRF_K,
    TOPIC_BOOST,
    GameIndex,
)
from gamelens.rag.tools import SearchArgs
from gamelens.rag.reranker import (
    BATCH_SIZE, DEFAULT_TIMEOUT_SECONDS, MAX_PAIR_TOKENS, BGEReranker, CANDIDATE_LIMIT,
    MODEL_ID as RERANKER_MODEL_ID,
    MODEL_REVISION as RERANKER_MODEL_REVISION, RerankerError,
)
from gamelens.rag.screening import screen_candidates
from gamelens.storage.atomic_io import atomic_write_json, atomic_write_text


VALID_KINDS = {"reviews", "announcements", "both"}
VALID_LANGUAGES = {"zh", "en", "mixed"}
VALID_ROLES = {"support", "counterexample", "partial"}
VALID_REVIEW_STATUS = {"reviewed", "pending_review"}
VALID_SPLITS = {"development", "holdout"}


@dataclass(frozen=True)
class RetrievalDataset:
    dataset_id: str
    cases: list[dict[str, Any]]
    source_path: Path
    sha256: str
    label_provenance: dict[str, dict[str, Any]]

    @property
    def formal_cases(self):
        return [case for case in self.cases if case["review_status"] == "reviewed"]

    @property
    def total_cases(self):
        return len(self.cases)

    @property
    def reviewed_cases(self):
        return len(self.formal_cases)

    @property
    def pending_cases(self):
        return self.total_cases - self.reviewed_cases


def _validate_case(case, seen):
    required = {"id", "query", "kind", "split", "language", "tags", "review_status", "labels"}
    missing = required - set(case)
    if missing:
        raise ValueError(f"检索题缺少字段：{', '.join(sorted(missing))}")
    if not isinstance(case["id"], str) or not case["id"] or case["id"] in seen:
        raise ValueError("检索题ID必须非空且唯一")
    seen.add(case["id"])
    if not isinstance(case["query"], str) or len(case["query"].strip()) < 2:
        raise ValueError(f"{case['id']}的query无效")
    if case["kind"] not in VALID_KINDS or case["language"] not in VALID_LANGUAGES:
        raise ValueError(f"{case['id']}的kind或language无效")
    if case["split"] not in VALID_SPLITS or case["review_status"] not in VALID_REVIEW_STATUS:
        raise ValueError(f"{case['id']}的split或review_status无效")
    if not isinstance(case["tags"], list) or not all(isinstance(tag, str) and tag for tag in case["tags"]):
        raise ValueError(f"{case['id']}的tags无效")
    if not isinstance(case["labels"], list):
        raise ValueError(f"{case['id']}的labels无效")
    source_ids = set()
    for label in case["labels"]:
        if set(label) != {"source_id", "relevance", "role"}:
            raise ValueError(f"{case['id']}的label字段无效")
        if not isinstance(label["source_id"], str) or not label["source_id"] or label["source_id"] in source_ids:
            raise ValueError(f"{case['id']}的source_id必须非空且唯一")
        source_ids.add(label["source_id"])
        if label["relevance"] not in {1, 2, 3} or label["role"] not in VALID_ROLES:
            raise ValueError(f"{case['id']}的相关性等级或证据角色无效")


def load_retrieval_dataset(path):
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload.get("dataset_id"), str) or not payload["dataset_id"]:
        raise ValueError("检索评测集缺少dataset_id")
    provenance = payload.get("label_provenance")
    if not isinstance(provenance, dict) or set(provenance) != VALID_REVIEW_STATUS:
        raise ValueError("检索评测集必须记录reviewed和pending_review标签来源")
    for status in VALID_REVIEW_STATUS:
        item = provenance[status]
        if (not isinstance(item, dict) or not isinstance(item.get("origin"), str)
                or not item["origin"] or not isinstance(item.get("formal_metrics_allowed"), bool)):
            raise ValueError(f"{status}的标签来源无效")
    if provenance["pending_review"]["formal_metrics_allowed"]:
        raise ValueError("pending_review标签不能进入正式指标")
    if not provenance["reviewed"]["formal_metrics_allowed"]:
        raise ValueError("reviewed标签必须明确允许进入正式指标")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("检索评测集必须包含题目")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("检索题必须是对象")
        _validate_case(case, seen)
    return RetrievalDataset(payload["dataset_id"], cases, path, hashlib.sha256(raw).hexdigest(), provenance)


def validate_dataset_sources(dataset, corpus_dir):
    """Fail before measurement when a label does not exist in the frozen corpus."""
    corpus_dir = Path(corpus_dir)
    reviews = json.loads((corpus_dir / "730_reviews.json").read_text(encoding="utf-8"))
    news = json.loads((corpus_dir / "730_news.json").read_text(encoding="utf-8"))
    by_kind = {
        "reviews": {str(row["review_id"]) for row in reviews},
        "announcements": {str(row["news_id"]) for row in news},
    }
    for case in dataset.cases:
        allowed = (by_kind[case["kind"]] if case["kind"] != "both"
                   else by_kind["reviews"] | by_kind["announcements"])
        for label in case["labels"]:
            if label["source_id"] not in allowed:
                raise ValueError(
                    f"{case['id']}标签{label['source_id']}不在{case['kind']}冻结语料中")


def percentile(values, fraction):
    if not values:
        return None
    if not 0 < fraction <= 1:
        raise ValueError("百分位必须在0到1之间")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def evaluate_ranking(case, retrieved_ids):
    grades = {label["source_id"]: label["relevance"] for label in case["labels"]}
    retrieved_ids = list(dict.fromkeys(retrieved_ids))
    if not grades:
        return {"is_no_answer": True, "no_answer_correct": not retrieved_ids,
                "no_answer_candidate_count": len(retrieved_ids),
                "recall_at_5": None, "recall_at_10": None,
                "mrr_at_10": None, "ndcg_at_10": None}
    relevant = set(grades)
    recall = lambda k: len(relevant & set(retrieved_ids[:k])) / len(relevant)
    reciprocal = next((1 / rank for rank, source_id in enumerate(retrieved_ids[:10], 1)
                       if source_id in relevant), 0.0)
    dcg = sum((2 ** grades.get(source_id, 0) - 1) / math.log2(rank + 1)
              for rank, source_id in enumerate(retrieved_ids[:10], 1))
    ideal = sum((2 ** grade - 1) / math.log2(rank + 1)
                for rank, grade in enumerate(sorted(grades.values(), reverse=True)[:10], 1))
    return {"is_no_answer": False, "no_answer_correct": None, "no_answer_candidate_count": None,
            "recall_at_5": recall(5), "recall_at_10": recall(10),
            "mrr_at_10": reciprocal, "ndcg_at_10": dcg / ideal if ideal else 0.0}


def _mean(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def summarize_mode(rows):
    no_answer = [row for row in rows if row.get("is_no_answer")]
    latencies = [measurement for row in rows
                 for measurement in row.get("measurements_ms", [row["latency_ms"]])]
    return {"cases": len(rows), "answerable_cases": sum(not row.get("is_no_answer") for row in rows),
            "no_answer_cases": len(no_answer), "mean_recall_at_5": _mean(rows, "recall_at_5"),
            "mean_recall_at_10": _mean(rows, "recall_at_10"), "mrr_at_10": _mean(rows, "mrr_at_10"),
            "ndcg_at_10": _mean(rows, "ndcg_at_10"),
            "no_answer_accuracy": _mean(no_answer, "no_answer_correct"),
            "latency_p50_ms": percentile(latencies, 0.5), "latency_p95_ms": percentile(latencies, 0.95)}


def summarize_slices(cases, rows):
    by_id = {case["id"]: case for case in cases}
    groups = {"split": {}, "language": {}, "kind": {}, "tags": {}}
    for row in rows:
        case = by_id[row["case_id"]]
        values = {"split": [case["split"]], "language": [case["language"]],
                  "kind": [case["kind"]], "tags": case["tags"]}
        for dimension, labels in values.items():
            for label in labels:
                groups[dimension].setdefault(label, []).append(row)
    return {dimension: {label: summarize_mode(group) for label, group in sorted(values.items())}
            for dimension, values in groups.items()}


def _source_hashes(root):
    files = [root / "evals/retrieval/730_reviews.json", root / "evals/retrieval/730_news.json"]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _implementation_hashes(project_root):
    module_root = Path(__file__).resolve().parents[2]
    paths = {
        "evaluation_retrieval.py": project_root / "gamelens/evaluation/retrieval.py",
        "rag_index.py": project_root / "gamelens/rag/index.py",
        "rag_reranker.py": project_root / "gamelens/rag/reranker.py",
        "rag_text.py": project_root / "gamelens/rag/text.py",
        "rag_relevance.py": project_root / "gamelens/rag/relevance.py",
        "rag_embedding.py": project_root / "gamelens/rag/embedding.py",
    }
    fallbacks = {
        "evaluation_retrieval.py": Path(__file__),
        "rag_index.py": module_root / "gamelens/rag/index.py",
        "rag_reranker.py": module_root / "gamelens/rag/reranker.py",
        "rag_text.py": module_root / "gamelens/rag/text.py",
        "rag_relevance.py": module_root / "gamelens/rag/relevance.py",
        "rag_embedding.py": module_root / "gamelens/rag/embedding.py",
    }
    return {name: hashlib.sha256((path if path.is_file() else fallbacks[name]).read_bytes()).hexdigest()
            for name, path in paths.items()}


def _memory_bytes():
    try:
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                        ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended_virtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _acceptance_gaps(dataset):
    cases = dataset.formal_cases
    tags = {tag for case in cases for tag in case["tags"]}
    roles = {label["role"] for case in cases for label in case["labels"]}
    checks = {
        "at_least_40_reviewed": len(cases) >= 40,
        "mixed_language": any(case["language"] == "mixed" for case in cases),
        "no_answer": any(not case["labels"] for case in cases),
        "negation": "negation" in tags,
        "counterexample": "counterexample" in roles,
        "kind_both": any(case["kind"] == "both" for case in cases),
        "long_announcement": "long_announcement" in tags,
    }
    return [name for name, passed in checks.items() if not passed]


def _reranker_gate(dataset, mode_results, availability):
    if availability != "measured":
        return {"status": availability, "recommended_default_mode": "off",
                "reason": "本地固定revision的Reranker不可用，未生成质量指标"}
    if _acceptance_gaps(dataset):
        return {"status": "blocked_pending_human_review", "recommended_default_mode": "off",
                "reason": "人工复核Holdout尚未达到正式门槛，不能据小样本切换默认值"}
    base = mode_results["hybrid"]["slices"]["split"].get("holdout", {})
    reranked = mode_results["hybrid_reranker"]["slices"]["split"].get("holdout", {})
    improved = (reranked.get("ndcg_at_10") or 0) > (base.get("ndcg_at_10") or 0)
    recall_ok = (reranked.get("mean_recall_at_10") or 0) >= (base.get("mean_recall_at_10") or 0)
    cases={case["id"]:case for case in dataset.formal_cases}
    base_rows={row["case_id"]:row for row in mode_results["hybrid"]["results"]}
    reranked_rows={row["case_id"]:row for row in mode_results["hybrid_reranker"]["results"]}
    critical=[case_id for case_id,case in cases.items() if "negation" in case["tags"]
              or any(label["role"]=="counterexample" for label in case["labels"])]
    critical_ok=all((reranked_rows[case_id].get("recall_at_10") or 0)
                    >= (base_rows[case_id].get("recall_at_10") or 0) for case_id in critical)
    status = "accepted" if improved and recall_ok and critical_ok else "rejected_quality_gate"
    return {"status": status, "recommended_default_mode": "auto" if status == "accepted" else "off",
            "holdout_hybrid": base, "holdout_hybrid_reranker": reranked,
            "checks": {"ndcg_at_10_improved": improved, "recall_at_10_not_lower": recall_ok,
                       "negation_and_counterexample_no_recall_regression":critical_ok},
            "critical_case_ids":critical}


def _render_report(payload):
    dataset = payload["dataset"]
    lines = ["# GameLens 检索 Baseline", "", f"生成时间：{payload['created_at']}",
             f"评测集：{dataset['id']}（SHA-256 `{dataset['sha256']}`）", "",
             "## 标签状态", "",
             f"总题数：{dataset['total_cases']}；正式人工题：{dataset['reviewed_cases']}；"
             f"待人工复核：{dataset['pending_cases']}。正式指标只计算 `reviewed` 题目。", "",
             f"Baseline 状态：`{payload['baseline_status']}`。", "",
             "未满足门槛：" + ("、".join(payload["acceptance_gaps"]) or "无") + "。", "",
             "## 预备结果", "",
             "| 模式 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | 无答案准确率 | P50 ms | P95 ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def show(value):
        return "N/A" if value is None else f"{value:.4f}"
    for mode, result in payload["modes"].items():
        summary = result["summary"]
        lines.append(f"| {mode} | {show(summary['mean_recall_at_5'])} | {show(summary['mean_recall_at_10'])} | "
                     f"{show(summary['mrr_at_10'])} | {show(summary['ndcg_at_10'])} | "
                     f"{show(summary['no_answer_accuracy'])} | {show(summary['latency_p50_ms'])} | "
                     f"{show(summary['latency_p95_ms'])} |")
    lines += ["", "## Reranker 门槛", "",
              f"状态：`{payload['reranker_gate']['status']}`；运行默认值："
              f"`{payload['configuration']['reranker']['current_runtime_default']}`；建议默认值："
              f"`{payload['reranker_gate']['recommended_default_mode']}`。",
              payload['reranker_gate'].get('reason','未通过正式门槛，不切换默认值。'), "",
              "## Chunk 消融", "",
              f"状态：`{payload['chunk_ablation']['status']}`；默认策略："
              f"`{payload['chunk_ablation'].get('default_strategy','fixed')}`。", ""]
    ablation=payload["chunk_ablation"]
    if isinstance(ablation.get("fixed"),dict) and isinstance(ablation.get("sentence"),dict):
        lines += ["| 分段策略 | Recall@10 | MRR@10 | nDCG@10 | P50 ms | P95 ms |",
                  "|---|---:|---:|---:|---:|---:|"]
        for strategy in ("fixed","sentence"):
            summary=ablation[strategy]
            lines.append(f"| {strategy} | {show(summary['mean_recall_at_10'])} | "
                         f"{show(summary['mrr_at_10'])} | {show(summary['ndcg_at_10'])} | "
                         f"{show(summary['latency_p50_ms'])} | {show(summary['latency_p95_ms'])} |")
        lines += ["",ablation.get("reason","未达到切换默认分段的门槛。"),""]
    lines += ["", "## 尚未测量的条件", ""]
    for name, condition in payload["unavailable_conditions"].items():
        lines.append(f"- `{name}`：{condition['status']}；{condition['reason']}")
    lines += ["", "## 证据边界", "",
              "这是小型本地合成检索评测，不是完整 Agent 准确率，也不证明 Steam 总体表现。"
              "待复核题不会进入正式指标；无答案准确率只检查检索器是否返回候选，不等同于事实不存在。", "",
              "Embedding 严格从本地缓存加载；本评测不会下载模型，也不会调用付费 LLM API。", ""]
    return "\n".join(lines)


def run_benchmark(project_root=PROJECT_ROOT, output_dir=None, *, warmups=1, repeats=5,
                  modes=("bm25", "dense", "hybrid"), reranker=None,
                  screening_client=None, include_reranker=True, include_chunk_ablation=True):
    project_root = Path(project_root).resolve()
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups必须非负且repeats至少为1")
    dataset = load_retrieval_dataset(project_root / "evals/retrieval/cases.json")
    if not dataset.formal_cases:
        raise ValueError("没有已人工复核的检索题，不能生成正式指标")
    validate_dataset_sources(dataset, project_root / "evals/retrieval")
    modes = tuple(dict.fromkeys(modes))
    if not modes or set(modes) - {"bm25", "dense", "hybrid"}:
        raise ValueError("modes只能包含bm25、dense、hybrid")
    created = datetime.now(timezone.utc)
    output_dir = Path(output_dir) if output_dir else (
        project_root / "evals/results/retrieval" / created.strftime("%Y%m%d_%H%M%S_%f"))
    if output_dir.exists():
        raise ValueError("输出目录非空，不能覆盖历史检索基线")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gamelens-rag-eval-", dir=output_dir.parent) as directory:
        root = Path(directory)
        lexical_root = root / "lexical"
        vector_root = root / "vector"
        for index_root in (lexical_root, vector_root):
            raw = index_root / "data/raw"
            raw.mkdir(parents=True)
            for name in ("730_reviews.json", "730_news.json"):
                shutil.copyfile(project_root / "evals/retrieval" / name, raw / name)
        state = {"appid": 730, "artifacts": {}, "window_days": 7}
        indexes = {}
        build = reuse = None
        lexical_build_ms = vector_build_ms = model_load_ms = None
        if "bm25" in modes:
            lexical_index = GameIndex(lexical_root, encoder=object())
            build_started = time.perf_counter()
            lexical_build = lexical_index.build_lexical(730, state)
            lexical_build_ms = round((time.perf_counter() - build_started) * 1000, 3)
            indexes["bm25"] = lexical_index
        if set(modes) & {"dense", "hybrid"}:
            encoder = LocalEmbedding(project_root)
            if not encoder.ready:
                raise RuntimeError("本地BGE-M3缓存不可用；评测不会自动下载模型")
            model_started = time.perf_counter()
            encoder.load(download=False)
            model_load_ms = round((time.perf_counter() - model_started) * 1000, 3)
            vector_index = GameIndex(vector_root, encoder)
            build_started = time.perf_counter()
            build = vector_index.build(730, state)
            vector_build_ms = round((time.perf_counter() - build_started) * 1000, 3)
            reuse = vector_index.build(730, state)
            for mode in set(modes) & {"dense", "hybrid"}:
                indexes[mode] = vector_index
        mode_results = {}

        def measure_mode(mode,index):
            rows = []
            for case in dataset.formal_cases:
                args = SearchArgs(appid=730, query=case["query"], kind=case["kind"], top_k=10)
                for _ in range(warmups):
                    index.search(args, state, mode=mode, use_topic_boost=False)
                rankings = []
                measurements = []
                for _ in range(repeats):
                    started = time.perf_counter()
                    hits, _ = index.search(args, state, mode=mode, use_topic_boost=False)
                    measurements.append((time.perf_counter() - started) * 1000)
                    rankings.append([hit["source_id"] for hit in hits])
                if any(ranking != rankings[0] for ranking in rankings[1:]):
                    raise RuntimeError(f"{mode}/{case['id']}重复运行排名不一致")
                metrics = evaluate_ranking(case, rankings[0])
                rows.append({"case_id": case["id"], "query": case["query"],
                             "retrieved_ids": rankings[0], "latency_ms": sum(measurements) / len(measurements),
                             "measurements_ms": measurements, **metrics})
            return {"summary": summarize_mode(rows),
                    "slices": summarize_slices(dataset.formal_cases, rows), "results": rows}

        for mode in modes:
            mode_results[mode] = measure_mode(mode,indexes[mode])

        reranker_status = "not_requested"
        reranker_reason = "评测未请求Reranker"
        reranker_load_ms = None
        active_reranker = reranker
        if include_reranker and "hybrid" in indexes:
            active_reranker = active_reranker or BGEReranker(project_root)
            if not active_reranker.ready:
                reranker_status = "blocked_model_unavailable"
                reranker_reason = "固定revision的BGE Reranker不在本地；评测不会自动下载模型"
            else:
                try:
                    started = time.perf_counter()
                    active_reranker.load()
                    reranker_load_ms = round((time.perf_counter()-started)*1000,3)
                    rows=[]
                    index=indexes["hybrid"]
                    for case in dataset.formal_cases:
                        args=SearchArgs(appid=730,query=case["query"],kind=case["kind"],top_k=10)
                        for _ in range(warmups):
                            pool,_=index.search(args,state,mode="hybrid",use_topic_boost=False,
                                                finalize=False,candidate_limit=CANDIDATE_LIMIT)
                            index.select_hits(active_reranker.rerank(case["query"],pool),args)
                        rankings=[];measurements=[];candidate_ms=[];rerank_ms=[]
                        for _ in range(repeats):
                            total_started=time.perf_counter();candidate_started=total_started
                            pool,_=index.search(args,state,mode="hybrid",use_topic_boost=False,
                                                finalize=False,candidate_limit=CANDIDATE_LIMIT)
                            rerank_started=time.perf_counter()
                            ranked=active_reranker.rerank(case["query"],pool)
                            hits=index.select_hits(ranked,args)
                            finished=time.perf_counter()
                            candidate_ms.append((rerank_started-candidate_started)*1000)
                            rerank_ms.append((finished-rerank_started)*1000)
                            measurements.append((finished-total_started)*1000)
                            rankings.append([hit["source_id"] for hit in hits])
                        if any(ranking!=rankings[0] for ranking in rankings[1:]):
                            raise RuntimeError(f"hybrid_reranker/{case['id']}重复运行排名不一致")
                        metrics=evaluate_ranking(case,rankings[0])
                        rows.append({"case_id":case["id"],"query":case["query"],
                            "retrieved_ids":rankings[0],"latency_ms":sum(measurements)/len(measurements),
                            "measurements_ms":measurements,"candidate_measurements_ms":candidate_ms,
                            "reranker_measurements_ms":rerank_ms,**metrics})
                    mode_results["hybrid_reranker"]={"summary":summarize_mode(rows),
                        "slices":summarize_slices(dataset.formal_cases,rows),"results":rows}
                    reranker_status="measured";reranker_reason="已使用本地固定revision测量"
                except RerankerError as error:
                    reranker_status=error.code
                    reranker_reason=str(error)

        screening_status = "not_measured"
        screening_reason = "LLM Screening需显式注入客户端并单独授权付费评测"
        if screening_client is not None and reranker_status == "measured":
            rows=[]
            index=indexes["hybrid"]
            for case in dataset.formal_cases:
                args=SearchArgs(appid=730,query=case["query"],kind=case["kind"],top_k=10)

                def run_screening_once():
                    total_started=time.perf_counter()
                    pool,_=index.search(args,state,mode="hybrid",use_topic_boost=False,
                                        finalize=False,candidate_limit=CANDIDATE_LIMIT)
                    rerank_started=time.perf_counter()
                    ranked=active_reranker.rerank(case["query"],pool)
                    hits=index.select_hits(ranked,args)
                    screening_started=time.perf_counter()
                    selected,_=screen_candidates(
                        screening_client,case["query"],case["query"],hits)
                    finished=time.perf_counter()
                    return selected,(
                        (rerank_started-total_started)*1000,
                        (screening_started-rerank_started)*1000,
                        (finished-screening_started)*1000,
                        (finished-total_started)*1000,
                    )

                for _ in range(warmups):
                    run_screening_once()
                rankings=[];measurements=[];candidate_ms=[];rerank_ms=[];screening_ms=[]
                for _ in range(repeats):
                    selected,timings=run_screening_once()
                    candidate_elapsed,rerank_elapsed,screening_elapsed,total_elapsed=timings
                    rankings.append([hit["source_id"] for hit in selected])
                    candidate_ms.append(candidate_elapsed)
                    rerank_ms.append(rerank_elapsed)
                    screening_ms.append(screening_elapsed)
                    measurements.append(total_elapsed)
                if any(ranking!=rankings[0] for ranking in rankings[1:]):
                    raise RuntimeError(
                        f"hybrid_reranker_screening/{case['id']}重复运行排名不一致")
                metrics=evaluate_ranking(case,rankings[0])
                rows.append({"case_id":case["id"],"query":case["query"],
                    "retrieved_ids":rankings[0],"latency_ms":sum(measurements)/len(measurements),
                    "measurements_ms":measurements,"candidate_measurements_ms":candidate_ms,
                    "reranker_measurements_ms":rerank_ms,
                    "screening_measurements_ms":screening_ms,**metrics})
            mode_results["hybrid_reranker_screening"]={"summary":summarize_mode(rows),
                "slices":summarize_slices(dataset.formal_cases,rows),"results":rows}
            screening_status="measured"
            screening_reason="已使用显式注入的Screening客户端独立测量"
        elif screening_client is not None:
            screening_status="blocked_reranker_unavailable"
            screening_reason="Screening必须在成功的Reranker测量之后执行"

        chunk_ablation={"status":"not_requested"}
        if include_chunk_ablation and "hybrid" in indexes:
            sentence_root=root/"sentence"
            raw=sentence_root/"data/raw";raw.mkdir(parents=True)
            for name in ("730_reviews.json","730_news.json"):
                shutil.copyfile(project_root/"evals/retrieval"/name,raw/name)
            sentence_index=GameIndex(sentence_root,indexes["hybrid"].encoder,chunk_strategy="sentence")
            sentence_build=sentence_index.build(730,state)
            sentence_result=measure_mode("hybrid",sentence_index)
            chunk_ablation={"status":"exploratory_pending_human_review","default_strategy":"fixed",
                "fixed":mode_results["hybrid"]["summary"],"sentence":sentence_result["summary"],
                "sentence_chunks":sentence_build["manifest"]["chunks"],
                "reason":"需人工复核集通过门槛后才能切换默认分段"}

    failures = [{"mode": mode, "case_id": row["case_id"],
                 "reason": "no_answer_false_positive" if row["is_no_answer"] else "recall_at_10_below_1",
                 "retrieved_ids": row["retrieved_ids"]}
                for mode, result in mode_results.items() for row in result["results"]
                if (row["is_no_answer"] and not row["no_answer_correct"])
                or (not row["is_no_answer"] and row["recall_at_10"] < 1)]
    gaps = _acceptance_gaps(dataset)
    gate=_reranker_gate(dataset,mode_results,reranker_status)
    payload = {"schema_version": 3, "created_at": created.isoformat(),
               "baseline_status": "blocked_pending_human_review" if gaps else "accepted",
               "acceptance_gaps": gaps,
               "dataset": {"id": dataset.dataset_id, "sha256": dataset.sha256,
                           "total_cases": dataset.total_cases, "reviewed_cases": dataset.reviewed_cases,
                           "pending_cases": dataset.pending_cases,
                           "label_provenance": dataset.label_provenance,
                           "source_hashes": _source_hashes(project_root)},
               "configuration": {"warmups_per_case": warmups, "measured_repeats_per_case": repeats,
                                 "top_k": 10, "embedding": CONFIG, "python": platform.python_version(),
                                 "reranker":{"model":RERANKER_MODEL_ID,"revision":RERANKER_MODEL_REVISION,
                                    "candidate_limit":CANDIDATE_LIMIT,"batch_size":BATCH_SIZE,
                                    "max_pair_tokens":MAX_PAIR_TOKENS,"timeout_seconds":DEFAULT_TIMEOUT_SECONDS,
                                    "text_cleaning_revision":CONFIG["text_cleaning_revision"],
                                    "current_runtime_default":"off",
                                    "recommended_default":gate["recommended_default_mode"]},
                                 "screening":{"measured_separately":True,
                                    "client_injected":screening_client is not None,
                                    "position":"after_reranker"},
                                 "platform": platform.platform(),
                                 "hardware": {"processor": platform.processor(), "machine": platform.machine(),
                                              "logical_cpu_count": os.cpu_count(),
                                              "total_memory_bytes": _memory_bytes()},
                                 "retrieval_parameters": {"bm25_k1": BM25_K1, "bm25_b": BM25_B,
                                    "candidate_pool_min": MIN_CANDIDATE_POOL,
                                    "candidate_pool_factor": CANDIDATE_POOL_FACTOR,
                                    "rrf_k": RRF_K, "topic_boost": TOPIC_BOOST,
                                    "topic_boost_enabled_for_evaluation": False,
                                    "dense_min_similarity": SearchArgs(appid=730,query="xx").min_similarity},
                                 "implementation_hashes": _implementation_hashes(project_root)},
               "timings": {"embedding_model_load_ms": model_load_ms,
                           "reranker_model_load_ms":reranker_load_ms,
                           "bm25_index_build_ms": lexical_build_ms,
                           "vector_index_build_ms": vector_build_ms},
               "index": {"lexical_chunks": lexical_build["manifest"]["chunks"] if "bm25" in modes else None,
                         "embedded_chunks": build["embedded_chunks"] if build else None,
                         "second_build_embedded_chunks": reuse["embedded_chunks"] if reuse else None},
               "modes": mode_results, "failures": failures,"reranker_gate":gate,
               "chunk_ablation":chunk_ablation,
               "unavailable_conditions": {
                   "hybrid_reranker": {"status":reranker_status,"reason":reranker_reason},
                   "hybrid_reranker_screening": {"status":screening_status,
                                                   "reason":screening_reason}},
               "scope_note": "检索专项小型合成集；不是完整Agent准确率或生产性能证明。"}
    staging = Path(tempfile.mkdtemp(prefix=".gamelens-retrieval-result-", dir=output_dir.parent))
    try:
        atomic_write_json(staging / "retrieval.json", payload)
        atomic_write_json(staging / "failures.json", failures)
        atomic_write_text(staging / "report.md", _render_report(payload))
        staging.replace(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="GameLens本地检索基线；不调用付费LLM")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--modes", nargs="+", choices=("bm25", "dense", "hybrid"),
                        default=("bm25", "dense", "hybrid"))
    args = parser.parse_args(argv)
    payload = run_benchmark(output_dir=args.output_dir, warmups=args.warmups,
                            repeats=args.repeats, modes=args.modes)
    print(json.dumps({"dataset": payload["dataset"], "timings": payload["timings"],
                      "modes": {name: value["summary"] for name, value in payload["modes"].items()},
                      "unavailable_conditions": payload["unavailable_conditions"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
