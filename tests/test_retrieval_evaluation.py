import json
import math

import pytest

from gamelens.evaluation.context_benchmark import measure_messages
from gamelens.evaluation.retrieval import (
    evaluate_ranking,
    load_retrieval_dataset,
    percentile,
    run_benchmark,
    summarize_slices,
    summarize_mode,
    validate_dataset_sources,
)
from gamelens.rag.index import bm25_scores, tokens
from gamelens.rag.tools import SearchArgs
from test_unified_retrieval import setup_index


def test_metrics_use_graded_labels_and_literal_expected_values():
    case = {
        "id": "graded",
        "labels": [
            {"source_id": "a", "relevance": 3, "role": "support"},
            {"source_id": "b", "relevance": 1, "role": "counterexample"},
        ],
    }
    result = evaluate_ranking(case, ["x", "a", "b"])

    expected_dcg = 7 / math.log2(3) + 1 / math.log2(4)
    expected_idcg = 7 + 1 / math.log2(3)
    assert result["recall_at_5"] == 1.0
    assert result["recall_at_10"] == 1.0
    assert result["mrr_at_10"] == 0.5
    assert result["ndcg_at_10"] == pytest.approx(expected_dcg / expected_idcg)


def test_no_answer_is_scored_separately_from_relevance_metrics():
    case = {"id": "absent", "labels": []}
    empty = evaluate_ranking(case, [])
    candidates = evaluate_ranking(case, ["unrelated"])

    assert empty["is_no_answer"] is True
    assert empty["no_answer_correct"] is True
    assert candidates["no_answer_correct"] is False
    assert candidates["no_answer_candidate_count"] == 1
    assert empty["recall_at_5"] is None
    assert empty["mrr_at_10"] is None
    assert empty["ndcg_at_10"] is None


def test_pending_labels_are_excluded_from_formal_metrics(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"dataset_id": "fixture", "label_provenance": {
        "reviewed": {"origin": "human_fixture", "formal_metrics_allowed": True},
        "pending_review": {"origin": "generated_fixture", "formal_metrics_allowed": False},
    }, "cases": [
        {"id": "approved", "query": "crash", "kind": "reviews", "split": "holdout",
         "language": "en", "tags": ["crash"], "review_status": "reviewed",
         "labels": [{"source_id": "a", "relevance": 3, "role": "support"}]},
        {"id": "draft", "query": "闪退", "kind": "reviews", "split": "development",
         "language": "zh", "tags": ["crash"], "review_status": "pending_review",
         "labels": [{"source_id": "a", "relevance": 3, "role": "support"}]},
    ]}), encoding="utf-8")

    dataset = load_retrieval_dataset(path)

    assert [case["id"] for case in dataset.formal_cases] == ["approved"]
    assert dataset.total_cases == 2
    assert dataset.reviewed_cases == 1
    assert dataset.pending_cases == 1


def test_checked_in_dataset_has_unique_versioned_cases_and_pending_is_visible():
    dataset = load_retrieval_dataset(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "evals/retrieval/cases.json"
    )

    assert dataset.dataset_id
    assert dataset.total_cases >= 40
    assert dataset.reviewed_cases >= 8
    assert dataset.pending_cases == dataset.total_cases - dataset.reviewed_cases
    assert len({case["id"] for case in dataset.cases}) == dataset.total_cases
    assert all(case["language"] in {"zh", "en", "mixed"} for case in dataset.cases)
    assert dataset.label_provenance["reviewed"]["origin"] == "project_authored_v1"
    assert dataset.label_provenance["pending_review"]["origin"] == "coding_assistant_candidate"
    assert dataset.label_provenance["pending_review"]["formal_metrics_allowed"] is False


def test_bm25_scores_reward_repeated_matching_term():
    documents = [
        __import__("collections").Counter(tokens("crash crash")),
        __import__("collections").Counter(tokens("crash")),
        __import__("collections").Counter(tokens("music")),
    ]
    scores = bm25_scores(documents, {"crash"})

    assert scores[0] > scores[1] > scores[2]


def test_bm25_mode_never_calls_embedding_encoder(tmp_path):
    index, state, encoder = setup_index(
        tmp_path,
        [("a", "DXGI_ERROR_DEVICE_HUNG crash", False), ("b", "music only", True)],
        mapping={"DXGI_ERROR_DEVICE_HUNG crash": 0, "music only": 1, "crash": 0},
    )
    encoder.calls.clear()

    hits, _ = index.search(
        SearchArgs(appid=730, query="crash", kind="reviews", top_k=10),
        state,
        mode="bm25",
    )

    assert [hit["source_id"] for hit in hits] == ["a"]
    assert encoder.calls == []
    assert hits[0]["retrieval_channels"] == ["lexical"]


def test_dense_and_hybrid_modes_share_filters_but_use_expected_channels(tmp_path):
    text = "一加载地图就回到桌面"
    index, state, _ = setup_index(
        tmp_path,
        [("a", text, False), ("b", "crash", False)],
        mapping={text: 0, "crash": 1, "闪退": 0},
    )
    args = SearchArgs(appid=730, query="闪退", lexical_terms=["crash"], kind="reviews", top_k=10)

    dense, _ = index.search(args, state, mode="dense")
    hybrid, _ = index.search(args, state, mode="hybrid")

    assert [hit["source_id"] for hit in dense] == ["a"]
    assert dense[0]["retrieval_channels"] == ["dense"]
    assert {hit["source_id"] for hit in hybrid} == {"a", "b"}


def test_evaluation_can_disable_topic_boost_for_pure_channel_scores(tmp_path):
    index, state, _ = setup_index(
        tmp_path,
        [("a", "crash crash", False), ("b", "crash", False)],
        mapping={"crash crash": 0, "crash": 1},
    )

    hits, _ = index.search(
        SearchArgs(appid=730, query="crash", kind="reviews", top_k=10),
        state,
        mode="bm25",
        use_topic_boost=False,
    )

    assert hits[0]["fusion_score"] == hits[0]["lexical_score"]


def test_mode_summary_uses_nearest_rank_percentiles_and_ignores_no_answer_for_recall():
    rows = [
        {"latency_ms": 10.0, "recall_at_5": 1.0, "recall_at_10": 1.0,
         "mrr_at_10": 1.0, "ndcg_at_10": 1.0, "is_no_answer": False,
         "measurements_ms": [1.0, 2.0, 3.0, 4.0, 100.0]},
        {"latency_ms": 30.0, "recall_at_5": None, "recall_at_10": None,
         "mrr_at_10": None, "ndcg_at_10": None, "is_no_answer": True,
         "no_answer_correct": False, "measurements_ms": [5.0, 6.0, 7.0, 8.0, 9.0]},
    ]

    summary = summarize_mode(rows)

    assert percentile([10.0, 30.0], 0.5) == 10.0
    assert percentile([10.0, 30.0], 0.95) == 30.0
    assert summary["mean_recall_at_10"] == 1.0
    assert summary["no_answer_accuracy"] == 0.0
    assert summary["latency_p50_ms"] == 5.0
    assert summary["latency_p95_ms"] == 100.0


def test_slice_summary_separates_split_language_kind_and_tags():
    cases = [
        {"id": "a", "split": "development", "language": "zh", "kind": "reviews", "tags": ["crash"]},
        {"id": "b", "split": "holdout", "language": "en", "kind": "announcements", "tags": ["startup"]},
    ]
    rows = [
        {"case_id": "a", "latency_ms": 10.0, "recall_at_5": 1.0, "recall_at_10": 1.0,
         "mrr_at_10": 1.0, "ndcg_at_10": 1.0, "is_no_answer": False},
        {"case_id": "b", "latency_ms": 20.0, "recall_at_5": 0.0, "recall_at_10": 1.0,
         "mrr_at_10": 0.5, "ndcg_at_10": 0.6, "is_no_answer": False},
    ]

    slices = summarize_slices(cases, rows)

    assert slices["split"]["development"]["mean_recall_at_5"] == 1.0
    assert slices["split"]["holdout"]["mean_recall_at_5"] == 0.0
    assert slices["language"]["en"]["cases"] == 1
    assert slices["kind"]["announcements"]["cases"] == 1
    assert slices["tags"]["crash"]["cases"] == 1


def test_context_measurement_counts_complete_serialized_messages_as_bytes():
    messages = [{"role": "system", "content": "系统"}, {"role": "user", "content": "schema:{}"}]

    result = measure_messages(messages)

    literal = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    assert result == {"characters": len(literal), "utf8_bytes": len(literal.encode("utf-8")),
                      "exact_tokens": None, "measurement": "serialized_messages_utf8"}


def test_missing_real_snapshot_source_is_explicitly_blocked_without_fetch(tmp_path, monkeypatch):
    from gamelens.evaluation import real_snapshot

    monkeypatch.setattr(real_snapshot, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(real_snapshot, "IDS", {730: ["missing"]})

    with pytest.raises(real_snapshot.SnapshotUnavailableError, match="不会自动获取"):
        real_snapshot.prepare(tmp_path / "snapshot")


def test_benchmark_writes_hashed_immutable_outputs_without_paid_model(tmp_path, monkeypatch):
    from gamelens.evaluation import retrieval
    from gamelens.rag.embedding import CONFIG
    from test_rag import TestEncoder

    project = tmp_path / "project"
    fixture = project / "evals/retrieval"
    fixture.mkdir(parents=True)
    source = __import__("pathlib").Path(__file__).resolve().parents[1] / "evals/retrieval"
    for name in ("cases.json", "730_reviews.json", "730_news.json"):
        (fixture / name).write_bytes((source / name).read_bytes())

    class ReadyEncoder(TestEncoder):
        ready = True

        def load(self, download=False):
            assert download is False

    encoder = ReadyEncoder()
    monkeypatch.setattr(retrieval, "LocalEmbedding", lambda root: encoder)
    output = tmp_path / "result"

    payload = run_benchmark(project, output, warmups=0, repeats=2)

    assert set(payload["modes"]) == {"bm25", "dense", "hybrid"}
    assert payload["dataset"]["reviewed_cases"] == 8
    assert payload["configuration"]["measured_repeats_per_case"] == 2
    assert (output / "retrieval.json").is_file()
    assert (output / "failures.json").is_file()
    assert (output / "report.md").is_file()
    assert payload["configuration"]["embedding"] == CONFIG
    assert payload["baseline_status"] == "blocked_pending_human_review"
    assert payload["configuration"]["retrieval_parameters"]["rrf_k"] == 60
    assert payload["configuration"]["implementation_hashes"]
    assert payload["unavailable_conditions"]["hybrid_reranker"]["status"]=="blocked_model_unavailable"
    assert payload["reranker_gate"]["status"]=="blocked_model_unavailable"
    report=(output/"report.md").read_text(encoding="utf-8")
    assert "| fixed |" in report
    assert "| sentence |" in report
    assert "需人工复核集通过门槛后才能切换默认分段" in report
    with pytest.raises(ValueError, match="不能覆盖"):
        run_benchmark(project, output, warmups=0, repeats=1)


def test_benchmark_measures_injected_reranker_but_does_not_enable_default_gate(tmp_path,monkeypatch):
    from gamelens.evaluation import retrieval
    from test_rag import TestEncoder

    project=tmp_path/"project";fixture=project/"evals/retrieval";fixture.mkdir(parents=True)
    source=__import__("pathlib").Path(__file__).resolve().parents[1]/"evals/retrieval"
    for name in ("cases.json","730_reviews.json","730_news.json"):
        (fixture/name).write_bytes((source/name).read_bytes())

    class ReadyEncoder(TestEncoder):
        ready=True
        def load(self,download=False): assert download is False
    class FakeReranker:
        ready=True
        last_load_ms=0.1
        last_inference_ms=0.2
        def load(self): pass
        def rerank(self,query,hits):
            return [{**hit,"reranker_score":float(len(hits)-rank),"hybrid_rank":rank+1}
                    for rank,hit in enumerate(hits)]

    monkeypatch.setattr(retrieval,"LocalEmbedding",lambda root:ReadyEncoder())
    payload=run_benchmark(project,tmp_path/"result",warmups=0,repeats=1,
                          reranker=FakeReranker(),include_chunk_ablation=False)

    assert "hybrid_reranker" in payload["modes"]
    assert payload["unavailable_conditions"]["hybrid_reranker"]["status"]=="measured"
    assert payload["reranker_gate"]["status"]=="blocked_pending_human_review"
    assert payload["configuration"]["reranker"]["current_runtime_default"]=="off"
    assert payload["reranker_gate"]["recommended_default_mode"]=="off"
    assert payload["modes"]["hybrid_reranker"]["summary"]["cases"]==8


def test_screening_is_measured_as_separate_post_reranker_mode(tmp_path,monkeypatch):
    from gamelens.evaluation import retrieval
    from test_rag import TestEncoder

    project=tmp_path/"project";fixture=project/"evals/retrieval";fixture.mkdir(parents=True)
    source=__import__("pathlib").Path(__file__).resolve().parents[1]/"evals/retrieval"
    for name in ("cases.json","730_reviews.json","730_news.json"):
        (fixture/name).write_bytes((source/name).read_bytes())

    class ReadyEncoder(TestEncoder):
        ready=True
        def load(self,download=False): assert download is False
    class FakeReranker:
        ready=True;last_load_ms=0;last_inference_ms=0
        def load(self): pass
        def rerank(self,query,hits):
            return [{**hit,"reranker_score":1.0,"hybrid_rank":rank+1}
                    for rank,hit in enumerate(hits)]
    class ScreeningClient:
        def generate_json(self,messages):
            candidates=json.loads(messages[1]["content"])["candidates"]
            return {"verdicts":[{"kind":item["kind"],"source_id":item["source_id"],
                "role":"partial","quote":item["excerpt"][:20],"reason":"离线顺序测试"}
                for item in candidates]}

    monkeypatch.setattr(retrieval,"LocalEmbedding",lambda root:ReadyEncoder())
    payload=run_benchmark(project,tmp_path/"screen-result",warmups=0,repeats=1,
                          reranker=FakeReranker(),screening_client=ScreeningClient(),
                          include_chunk_ablation=False)
    assert "hybrid_reranker_screening" in payload["modes"]
    assert payload["unavailable_conditions"]["hybrid_reranker_screening"]["status"]=="measured"
    assert payload["modes"]["hybrid_reranker_screening"]["summary"]["cases"]==8


def test_bm25_only_benchmark_never_constructs_embedding(tmp_path, monkeypatch):
    from gamelens.evaluation import retrieval

    project = tmp_path / "project"
    fixture = project / "evals/retrieval"
    fixture.mkdir(parents=True)
    source = __import__("pathlib").Path(__file__).resolve().parents[1] / "evals/retrieval"
    for name in ("cases.json", "730_reviews.json", "730_news.json"):
        (fixture / name).write_bytes((source / name).read_bytes())

    monkeypatch.setattr(retrieval, "LocalEmbedding", lambda root: (_ for _ in ()).throw(
        AssertionError("BM25-only不得构造Embedding")))

    payload = run_benchmark(project, tmp_path / "bm25-result", warmups=0, repeats=1,
                            modes=("bm25",))

    assert set(payload["modes"]) == {"bm25"}
    assert payload["timings"]["embedding_model_load_ms"] is None


def test_dataset_labels_must_exist_in_matching_frozen_corpus(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"dataset_id": "bad", "label_provenance": {
        "reviewed": {"origin": "human", "formal_metrics_allowed": True},
        "pending_review": {"origin": "generated", "formal_metrics_allowed": False},
    }, "cases": [{"id": "bad", "query": "missing", "kind": "reviews",
                    "split": "holdout", "language": "en", "tags": ["missing"],
                    "review_status": "reviewed", "labels": [
                        {"source_id": "not-in-corpus", "relevance": 3, "role": "support"}]}]}),
                    encoding="utf-8")
    dataset = load_retrieval_dataset(path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "730_reviews.json").write_text("[]", encoding="utf-8")
    (corpus / "730_news.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="not-in-corpus"):
        validate_dataset_sources(dataset, corpus)


def test_real_snapshot_retrieval_is_exploratory_until_human_review(tmp_path, monkeypatch):
    from gamelens.evaluation import real_snapshot
    from gamelens.evaluation.business_cases import materialize_dataset
    from gamelens.rag import embedding
    from test_rag import TestEncoder

    source = tmp_path / "source"
    materialize_dataset(source)
    monkeypatch.setattr(real_snapshot, "PROJECT_ROOT", source)
    monkeypatch.setattr(real_snapshot, "IDS", {730: ["b1", "b2"]})
    monkeypatch.setattr(real_snapshot, "RETRIEVAL_CASES", [
        (730, "闪退", ["b2"], ["b1"]),
        (730, "存档损坏", [], ["b1", "b2"]),
    ])

    class ReadyEncoder(TestEncoder):
        ready = True

        def load(self, download=False):
            assert download is False

    monkeypatch.setattr(embedding, "LocalEmbedding", lambda root: ReadyEncoder())

    payload = real_snapshot.retrieval(tmp_path / "snapshot")

    assert set(payload["modes"]) == {"bm25", "dense", "hybrid"}
    assert payload["formal_metrics"] is None
    assert payload["annotation_status"] == "pending_human_review"
    assert payload["exploratory_only"] is True
    assert payload["manifest"]["retrieval_label_set_sha256"]
    assert payload["runtime_implementation_sha256"]
    assert payload["implementation_matches_snapshot"] is True
