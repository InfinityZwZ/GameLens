import threading

import pytest

from gamelens.rag.reranker import (
    MODEL_ID,
    MODEL_REVISION,
    BGEReranker,
    RerankerError,
    apply_reranker,
)


def _ready(root):
    cache = root / "data/rag/models/reranker"
    cache.mkdir(parents=True)
    (cache / f"{MODEL_REVISION}.ready").write_text(MODEL_ID, encoding="utf-8")


class FakeModel:
    def __init__(self, scores=None, error=None):
        self.scores = scores or []
        self.error = error
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        if self.error:
            raise self.error
        return self.scores[: len(pairs)]


def test_reranker_load_is_revision_locked_cpu_and_local_only(tmp_path):
    _ready(tmp_path)
    captured = {}
    model = FakeModel([0.5])

    def factory(model_id, **kwargs):
        captured.update({"model_id": model_id, **kwargs})
        return model

    reranker = BGEReranker(tmp_path, model_factory=factory)
    reranker.load()

    assert captured == {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "cache_folder": str(tmp_path / "data/rag/models/reranker"),
        "device": "cpu",
        "trust_remote_code": False,
        "local_files_only": True,
        "max_length": 1024,
    }


def test_reranker_score_is_primary_and_hybrid_rank_breaks_ties(tmp_path):
    _ready(tmp_path)
    model = FakeModel([0.2, 0.9, 0.9])
    reranker = BGEReranker(tmp_path, model_factory=lambda *args, **kwargs: model)
    hits = [
        {"id": "a", "kind": "review", "source_id": "a", "text": "alpha", "fusion_score": 3.0},
        {"id": "b", "kind": "review", "source_id": "b", "text": "beta", "fusion_score": 2.0},
        {"id": "c", "kind": "review", "source_id": "c", "text": "gamma", "fusion_score": 1.0},
    ]

    ranked = reranker.rerank("query", hits)

    assert [hit["source_id"] for hit in ranked] == ["b", "c", "a"]
    assert [hit["hybrid_rank"] for hit in ranked] == [2, 3, 1]
    assert all("reranker_score" in hit for hit in ranked)
    assert model.calls[0][1]["batch_size"] == 2
    assert model.calls[0][1]["show_progress_bar"] is False


def test_auto_mode_falls_back_with_explicit_reason_when_model_missing(tmp_path):
    hits = [{"id": "a", "source_id": "a", "text": "alpha", "fusion_score": 1.0}]
    ranked, metadata = apply_reranker("query", hits, mode="auto", reranker=BGEReranker(tmp_path))

    assert ranked == hits
    assert metadata["status"] == "hybrid_fallback"
    assert metadata["reason_code"] == "reranker_unavailable"


def test_required_mode_stops_when_model_missing(tmp_path):
    with pytest.raises(RerankerError) as raised:
        apply_reranker("query", [{"id":"a","source_id":"a","text":"x"}],
                       mode="required", reranker=BGEReranker(tmp_path))
    assert raised.value.code == "reranker_unavailable"


def test_no_candidates_never_loads_model_even_in_required_mode(tmp_path):
    ranked,metadata=apply_reranker("query",[],mode="required",reranker=BGEReranker(tmp_path))
    assert ranked==[] and metadata["status"]=="no_candidates"


def test_inference_failure_and_timeout_have_distinct_codes(tmp_path):
    _ready(tmp_path)
    failing = BGEReranker(tmp_path, model_factory=lambda *a, **k: FakeModel(error=RuntimeError("boom")))
    ranked, metadata = apply_reranker("q", [{"id": "a", "source_id": "a", "text": "x"}],
                                         mode="auto", reranker=failing)
    assert ranked[0]["id"] == "a"
    assert metadata["reason_code"] == "reranker_inference_failed"

    class BlockingModel(FakeModel):
        def predict(self,pairs,**kwargs):
            threading.Event().wait(1)
            return [0.1]
    slow = BGEReranker(tmp_path, model_factory=lambda *a, **k: BlockingModel(),timeout_seconds=0.01)
    with pytest.raises(RerankerError) as raised:
        apply_reranker("q", [{"id": "a", "source_id": "a", "text": "x"}],
                       mode="required", reranker=slow)
    assert raised.value.code == "reranker_timeout"


def test_invalid_prediction_shape_falls_back_in_auto_mode(tmp_path):
    _ready(tmp_path)
    model=FakeModel()
    model.predict=lambda *args,**kwargs: object()
    ranked,metadata=apply_reranker("q",[{"id":"a","source_id":"a","text":"x"}],
                                     mode="auto",reranker=BGEReranker(tmp_path,model_factory=lambda *a,**k:model))
    assert ranked[0]["id"]=="a"
    assert metadata["reason_code"]=="reranker_inference_failed"


def test_candidate_limit_is_twenty(tmp_path):
    _ready(tmp_path)
    model = FakeModel([float(i) for i in range(20)])
    reranker = BGEReranker(tmp_path, model_factory=lambda *a, **k: model)
    hits = [{"id": str(i), "source_id": str(i), "text": str(i), "fusion_score": 1.0}
            for i in range(25)]

    ranked = reranker.rerank("query", hits)

    assert len(ranked) == 20
    assert len(model.calls[0][0]) == 20
