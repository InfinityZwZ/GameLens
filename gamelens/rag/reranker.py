"""Optional local BGE cross-encoder reranking with explicit fallback semantics."""
from __future__ import annotations

import math
import queue
import threading
import time
from pathlib import Path

from gamelens.rag.text import clean_retrieval_text

MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "ec9b3043220656d7f04860fb8aa88bb289eb3408"
CANDIDATE_LIMIT = 20
BATCH_SIZE = 2
MAX_PAIR_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 60.0
VALID_MODES = {"off", "auto", "required"}


class RerankerError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


class BGEReranker:
    def __init__(self, root, *, model_factory=None, clock=time.perf_counter,
                 timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS):
        self.root = Path(root).resolve()
        self.cache = self.root / "data/rag/models/reranker"
        self.model_factory = model_factory
        self.clock = clock
        self.timeout_seconds = timeout_seconds
        self.model = None
        self.last_load_ms = None
        self.last_inference_ms = None

    @property
    def ready(self):
        return (self.cache / f"{MODEL_REVISION}.ready").is_file()

    def load(self, download=False):
        if self.model is not None:
            return
        if not download and not self.ready:
            raise RerankerError("reranker_unavailable", "本地BGE Reranker尚未下载")
        factory = self.model_factory
        if factory is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RerankerError("reranker_unavailable", "sentence-transformers未安装") from error
            factory = CrossEncoder
        started = self.clock()
        try:
            self.model = factory(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_folder=str(self.cache),
                device="cpu",
                trust_remote_code=False,
                local_files_only=not download,
                max_length=MAX_PAIR_TOKENS,
            )
        except RerankerError:
            raise
        except Exception as error:
            raise RerankerError(
                "reranker_load_failed", f"本地BGE Reranker加载失败：{type(error).__name__}: {error}"
            ) from error
        self.last_load_ms = round((self.clock() - started) * 1000, 3)
        if download:
            from gamelens.storage.atomic_io import atomic_write_text
            self.cache.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.cache / f"{MODEL_REVISION}.ready", MODEL_ID)

    @staticmethod
    def _document(hit):
        title = str(hit.get("original", {}).get("title", ""))[:200]
        body = clean_retrieval_text(str(hit.get("text", "")))
        return f"{title}\n{body}".strip()

    def rerank(self, query, hits):
        self.load()
        candidates = list(hits)[:CANDIDATE_LIMIT]
        pairs = [(query, self._document(hit)) for hit in candidates]
        started = self.clock()
        outcome = queue.Queue(maxsize=1)

        def predict():
            try:
                outcome.put(("result", self.model.predict(
                    pairs, batch_size=BATCH_SIZE, show_progress_bar=False,
                    convert_to_numpy=True)))
            except BaseException as error:
                outcome.put(("error", error))

        worker = threading.Thread(target=predict, name="gamelens-reranker", daemon=True)
        worker.start()
        worker.join(self.timeout_seconds)
        elapsed = self.clock() - started
        self.last_inference_ms = round(elapsed * 1000, 3)
        if worker.is_alive():
            raise RerankerError("reranker_timeout", f"BGE Reranker超过{self.timeout_seconds:g}秒预算")
        kind, value = outcome.get_nowait()
        if kind == "error":
            raise RerankerError(
                "reranker_inference_failed", f"BGE Reranker推理失败：{type(value).__name__}: {value}"
            ) from value
        try:
            values = [float(score) for score in value]
            if len(values) != len(candidates) or any(not math.isfinite(score) for score in values):
                raise ValueError("返回数量或数值无效")
            ranked = [{**hit, "reranker_score": score, "hybrid_rank": rank}
                      for rank, (hit, score) in enumerate(zip(candidates, values), 1)]
            return sorted(ranked, key=lambda hit: (
                -hit["reranker_score"], hit["hybrid_rank"], hit.get("kind", ""),
                str(hit.get("source_id", "")), int(hit.get("offset", 0))))
        except (TypeError, ValueError, OverflowError) as error:
            raise RerankerError("reranker_inference_failed", f"BGE Reranker输出无效：{error}") from error


def apply_reranker(query, hits, *, mode, reranker):
    if mode not in VALID_MODES:
        raise ValueError("reranker mode必须是off、auto或required")
    original = list(hits)
    if mode == "off":
        return original, {"mode": mode, "status": "disabled", "candidate_count": len(original)}
    if not original:
        return [], {"mode": mode, "status": "no_candidates", "candidate_count": 0,
                    "model": MODEL_ID, "revision": MODEL_REVISION}
    try:
        ranked = reranker.rerank(query, original)
        return ranked, {"mode": mode, "status": "applied", "candidate_count": len(original),
                        "returned_candidates": len(ranked), "model": MODEL_ID,
                        "revision": MODEL_REVISION, "load_ms": getattr(reranker,"last_load_ms",None),
                        "inference_ms": getattr(reranker,"last_inference_ms",None)}
    except RerankerError as error:
        if mode == "required":
            raise
        return original, {"mode": mode, "status": "hybrid_fallback",
                          "reason_code": error.code, "reason": str(error),
                          "candidate_count": len(original), "model": MODEL_ID,
                          "revision": MODEL_REVISION}
