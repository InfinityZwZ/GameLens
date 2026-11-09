"""Fixed, free local multilingual embedding; no hosted inference API."""
from pathlib import Path

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
CONFIG = {"model": MODEL_ID, "revision": MODEL_REVISION, "dimension": 1024,
          "max_seq_length": 1024, "chunk_chars": 700, "overlap_chars": 100,
          "chunk_strategy": "fixed", "text_cleaning_revision": 1, "schema": 2}


class LocalEmbedding:
    def __init__(self, root: Path):
        self.cache = root / "data" / "rag" / "models"
        self.model = None

    @property
    def ready(self):
        return (self.cache / f"{MODEL_REVISION}.ready").is_file()

    def load(self, download=False):
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("RAG依赖未安装，请执行 python -m pip install -r requirements-rag.txt") from error
        self.cache.mkdir(parents=True, exist_ok=True)
        try:
            self.model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION,
                cache_folder=str(self.cache), device="cpu", trust_remote_code=False,
                local_files_only=not download)
            self.model.max_seq_length = CONFIG["max_seq_length"]
        except Exception as error:
            raise RuntimeError(f"本地Embedding加载失败；首次需下载免费模型，可运行 python -m gamelens.rag.cli download。{type(error).__name__}: {error}") from error
        from gamelens.storage.atomic_io import atomic_write_text
        atomic_write_text(self.cache / f"{MODEL_REVISION}.ready", MODEL_ID)

    def encode(self, texts):
        self.load()
        vectors = self.model.encode(texts, batch_size=4, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    def token_count(self, text):
        self.load()
        encoded = self.model.tokenizer(text, truncation=False, add_special_tokens=True)
        return len(encoded["input_ids"])
