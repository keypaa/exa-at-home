# exa_home/embed.py
from __future__ import annotations
import hashlib
import numpy as np

# Arctic-m-v2.0 prompt contract (VERIFY ON BOX: confirm doc side needs no prefix).
# ST usage: model.encode(texts, prompt_name="query") for queries, plain for docs.
QUERY_KWARGS = {"prompt_name": "query"}
DOC_KWARGS = {}

class HashEmbedder:
    """Deterministic stub for Tier 0/1. Same interface, no model."""
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.normal(size=self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one("query: " + t) for t in texts])

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts])

_singleton = None

def get_embedder(model: str = "Snowflake/snowflake-arctic-embed-m-v2.0", dim: int = 256):
    """Persistent singleton. First call loads GPU model; never call per-query cold."""
    global _singleton
    if _singleton is None:
        _singleton = Embedder(model, dim)
    return _singleton

class Embedder:
    def __init__(self, model: str = "Snowflake/snowflake-arctic-embed-m-v2.0", dim: int = 256):
        from sentence_transformers import SentenceTransformer
        self.dim = dim
        # Native ST model: no trust_remote_code unless on-box load proves otherwise.
        # truncate_dim=256 maps to the model's native two-stage MRL-256 point.
        self.model = SentenceTransformer(model, truncate_dim=dim)

    def _encode(self, texts: list[str], **prompt_kwargs) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True,
                              **prompt_kwargs)
        return v[:, : self.dim].astype(np.float32) / (
            np.linalg.norm(v[:, : self.dim], axis=1, keepdims=True) + 1e-12)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, **QUERY_KWARGS)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, **DOC_KWARGS)
