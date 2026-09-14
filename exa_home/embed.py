# exa_home/embed.py
from __future__ import annotations
import hashlib
import numpy as np

# mxbai-embed-large-v1 prompt contract (verified on card: query prompt is built
# into the ST config as prompt_name="query"; docs encode unprefixed).
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

def get_embedder(model: str = "mixedbread-ai/mxbai-embed-large-v1", dim: int = 256):
    """Persistent singleton. First call loads GPU model; never call per-query cold."""
    global _singleton
    if _singleton is None:
        _singleton = Embedder(model, dim)
    return _singleton

class Embedder:
    def __init__(self, model: str = "mixedbread-ai/mxbai-embed-large-v1", dim: int = 256):
        from sentence_transformers import SentenceTransformer
        self.dim = dim
        # mxbai-large-v1: plain BERT, no custom modeling code, no
        # trust_remote_code, no xformers (vendor: "No fancy custom code or
        # trust remote code required"). Chosen 2026-09-14 after Arctic-m-v2.0
        # proved unloadable (hard xformers assert in its custom modeling
        # file). truncate_dim=256 uses the model's MRL support; 256-dim
        # quality is MEASURED by our recall harness, not vendor-claimed.
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
