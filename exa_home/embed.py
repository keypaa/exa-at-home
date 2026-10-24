# exa_home/embed.py
from __future__ import annotations

import hashlib

import numpy as np

from .config import DIM, EMBED_BATCH, EMBEDDER_MODEL

# mxbai-embed-large-v1 prompt contract (verified on card: query prompt is built
# into the ST config as prompt_name="query"; docs encode unprefixed).
# ST usage: model.encode(texts, prompt_name="query") for queries, plain for docs.
QUERY_KWARGS = {"prompt_name": "query"}
DOC_KWARGS = {}


def length_sorted_order(lengths: list[int] | np.ndarray) -> np.ndarray:
    """Indices that sort `lengths` ascending, STABLY (ties keep input
    order — the restored output must be exact)."""
    return np.argsort(np.asarray(lengths), kind="stable")


class HashEmbedder:
    """Deterministic stub for Tier 0/1. Same interface, no model."""

    def __init__(self, dim: int = DIM):
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


class Embedder:
    def __init__(
        self,
        model: str = EMBEDDER_MODEL,
        dim: int = DIM,
        device: str | None = None,
        max_seq_length: int | None = None,
        batch_size: int | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.dim = dim
        # mxbai-large-v1: plain BERT, no custom modeling code, no
        # trust_remote_code, no xformers (vendor: "No fancy custom code or
        # trust remote code required"). Chosen 2026-09-14 after Arctic-m-v2.0
        # proved unloadable (hard xformers assert in its custom modeling
        # file). truncate_dim=256 uses the model's MRL support; 256-dim
        # quality is MEASURED by our recall harness, not vendor-claimed.
        self.device = device or ("cuda" if self._is_cuda() else "cpu")
        self.model = SentenceTransformer(
            model, truncate_dim=dim, device=self.device
        )
        # The model's own max_seq_length (512) is what actually gets
        # embedded — a `text[:12000]` slice upstream is a no-op beyond
        # ~2000 characters. Set it explicitly so the number is visible
        # rather than inherited from the model card.
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        self.max_seq_length = int(self.model.max_seq_length)
        # Half precision on CUDA: the single biggest lever (tensor cores).
        # sentence-transformers 6.0.1 does NOT accept precision='float16'
        # (its encode() allows only float32/int8/uint8/binary/ubinary), so
        # fp16 must be achieved by casting the weights — the forward pass
        # then runs in half without autocast, and _encode() casts back to
        # float32 for storage.
        if self._is_cuda():
            self.model.half()
        self.precision = "float16" if self._is_cuda() else "float32"
        self.batch_size = batch_size or EMBED_BATCH

    def _is_cuda(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 — torch missing/broken means CPU
            return False

    def _token_lengths(self, texts: list[str]) -> np.ndarray:
        """Token length of each text (CPU, tokenizer only)."""
        enc = self.model.tokenizer(
            texts, add_special_tokens=True, truncation=True, return_length=True
        )
        return np.asarray(enc["length"], dtype=np.int64)

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        v = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return v[:, : self.dim].astype(np.float32) / (
            np.linalg.norm(v[:, : self.dim], axis=1, keepdims=True) + 1e-12
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """One query at a time (serving path): no bucketing, no sorting —
        the extra work would only add latency."""
        return self._encode(texts, batch_size=1)

    def encode_docs(
        self,
        texts: list[str],
        batch_size: int | None = None,
        sort_by_length: bool = True,
    ) -> np.ndarray:
        """Bulk document encoding (offline path).

        Length bucketing: pages vary hugely in length, and every batch is
        padded to its longest member. Sorting by length cuts that waste.
        The OUTPUT is restored to the input order — a silent reorder would
        corrupt every vector (and the index built from them).
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        bs = batch_size or self.batch_size
        if not sort_by_length or len(texts) < 2:
            return self._encode(texts, batch_size=bs)
        order = length_sorted_order(self._token_lengths(texts))
        sorted_texts = [texts[i] for i in order]
        v_sorted = self._encode(sorted_texts, batch_size=bs)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return v_sorted[inverse]
