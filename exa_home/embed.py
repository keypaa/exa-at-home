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
        self.device = device or ("cuda" if self._is_cuda() else "cpu")
        self.model = SentenceTransformer(model, truncate_dim=dim, device=self.device)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        self.max_seq_length = int(self.model.max_seq_length)
        # Half precision on CUDA: the single biggest lever (tensor cores).
        if self._is_cuda():
            self.model.half()
        self.precision = "float16" if self._is_cuda() else "float32"
        self.batch_size = batch_size or EMBED_BATCH
        # Production path tokenizes once and drives the model manually (item 1b).
        # Tier-0 stubs created via __new__ lack this flag and use the legacy
        # SentenceTransformer.encode path so their mocks keep working.
        self._manual = True

    def _is_cuda(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 — torch missing/broken means CPU
            return False

    def _token_lengths(self, texts: list[str]) -> np.ndarray:
        enc = self.model.tokenizer(
            texts, add_special_tokens=True, truncation=True, return_length=True
        )
        return np.asarray(enc["length"], dtype=np.int64)

    def _encode_legacy(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Tier-0/stub path: delegates to SentenceTransformer.encode."""
        v = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        v = np.asarray(v, dtype=np.float32)[:, : self.dim]
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not getattr(self, "_manual", False):
            return self._encode_legacy(texts, batch_size)
        import torch

        # Tokenize ONCE (the lever from item 1b: item 1b measured 50.4% of
        # wall time spent re-running the tokenizer inside model.encode).
        features = self.model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        dev = self.device
        out_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                end = start + batch_size
                batch = {k: v[start:end].to(dev) for k, v in features.items()}
                res = self.model(batch)
                out_chunks.append(res["sentence_embedding"].detach().cpu())
        arr = torch.cat(out_chunks, dim=0).numpy().astype(np.float32)
        arr = arr[:, : self.dim]
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return arr

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, batch_size=1)

    def _encode_docs_manual(
        self,
        texts: list[str],
        batch_size: int,
        sort_by_length: bool,
    ) -> np.ndarray:
        """Manual path: tokenize ONCE without padding, sort by token length,
        then pad each batch to its own longest member.

        The old path tokenized every chunk twice: once in `_token_lengths`
        for sorting, then again in `_encode` with `padding=True` across the
        whole 8192-doc chunk. That second pass also padded every document to
        the chunk maximum (often 512), even when most were short. This path
        keeps the same vectors (attention masks exclude padding) but removes
        the duplicate tokenization and the chunk-wide padding.
        """
        import torch

        enc = self.model.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
        )
        ids = enc["input_ids"]
        lengths = [len(x) for x in ids]
        if sort_by_length and len(ids) > 1:
            order = length_sorted_order(lengths)
        else:
            order = np.arange(len(ids))

        pad_id = self.model.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        has_token_type = "token_type_ids" in enc
        out_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                max_len = max(lengths[i] for i in idx)
                input_ids = torch.full((len(idx), max_len), pad_id, dtype=torch.long)
                attention_mask = torch.zeros((len(idx), max_len), dtype=torch.long)
                for row, i in enumerate(idx):
                    n = lengths[i]
                    input_ids[row, :n] = torch.tensor(ids[i], dtype=torch.long)
                    attention_mask[row, :n] = 1
                batch = {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                }
                if has_token_type:
                    batch["token_type_ids"] = torch.zeros_like(input_ids).to(
                        self.device
                    )
                res = self.model(batch)
                out_chunks.append(res["sentence_embedding"].detach().cpu())

        arr = torch.cat(out_chunks, dim=0).numpy().astype(np.float32)
        arr = arr[:, : self.dim]
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        if sort_by_length and len(order) > 1:
            inverse = np.empty_like(order)
            inverse[order] = np.arange(len(order))
            arr = arr[inverse]
        return arr

    def encode_docs(
        self,
        texts: list[str],
        batch_size: int | None = None,
        sort_by_length: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        bs = batch_size or self.batch_size
        if getattr(self, "_manual", False):
            return self._encode_docs_manual(texts, bs, sort_by_length)
        # Tier-0/stub path: SentenceTransformer.encode does the tokenization.
        if not sort_by_length or len(texts) < 2:
            return self._encode(texts, batch_size=bs)
        order = length_sorted_order(self._token_lengths(texts))
        sorted_texts = [texts[i] for i in order]
        v_sorted = self._encode(sorted_texts, batch_size=bs)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return v_sorted[inverse]
