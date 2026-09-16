# exa_home/rerank.py
"""M5 cross-encoder reranker + degraded fallback.

Tier-0 safe: sentence_transformers/torch are lazy-imported inside
Reranker.__init__ (Task 3 pattern), so importing this module never
touches GPU deps. MockReranker + the injectable-_score_batch timeout
test run offline; the real model needs the GPU box + HF download.
"""
from __future__ import annotations
import concurrent.futures


class MockReranker:
    """Deterministic stub: reverses input, never degraded.

    Proves ordering flows through the orchestrator (Task 10) without a model.
    """

    def rerank(self, query, candidates):
        return list(reversed(candidates)), False


class Reranker:
    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L6-v2",
                 timeout_ms=100, batch_size=128, device=None, max_pair_tokens=160,
                 max_query_tokens=32):
        from sentence_transformers import CrossEncoder
        import torch
        self.timeout_ms = timeout_ms
        self.batch_size = batch_size
        self.max_pair_tokens = max_pair_tokens  # truncate pairs to 128-192 tokens for budget
        # The QUERY was unbounded: 200 chars of CJK spam tokenize to ~170
        # tokens and every pair pads to the longest, so one long query
        # multiplies the whole batch cost (found on-box 2026-09-15).
        self.max_query_tokens = max_query_tokens
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.model = CrossEncoder(model, device=dev)
        try:
            self.model.model.half()  # fp16: the 20-35K pairs/s figures assume it
        except Exception:
            pass
        # Warmup call: first predict pays CUDA init — never pay it on a live query.
        try:
            self.model.predict([["warmup query", "warmup passage"]])
        except Exception:
            pass
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _score_batch(self, query, texts):
        # Truncate each pair to max_pair_tokens so the 6-10ms/200-pair budget holds.
        # Query is truncated too (it was unbounded: ~170 tokens of CJK spam
        # pads every pair in the batch via attention-quadratic cost).
        q = " ".join(query.split()[: self.max_query_tokens])
        pairs = [[q, " ".join(t.split()[: self.max_pair_tokens])] for t in texts]
        return self.model.predict(pairs, batch_size=self.batch_size,
                                  show_progress_bar=False).tolist()

    def rerank(self, query, candidates):
        if not candidates:
            return [], False
        pool = getattr(self, "_pool", None)
        if pool is None:
            # Testability path: Reranker.__new__ without __init__ (no model
            # load). Run the injectable _score_batch under a one-shot pool so
            # the timeout contract still applies.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as tmp:
                fut = tmp.submit(self._score_batch, query, [c["text"] for c in candidates])
                try:
                    scores = fut.result(timeout=self.timeout_ms / 1000)
                except Exception:
                    return list(candidates), True  # degraded: keep coarse ANN order
        else:
            fut = pool.submit(self._score_batch, query, [c["text"] for c in candidates])
            try:
                scores = fut.result(timeout=self.timeout_ms / 1000)
            except Exception:
                return list(candidates), True  # degraded: keep coarse ANN order
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [{**candidates[i], "score": float(scores[i])} for i in order], False
