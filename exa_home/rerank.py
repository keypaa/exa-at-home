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
    def __init__(
        self,
        model="cross-encoder/ms-marco-MiniLM-L6-v2",
        timeout_ms=100,
        batch_size=128,
        device=None,
        max_pair_tokens=160,
        max_query_tokens=32,
    ):
        import torch
        from sentence_transformers import CrossEncoder

        self.timeout_ms = timeout_ms
        self.batch_size = batch_size
        self.max_pair_tokens = (
            max_pair_tokens  # truncate pairs to 128-192 tokens for budget
        )
        # The QUERY was unbounded: 200 chars of CJK spam tokenize to ~170
        # tokens and every pair pads to the longest, so one long query
        # multiplies the whole batch cost (found on-box 2026-09-15).
        self.max_query_tokens = max_query_tokens
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.model = CrossEncoder(model, device=dev)
        try:
            self.model.model.half()  # fp16: the 20-35K pairs/s figures assume it
        except Exception:  # noqa: BLE001, S110 — fp32 fallback is fine, keep quiet
            pass
        # Warmup call: first predict pays CUDA init — never pay it on a live query.
        try:
            self.model.predict([["warmup query", "warmup passage"]])
        except Exception:  # noqa: BLE001, S110 — warmup is best-effort by design
            pass

    def _score_batch(self, query, texts):
        # Truncate each pair to max_pair_tokens so the 6-10ms/200-pair budget holds.
        # Query is truncated too (it was unbounded: ~170 tokens of CJK spam
        # pads every pair in the batch via attention-quadratic cost).
        q = " ".join(query.split()[: self.max_query_tokens])
        pairs = [[q, " ".join(t.split()[: self.max_pair_tokens])] for t in texts]
        return self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        ).tolist()

    def _score(self, query, texts, timeout_ms):
        """Run _score_batch under a one-shot pool, bounded by timeout_ms.

        The pool is PER CALL and is abandoned (never joined) on timeout.
        A shared pool was the bug: one hung model call kept the single
        worker busy, so every later query queued behind it and degraded
        forever. Joining the abandoned worker had the mirror problem —
        the "fast" degraded answer still waited for the full model call.

        Returns None when the call timed out or failed.
        """
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(self._score_batch, query, texts)
            return fut.result(timeout=timeout_ms / 1000)
        except Exception:  # noqa: BLE001 — any failure degrades, never raises
            return None
        finally:
            # wait=False: do not block on the abandoned worker.
            # cancel_futures: drop anything still queued behind it.
            pool.shutdown(wait=False, cancel_futures=True)

    def rerank(self, query, candidates):
        if not candidates:
            return [], False
        scores = self._score(query, [c["text"] for c in candidates], self.timeout_ms)
        if scores is None:
            return list(candidates), True  # degraded: keep coarse ANN order
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [{**candidates[i], "score": float(scores[i])} for i in order], False
