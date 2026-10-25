# exa_home/rerank.py
"""M5 cross-encoder reranker + degraded fallback.

Tier-0 safe: sentence_transformers/torch are lazy-imported inside
Reranker.__init__ (Task 3 pattern), so importing this module never
touches GPU deps. ReversingReranker + the injectable-_score_batch timeout
test run offline; the real model needs the GPU box + HF download.
"""

from __future__ import annotations

import concurrent.futures

from .config import OPERATING_POINT, RERANKER_MODEL


class ReversingReranker:
    """Deterministic stub whose behaviour is to **reverse** the input.

    Deliberately *not* a neutral no-op: ordering becomes visible in tests,
    and it is impossible to ship a silent no-reorder by accident. Renamed
    from ``MockReranker`` (2026-10-24) after the old name hid the
    inversion and produced a published 10.7x "lift" (EXPERIMENTS.md, M5).
    """

    def rerank(self, query, candidates):
        return list(reversed(candidates)), False


# Back-compat alias for tests and notebooks imported under the old name.
MockReranker = ReversingReranker


class Reranker:
    def __init__(
        self,
        model=RERANKER_MODEL,
        timeout_ms=100,
        batch_size=OPERATING_POINT["batch_size"],
        device=None,
        max_pair_tokens=OPERATING_POINT["max_pair_tokens"],
        max_query_tokens=OPERATING_POINT["max_query_tokens"],
    ):
        import torch
        from sentence_transformers import CrossEncoder

        self.timeout_ms = timeout_ms
        self.batch_size = batch_size
        self.max_pair_tokens = max_pair_tokens
        self.max_query_tokens = max_query_tokens
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.model = CrossEncoder(model, device=dev)
        try:
            self.model.model.half()
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            self.model.predict([["warmup query", "warmup passage"]])
        except Exception:  # noqa: BLE001, S110
            pass

    def _score_batch(self, query, texts):
        q = " ".join(query.split()[:self.max_query_tokens])
        pairs = [[q, " ".join(t.split()[:self.max_pair_tokens])] for t in texts]
        return self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        ).tolist()

    def _score(self, query, texts, timeout_ms):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(self._score_batch, query, texts)
            return fut.result(timeout=timeout_ms / 1000)
        except Exception:  # noqa: BLE001
            return None
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def rerank(self, query, candidates):
        if not candidates:
            return [], False
        scores = self._score(query, [c["text"] for c in candidates], self.timeout_ms)
        if scores is None:
            return list(candidates), True
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [{**candidates[i], "score": float(scores[i])} for i in order], False
