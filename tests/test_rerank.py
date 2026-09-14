# tests/test_rerank.py
import time

import pytest

from exa_home.rerank import MockReranker


def test_mock_reranker_reorders():
    r = MockReranker()
    cands = [{"id": "a", "text": "t1"}, {"id": "b", "text": "t2"}]
    ranked, degraded = r.rerank("q", cands)
    assert [c["id"] for c in ranked] == ["b", "a"] and degraded is False


def test_slow_model_falls_back_to_coarse_order():
    from exa_home.rerank import Reranker
    r = Reranker.__new__(Reranker)  # no model load
    r.timeout_ms = 50
    r._score_batch = lambda q, texts: (time.sleep(0.5), [0.0] * len(texts))[1]
    cands = [{"id": "a", "text": "t1"}, {"id": "b", "text": "t2"}]
    ranked, degraded = r.rerank("q", cands)
    assert [c["id"] for c in ranked] == ["a", "b"] and degraded is True


def test_empty_candidates_returns_empty_not_degraded():
    from exa_home.rerank import Reranker
    r = Reranker.__new__(Reranker)
    r.timeout_ms = 50
    ranked, degraded = r.rerank("q", [])
    assert ranked == [] and degraded is False


def test_fast_injected_scorer_sorts_desc():
    from exa_home.rerank import Reranker
    r = Reranker.__new__(Reranker)
    r.timeout_ms = 1000
    r._score_batch = lambda q, texts: [1.0 if "good" in t else 0.0 for t in texts]
    cands = [{"id": "a", "text": "bad"}, {"id": "b", "text": "good stuff"}]
    ranked, degraded = r.rerank("q", cands)
    assert [c["id"] for c in ranked] == ["b", "a"] and degraded is False
    assert ranked[0]["score"] > ranked[1]["score"]


@pytest.mark.cloud_only
def test_real_model_200_pairs_under_40ms():
    """GPU box only: MiniLM-L6-v2, batch 128 fp16, 200 pairs < 40ms.

    First run downloads ~90MB from HuggingFace (no egress in sandbox).
    """
    import torch
    from exa_home.rerank import Reranker
    assert torch.cuda.is_available(), "rerank budget test needs a CUDA GPU"
    r = Reranker()
    cands = [{"id": str(i), "text": f"passage number {i} about neural search"} for i in range(200)]
    t0 = time.perf_counter()
    ranked, degraded = r.rerank("what is neural search?", cands)
    dt_ms = (time.perf_counter() - t0) * 1000
    assert degraded is False
    assert len(ranked) == 200
    assert dt_ms < 40, f"200-pair rerank took {dt_ms:.1f}ms, budget is 40ms"
