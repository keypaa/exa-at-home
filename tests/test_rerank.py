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


def test_degraded_path_returns_within_the_timeout_budget():
    """BUG: the degraded path returned from INSIDE the `with pool` block,
    so __exit__ joined the still-running model call — the caller waited
    the full model time for a "fast" degraded answer."""
    from exa_home.rerank import Reranker

    r = Reranker.__new__(Reranker)  # no model load
    r.timeout_ms = 100
    r._score_batch = lambda q, texts: (time.sleep(1.0), [0.0] * len(texts))[1]
    cands = [{"id": "a", "text": "t1"}, {"id": "b", "text": "t2"}]
    t0 = time.perf_counter()
    _ranked, degraded = r.rerank("q", cands)
    dt_ms = (time.perf_counter() - t0) * 1000
    assert degraded is True
    assert dt_ms < 500, f"degraded rerank waited {dt_ms:.0f}ms (budget 100ms)"


def test_timed_out_call_does_not_poison_later_calls():
    """BUG: one hung model call stayed on the shared single-worker pool,
    so every later query queued behind it and degraded forever."""
    from exa_home.rerank import Reranker

    r = Reranker.__new__(Reranker)  # no model load
    r.timeout_ms = 100
    calls = {"n": 0}

    def scorer(query, texts):
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(1.0)  # hangs past the timeout budget
        return [float(len(texts))] * len(texts)

    r._score_batch = scorer
    cands = [{"id": "a", "text": "t1"}, {"id": "b", "text": "t2"}]

    t0 = time.perf_counter()
    ranked, degraded = r.rerank("q", cands)
    first_ms = (time.perf_counter() - t0) * 1000
    assert degraded is True and first_ms < 500

    # The next query must NOT queue behind the abandoned call.
    t0 = time.perf_counter()
    ranked, degraded = r.rerank("q", cands)
    second_ms = (time.perf_counter() - t0) * 1000
    assert degraded is False, "a hung call poisoned the next query"
    assert second_ms < 500, f"second rerank waited {second_ms:.0f}ms"
    assert [c["id"] for c in ranked] == ["a", "b"]


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
    cands = [
        {"id": str(i), "text": f"passage number {i} about neural search"}
        for i in range(200)
    ]
    t0 = time.perf_counter()
    ranked, degraded = r.rerank("what is neural search?", cands)
    dt_ms = (time.perf_counter() - t0) * 1000
    assert degraded is False
    assert len(ranked) == 200
    assert dt_ms < 40, f"200-pair rerank took {dt_ms:.1f}ms, budget is 40ms"


def test_reranker_pool_is_created_once_under_concurrency():
    import threading

    from exa_home.rerank import Reranker

    r = Reranker.__new__(Reranker)
    pools = []
    barrier = threading.Barrier(16)

    def grab():
        barrier.wait()
        pools.append(r._executor())

    threads = [threading.Thread(target=grab) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(p) for p in pools}) == 1, "concurrent callers built multiple pools"
    r.close()


def test_reranker_discarding_a_timed_out_pool_does_not_cancel_queued_work():
    """The reranker is a process-wide singleton; one timeout must not
    cancel a score another thread already submitted."""
    import threading

    from exa_home.rerank import Reranker

    r = Reranker.__new__(Reranker)
    pool = r._executor()
    gate = threading.Event()

    def queued():
        gate.wait(2)
        return "ran"

    fut = pool.submit(queued)
    r._discard_pool(pool)
    gate.set()
    assert fut.result(timeout=2) == "ran"
    assert getattr(r, "_pool", None) is None
    r.close()


def test_reranker_close_is_idempotent_and_safe_after_partial_init():
    from exa_home.rerank import Reranker

    Reranker.__new__(Reranker).close()
    r = Reranker.__new__(Reranker)
    r.close()
    r.close()
