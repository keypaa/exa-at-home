# tests/test_orch.py
"""Task 10: Canon-lite DAG orchestrator + profile spans. Tier 0.

Offline note: the real ANN backend needs `maturin develop` (no sandbox
egress), so the search-waterfall test uses FAKE backends with the exact
method shapes (embedder.encode_queries / ann.search / reranker.rerank /
store.get). Integration against the real IvfIndex needs on-box pytest.
"""
from exa_home.orch import DAG, Node, build_search_dag, run_search


def test_dag_runs_in_dependency_order_and_memos():
    calls = []

    def a_fn(_):
        calls.append("a")
        return 1

    def b_fn(x):
        calls.append("b")
        return x["a"] + 1

    def c_fn(x):
        calls.append("c")
        return (x["a"], x["b"])

    dag = DAG([Node("a", a_fn, []), Node("b", b_fn, ["a"]),
               Node("c", c_fn, ["a", "b"])])
    out = dag.run({})
    assert out["outputs"]["c"] == (1, 2) and calls == ["a", "b", "c"]


def test_failure_cancels_dependents_keeps_spans():
    def bad(_):
        raise RuntimeError("boom")

    dag = DAG([Node("a", lambda _: 1, []), Node("b", bad, ["a"]),
               Node("c", lambda _: 2, ["b"])])
    out = dag.run({})
    assert out["error"]["node"] == "b" \
        and "a" in out["spans"] and "c" not in out["outputs"]


# --- Fake backends (offline stand-ins with the real method shapes) ---

class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def encode_queries(self, texts):
        self.calls.extend(texts)
        return [[float(len(t))] for t in texts]  # 1-d "embedding"


class FakeANN:
    def __init__(self, ids=("d0", "d1", "d2")):
        self.ids = list(ids)
        self.seen_kwargs = []

    def search(self, q, nprobe=8, top_k=200, **filters):
        self.seen_kwargs.append(dict(filters))
        ids = self.ids[:top_k]
        return ids, [1.0 - 0.1 * i for i in range(len(ids))]


class FakeReranker:
    """Success path: sorts by coarse desc + attaches score (real shape)."""

    def __init__(self, degraded=False):
        self.degraded = degraded

    def rerank(self, query, candidates):
        if self.degraded:  # degraded items LACK "score" (Task 9 contract)
            return list(candidates), True
        ranked = sorted(candidates, key=lambda c: c["coarse_score"],
                        reverse=True)
        return [{**c, "score": c["coarse_score"] + 0.01} for c in ranked], False


class FakeStore:
    def __init__(self, docs):
        self.docs = docs

    def get(self, doc_id):
        return self.docs.get(doc_id)


def _docs():
    return {f"d{i}": {"id": f"d{i}",
                      "text": f"passage {i} about neural search " * 20}
            for i in range(3)}


def test_search_waterfall_sums_to_wall():
    dag = build_search_dag(FakeEmbedder(), FakeANN(), FakeReranker(),
                           FakeStore(_docs()))
    res = run_search(dag, "topic 3", {}, top_k=5, profile=True)
    assert abs(sum(v for k, v in res["profile"].items() if k != "total_ms")
               - res["profile"]["total_ms"]) \
        < res["profile"]["total_ms"] * 0.2 + 5
    assert [r["id"] for r in res["results"]] == ["d0", "d1", "d2"]
    assert res["degraded"] is False
    assert all("score" in r and "snippet" in r for r in res["results"])
    assert all(len(r["snippet"]) <= 300 for r in res["results"])


def test_filters_pass_straight_through_to_ann():
    ann = FakeANN()
    dag = build_search_dag(FakeEmbedder(), ann, FakeReranker(),
                           FakeStore(_docs()))
    filters = {"domains": ["example.com"], "terms": ["alpha", "beta"]}
    res = run_search(dag, "q", filters, top_k=5)
    assert ann.seen_kwargs and ann.seen_kwargs[0] == filters
    assert res["results"]  # non-empty: orch only renames, never interprets


def test_spec_shape_filters_translate_to_rust_kwargs():
    from exa_home.orch import translate_filters
    assert translate_filters({"date_range": ("2026-01", "2026-08"),
                              "keywords": ["a"],
                              "domains": ["example.com"]}) == {
        "month_range": ("2026-01", "2026-08"),
        "terms": ["a"],
        "domains": ["example.com"]}
    import pytest
    with pytest.raises(ValueError, match="unknown filter key"):
        translate_filters({"bogus": 1})


def test_degraded_rerank_has_uniform_score_shape():
    dag = build_search_dag(FakeEmbedder(), FakeANN(),
                           FakeReranker(degraded=True), FakeStore(_docs()))
    res = run_search(dag, "q", {}, top_k=5)
    assert res["degraded"] is True
    assert res["results"]
    # Uniform shape: every result carries a float score, never missing.
    assert all(isinstance(r["score"], float) for r in res["results"])


def test_node_timeout_cancels_dependents():
    import time

    def slow(_):
        time.sleep(0.5)
        return 1

    dag = DAG([Node("a", slow, [], timeout_ms=50),
               Node("b", lambda _: 2, ["a"])])
    out = dag.run({})
    assert out["error"]["node"] == "a"
    assert "Timeout" in out["error"]["type"]
    assert "b" not in out["outputs"]


def test_run_search_reports_node_error_without_raising():
    def bad(_):
        raise RuntimeError("ann down")

    dag = DAG([Node("embed", lambda _: [0.0], []),
               Node("retrieve", bad, ["embed"]),
               Node("rerank", lambda x: x, ["retrieve"]),
               Node("snippet", lambda x: x, ["rerank"])])
    res = run_search(dag, "q", {}, top_k=5, profile=True)
    assert res["results"] == [] and res["degraded"] is True
    assert res["error"]["node"] == "retrieve"
    assert "total_ms" in res["profile"]


def test_dag_rejects_cycles_and_unknown_deps():
    import pytest
    with pytest.raises(ValueError, match="cycle"):
        DAG([Node("a", lambda _: 1, ["b"]), Node("b", lambda _: 1, ["a"])])
    with pytest.raises(ValueError, match="unknown"):
        DAG([Node("a", lambda _: 1, ["nope"])])
