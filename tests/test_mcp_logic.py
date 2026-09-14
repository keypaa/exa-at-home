# tests/test_mcp_logic.py — Task 11 fastmcp-free mirror suite (Tier 0, offline).
#
# mcp_server.py imports fastmcp at module top per the brief, so it cannot be
# imported where fastmcp is absent. This module installs a minimal stub
# `fastmcp` (FastMCP with identity @tool decorator) ONLY when the real
# package is missing, then exercises the REAL mcp_server code: wire(),
# validation, url/title enrichment, filter passthrough, pagination.
# On-box (real fastmcp present) the stub is skipped and the same tests run
# against the real FastMCP app object.
import sys
import types

import pytest

try:
    import fastmcp  # noqa: F401
except ModuleNotFoundError:
    _stub = types.ModuleType("fastmcp")

    class FastMCP:
        def __init__(self, name):
            self.name = name

        def tool(self, fn=None, **kwargs):
            if fn is None:
                return lambda f: f
            return fn

        def run(self, *args, **kwargs):
            raise RuntimeError("stub FastMCP cannot run a transport")

    _stub.FastMCP = FastMCP
    sys.modules["fastmcp"] = _stub

from exa_home.mcp_server import app, home_contents, home_search, wire


class FakeEmbedder:
    def encode_queries(self, texts):
        return [[float(len(t))] for t in texts]


class FakeANN:
    def __init__(self, ids):
        self.ids = list(ids)
        self.seen_kwargs = []

    def search(self, q, nprobe=8, top_k=200, **filters):
        self.seen_kwargs.append(dict(filters))
        ids = self.ids[:top_k]
        return ids, [1.0 - 0.1 * i for i in range(len(ids))]


class FakeReranker:
    def __init__(self, degraded=False):
        self.degraded = degraded

    def rerank(self, query, candidates):
        if self.degraded:
            return list(candidates), True
        ranked = sorted(candidates, key=lambda c: c["coarse_score"],
                        reverse=True)
        return [{**c, "score": c["coarse_score"] + 0.01} for c in ranked], False


class FakeStore:
    def __init__(self, docs):
        self.docs = docs

    def get(self, doc_id):
        return self.docs.get(doc_id)


def _docs(n=3):
    return {f"d{i}": {"id": f"d{i}",
                      "url": f"https://example.com/{i}",
                      "title": f"Title {i}",
                      "text": f"passage {i} about neural search " * 20}
            for i in range(n)}


def _wire(n=3, degraded=False):
    ann = FakeANN([f"d{i}" for i in range(n)])
    wire(FakeEmbedder(), ann, FakeReranker(degraded=degraded),
         FakeStore(_docs(n)))
    return ann


def test_app_is_exa_home():
    assert app.name == "exa-home"


def test_wire_stores_all_four_backends():
    from exa_home import mcp_server
    e, a, r, s = object(), object(), object(), object()
    wire(e, a, r, s)
    assert (mcp_server._deps["embedder"], mcp_server._deps["ann"],
            mcp_server._deps["reranker"], mcp_server._deps["store"]) == (e, a, r, s)


def test_home_search_contract_with_enrichment():
    _wire()
    res = home_search("neural search")
    assert set(res.keys()) >= {"results"}
    assert res["results"]
    assert all(set(r) >= {"id", "url", "title", "snippet", "score"}
               for r in res["results"])
    assert all(isinstance(r["score"], float) for r in res["results"])
    assert res["results"][0]["url"] == "https://example.com/0"
    assert res["results"][0]["title"] == "Title 0"
    assert res["degraded"] is False


def test_home_search_rejects_empty_query():
    _wire()
    with pytest.raises(ValueError):
        home_search("")
    with pytest.raises(ValueError):
        home_search("   ")


def test_home_search_forwards_filters_verbatim():
    ann = _wire()
    filters = {"terms": ["alpha"], "domains": ["example.com"],
               "month_range": ["2026-01", "2026-08"], "allow": ["d0"]}
    res = home_search("q", filters=filters)
    assert ann.seen_kwargs and ann.seen_kwargs[0] == filters
    assert res["results"]


def test_home_search_arg_order_matches_orch_contract():
    import inspect
    from exa_home import orch
    assert list(inspect.signature(orch.run_search).parameters) == [
        "dag", "query", "filters", "top_k", "profile"]


def test_home_search_top_k_profile_and_degraded():
    _wire(n=5)
    res = home_search("q", top_k=2, profile=True)
    assert len(res["results"]) == 2
    assert "total_ms" in res["profile"]
    _wire(n=3, degraded=True)
    res = home_search("q")
    assert res["degraded"] is True
    assert all(isinstance(r["score"], float) for r in res["results"])


def test_home_contents_paginates_never_silent():
    _wire()
    ids = [f"d{i}" for i in range(3)] + [f"missing-{i}" for i in range(22)]
    res = home_contents(ids)
    assert len(res["items"]) == 20
    assert res["next_offset"] == 20 and res["remaining"] == 5
    res2 = home_contents(ids[res["next_offset"]:])
    assert len(res2["items"]) == 5
    assert "next_offset" not in res2


def test_home_contents_rejects_empty_ids():
    _wire()
    with pytest.raises(ValueError):
        home_contents([])


def test_home_contents_not_found_and_truncation():
    _wire()
    res = home_contents(["d0", "nope"], max_chars_per_doc=10)
    by_id = {i["id"]: i for i in res["items"]}
    assert by_id["nope"] == {"id": "nope", "error": "not_found"}
    assert by_id["d0"]["url"] == "https://example.com/0"
    assert len(by_id["d0"]["text"]) <= 10
