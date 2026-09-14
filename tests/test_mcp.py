# tests/test_mcp.py — Task 11: MCP home_search + home_contents contract tests.
#
# On-box (real FastMCP): exercises the decorated tool functions directly
# (functions, not transport — FastMCP tools are plain functions under the
# decorator). Offline (no fastmcp in sandbox): skips cleanly; the fastmcp-free
# mirror suite in tests/test_mcp_logic.py covers the same logic via a stub app.
import pytest

pytest.importorskip("fastmcp",
                    reason="fastmcp not installed offline; on-box only")

from exa_home.mcp_server import app, home_contents, home_search, wire


# --- Stub backends (same method shapes as the real ones; cf. test_orch.py) ---

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
    """Success path attaches score; degraded path omits it (Task 9 contract)."""

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


def test_home_search_contract():
    _wire()
    res = home_search("neural search")
    assert set(res.keys()) >= {"results"}
    assert res["results"], "stub backends must yield non-empty results"
    assert all(set(r) >= {"id", "url", "title", "snippet", "score"}
               for r in res["results"])
    assert all(isinstance(r["score"], float) for r in res["results"])
    assert res["degraded"] is False


def test_home_search_rejects_empty_query():
    _wire()
    with pytest.raises(ValueError):
        home_search("")
    with pytest.raises(ValueError):
        home_search("   ")


def test_home_search_forwards_filters_verbatim():
    ann = _wire()
    filters = {"terms": ["alpha", "beta"], "domains": ["example.com"]}
    res = home_search("q", filters=filters)
    assert ann.seen_kwargs and ann.seen_kwargs[0] == filters
    assert res["results"]


def test_home_search_top_k_and_profile():
    _wire(n=5)
    res = home_search("q", top_k=2, profile=True)
    assert len(res["results"]) == 2
    assert "total_ms" in res["profile"]


def test_home_search_degraded_passthrough():
    _wire(degraded=True)
    res = home_search("q")
    assert res["degraded"] is True
    assert res["results"]
    assert all(isinstance(r["score"], float) for r in res["results"])


def test_home_contents_paginates():
    _wire()
    ids = [f"d{i}" for i in range(3)] + [f"missing-{i}" for i in range(22)]
    res = home_contents(ids)
    assert len(res["items"]) <= 20 and "next_offset" in res
    assert res["remaining"] == len(ids) - 20
    # Caller pages with the offset — never silent truncation.
    res2 = home_contents(ids[res["next_offset"]:])
    assert len(res2["items"]) == len(ids) - 20
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
