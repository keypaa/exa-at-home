# tests/test_mcp.py — the MCP tool surface (Tier 0, offline).
#
# The tool functions are module-level plain functions in exa_home.mcp_server
# (fastmcp is imported lazily by the app builder), so these tests need
# neither fastmcp nor a sys.modules stub — the stub dance is what made
# this file exist twice.
import pytest
from fixtures.backends import (
    wire_mcp,
)

from exa_home.mcp_server import home_contents, home_search, wire


def test_app_is_exa_home():
    from exa_home.mcp_server import _get_app

    assert _get_app().name == "exa-home"


def test_wire_stores_all_four_backends():
    from exa_home import mcp_server

    e, a, r, s = object(), object(), object(), object()
    wire(e, a, r, s)
    assert (
        mcp_server._deps["embedder"],
        mcp_server._deps["ann"],
        mcp_server._deps["reranker"],
        mcp_server._deps["store"],
    ) == (e, a, r, s)


def test_home_search_contract_with_enrichment():
    wire_mcp()
    res = home_search("neural search")
    assert set(res.keys()) >= {"results"}
    assert res["results"]
    assert all(
        set(r) >= {"id", "url", "title", "snippet", "score"} for r in res["results"]
    )
    assert all(isinstance(r["score"], float) for r in res["results"])
    assert res["results"][0]["url"] == "https://example.com/0"
    assert res["results"][0]["title"] == "Title 0"
    assert res["degraded"] is False


def test_home_search_rejects_empty_query():
    wire_mcp()
    with pytest.raises(ValueError):
        home_search("")
    with pytest.raises(ValueError):
        home_search("   ")


def test_home_search_forwards_rust_shape_filters():
    # Rust IvfIndex.search shapes: month_range is a tuple, allow is
    # Vec<u32>-equivalent (list[int] row ids). The FakeANN mirrors that.
    ann = wire_mcp()
    filters = {
        "terms": ["alpha"],
        "domains": ["example.com"],
        "month_range": ("2026-01", "2026-08"),
        "allow": [0],
    }
    res = home_search("q", filters=filters)
    assert ann.seen_kwargs and ann.seen_kwargs[0] == filters
    assert res["results"]


def test_home_search_translates_spec_shape_filters():
    # Spec §7 shape {date_range, keywords} maps through orch's translation
    # layer to Rust kwargs {month_range, terms} before ann.search.
    ann = wire_mcp()
    res = home_search(
        "q",
        filters={
            "date_range": ("2026-01", "2026-08"),
            "keywords": ["alpha"],
            "domains": ["example.com"],
        },
    )
    assert ann.seen_kwargs and ann.seen_kwargs[0] == {
        "month_range": ("2026-01", "2026-08"),
        "terms": ["alpha"],
        "domains": ["example.com"],
    }
    assert res["results"]


def test_home_search_rejects_unknown_filter_keys():
    from exa_home import orch

    with pytest.raises(ValueError, match="unknown filter key"):
        orch.translate_filters({"bogus": ["x"]})


def test_home_search_top_k_profile_and_degraded():
    wire_mcp(n=5)
    res = home_search("q", top_k=2, profile=True)
    assert len(res["results"]) == 2
    assert "total_ms" in res["profile"]
    wire_mcp(n=3, degraded=True)
    res = home_search("q")
    assert res["degraded"] is True
    assert all(isinstance(r["score"], float) for r in res["results"])


def test_home_contents_paginates_never_silent():
    wire_mcp()
    ids = [f"d{i}" for i in range(3)] + [f"missing-{i}" for i in range(22)]
    res = home_contents(ids)
    assert len(res["items"]) == 20
    assert res["next_offset"] == 20 and res["remaining"] == 5
    res2 = home_contents(ids[res["next_offset"] :])
    assert len(res2["items"]) == 5
    assert "next_offset" not in res2


def test_home_contents_rejects_empty_ids():
    wire_mcp()
    with pytest.raises(ValueError):
        home_contents([])


def test_home_contents_not_found_and_truncation():
    wire_mcp()
    res = home_contents(["d0", "nope"], max_chars_per_doc=10)
    by_id = {i["id"]: i for i in res["items"]}
    assert by_id["nope"] == {"id": "nope", "error": "not_found"}
    assert by_id["d0"]["url"] == "https://example.com/0"
    assert len(by_id["d0"]["text"]) <= 10


def test_home_health_reports_degraded_entities_when_unwired():
    from exa_home.mcp_server import home_health

    h = home_health()
    assert h["status"] in ("ok", "degraded")
    assert set(h["deps"]) == {"embedder", "ann", "reranker", "store"}
    assert "operating_point" in h


def test_home_health_reports_ok_after_wire():
    from exa_home import mcp_server

    e, a, r, s = object(), object(), object(), object()
    mcp_server.wire(e, a, r, s)
    assert mcp_server.home_health()["status"] == "ok"
    mcp_server._deps.clear()  # keep tests independent
