# exa_home/mcp_server.py — FastMCP app (home_search, home_contents).
#
# The serving surface of the project. Design notes:
#
# - `fastmcp` is imported LAZILY (inside `_get_app`). A production module
#   that cannot be imported without an optional dependency forces tests to
#   stub `sys.modules` — inverted, and it duplicated our test suite.
# - The serving configuration comes from exa_home.config.OPERATING_POINT,
#   so the tool that ships cannot drift from the configuration we measured.
# - `main()` is the runnable entry point: `python -m exa_home.mcp_server
#   --index index/ --port 8080`. Without it the MCP server could not be
#   started at all.
from __future__ import annotations

import os

from .config import OPERATING_POINT

_app = None
_deps: dict = {}  # wired by wire(); populated by main() or by tests


def wire(embedder=None, ann=None, reranker=None, store=None):
    """Attach the backends. Called by main() at startup, or by tests."""
    _deps.update(embedder=embedder, ann=ann, reranker=reranker, store=store)


def _get_app():
    """Build (once) the FastMCP app. Imported lazily so this module stays
    importable without fastmcp installed."""
    global _app
    if _app is None:
        from fastmcp import FastMCP

        _app = FastMCP("exa-home")
        # The tools are plain module-level functions (testable without
        # fastmcp); registering them here keeps the MCP surface identical.
        _app.tool(home_search)
        _app.tool(home_contents)
    return _app


def home_search(
    query: str,
    filters: dict | None = None,
    top_k: int = OPERATING_POINT["k"],
    profile: bool = False,
) -> dict:
    """Search the corpus. Returns {results, degraded, profile?, error?}."""
    if not query or not query.strip():
        raise ValueError("query must be non-empty")
    from .orch import build_search_dag, run_search

    dag = build_search_dag(
        _deps["embedder"],
        _deps["ann"],
        _deps["reranker"],
        _deps["store"],
        top_coarse=OPERATING_POINT["top_coarse"],
        nprobe=OPERATING_POINT["nprobe"],
    )
    # Exact Task 10 arg order; filters=None means {}. Spec-shaped filters
    # ({domains, date_range, keywords}) are translated to Rust kwargs in
    # orch.retrieve via translate_filters — never reinterpreted here.
    res = run_search(dag, query, filters or {}, top_k, profile)
    # run_search results carry {id, snippet, score, coarse_score}; enrich
    # with url/title from the content store for the MCP contract shape.
    enriched = []
    for r in res.get("results", []):
        doc = _deps["store"].get(r["id"])
        item = dict(r)
        item["url"] = doc.get("url", "") if doc else ""
        item["title"] = doc.get("title", "") if doc else ""
        enriched.append(item)
    res["results"] = enriched
    return res


def home_contents(ids: list[str], max_chars_per_doc: int = 8000) -> dict:
    """Fetch stored text for the given doc ids (paginated, 20 per call)."""
    if not ids:
        raise ValueError("ids must be non-empty")
    page, rest = ids[:20], ids[20:]
    items = []
    for i in page:
        d = _deps["store"].get(i)
        if d is None:
            items.append({"id": i, "error": "not_found"})
        else:
            items.append(
                {
                    "id": i,
                    "url": d["url"],
                    "title": d.get("title", ""),
                    "text": d["text"][:max_chars_per_doc],
                }
            )
    out = {"items": items}
    if rest:
        out["next_offset"] = len(page)
        out["remaining"] = len(rest)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the MCP server over stdio (default) or HTTP.

    Usage:
        python -m exa_home.mcp_server --index index/ [--store DIR]
                                    [--transport stdio|http] [--port 8080]
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="index/", help="index dir to serve")
    ap.add_argument("--store", default=None, help="content store root")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "http"])
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)

    from ann_core import IvfIndex

    from .embed import Embedder
    from .rerank import Reranker
    from .store import ContentStore

    wire(
        embedder=Embedder(),
        ann=IvfIndex.load(a.index),
        reranker=Reranker(
            max_pair_tokens=OPERATING_POINT["max_pair_tokens"],
            batch_size=OPERATING_POINT["batch_size"],
            max_query_tokens=OPERATING_POINT["max_query_tokens"],
        ),
        store=ContentStore(a.store or os.path.join(a.index, "store")),
    )
    app = _get_app()
    if a.transport == "http":
        app.run(transport="http", port=a.port)
    else:
        app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
