# exa_home/mcp_server.py — Task 11: FastMCP app (home_search, home_contents).
#
# Consumes Task 10's build_search_dag/run_search + Task 2's ContentStore.
# Pure Python; no Rust, no GPU. FastMCP tools are plain functions under the
# decorator — tests call them directly, not the transport.
from __future__ import annotations

from fastmcp import FastMCP

app = FastMCP("exa-home")

_deps = {}  # wired by wire() at startup; tests inject stubs


def wire(embedder=None, ann=None, reranker=None, store=None):
    _deps.update(embedder=embedder, ann=ann, reranker=reranker, store=store)


@app.tool
def home_search(query: str, filters: dict | None = None, top_k: int = 10,
                profile: bool = False) -> dict:
    if not query or not query.strip():
        raise ValueError("query must be non-empty")
    from .orch import build_search_dag, run_search
    dag = build_search_dag(_deps["embedder"], _deps["ann"],
                           _deps["reranker"], _deps["store"])
    # Exact Task 10 arg order; filters=None means {}. Filters pass through
    # to the Rust ANN backend verbatim — never reinterpreted here.
    res = run_search(dag, query, filters or {}, top_k, profile)
    # run_search results carry {id, snippet, score, coarse_score}; enrich
    # with url/title from the content store for the MCP contract shape.
    enriched = []
    for r in res.get("results", []):
        doc = _deps["store"].get(r["id"])
        item = dict(r)
        item["url"] = (doc.get("url", "") if doc else "")
        item["title"] = (doc.get("title", "") if doc else "")
        enriched.append(item)
    res["results"] = enriched
    return res


@app.tool
def home_contents(ids: list[str], max_chars_per_doc: int = 8000) -> dict:
    if not ids:
        raise ValueError("ids must be non-empty")
    page, rest = ids[:20], ids[20:]
    items = []
    for i in page:
        d = _deps["store"].get(i)
        if d is None:
            items.append({"id": i, "error": "not_found"})
        else:
            items.append({"id": i, "url": d["url"], "title": d.get("title", ""),
                          "text": d["text"][:max_chars_per_doc]})
    out = {"items": items}
    if rest:
        out["next_offset"] = len(page)
        out["remaining"] = len(rest)
    return out


def run_stdio():
    app.run()
