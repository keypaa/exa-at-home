# exa_home/orch.py
"""Canon-lite DAG orchestrator + profile spans (M6).

Sequential V1: nodes execute serially in topological order, each
span-instrumented with time.perf_counter. A threaded fan-out upgrade is an
M6 follow-up recorded in EXPERIMENTS; V1 stays serial so the parallel
speedup is measurable against these spans.

Consumer contracts (Task 11's MCP server relies on these — keep stable):
- Node(name, fn, deps, timeout_ms); DAG(nodes); DAG.run(inputs).
- build_search_dag(embedder, ann, reranker, store, top_coarse=200).
- run_search(dag, query, filters, top_k, profile).
- Filter semantics live in the ANN backend: the retrieve node translates
  the spec-§7 filter dict to Rust IvfIndex.search kwargs via
  translate_filters (date_range→month_range, keywords→terms,
  domains/allow passthrough; unknown keys raise ValueError fail-fast,
  surfacing as a retrieve node error, never silent), then passes the
  translated kwargs straight through to ann.search (terms-AND /
  domains-OR resolved in Rust); this module builds no planner here.
- Rerank degraded items lack a "score" key while success items have one;
  the rerank node normalizes (degraded score = coarse ANN score) so
  run_search results have a uniform shape.
"""
from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable
from dataclasses import dataclass, field

SNIPPET_CHARS = 300


@dataclass
class Node:
    name: str
    fn: Callable[[dict], object]  # (dict of dep outputs + inputs) -> output
    deps: list[str] = field(default_factory=list)
    timeout_ms: int | None = None


class DAG:
    def __init__(self, nodes: list[Node]):
        self.nodes: dict[str, Node] = {}
        for n in nodes:
            if n.name in self.nodes:
                raise ValueError(f"DAG: duplicate node {n.name!r}")
            self.nodes[n.name] = n
        for n in nodes:
            for d in n.deps:
                if d == n.name:
                    raise ValueError(f"DAG: node {n.name!r} depends on itself")
                if d not in self.nodes:
                    raise ValueError(
                        f"DAG: node {n.name!r} depends on unknown {d!r}")
        self._order = self._topo_sort()

    def _topo_sort(self) -> list[str]:
        indeg = {name: 0 for name in self.nodes}
        children: dict[str, list[str]] = {name: [] for name in self.nodes}
        for n in self.nodes.values():
            for d in dict.fromkeys(n.deps):  # dedupe, keep first-seen order
                indeg[n.name] += 1
                children[d].append(n.name)
        ready = [name for name in self.nodes if indeg[name] == 0]
        order = []
        while ready:
            name = ready.pop(0)
            order.append(name)
            for c in children[name]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("DAG: cycle detected")
        return order

    def run(self, inputs: dict) -> dict:
        """Execute nodes serially in topo order.

        Returns {"outputs", "spans"} on success; on node failure returns
        {"outputs" (completed + inputs so far), "spans" (incl. the failing
        node's span), "error": {node, error, type, span_ms}}. Dependents of
        a failed node never run. A node with timeout_ms set runs in a
        one-shot worker thread and fails on timeout (same wait-on-exit
        semantics as Reranker._score_batch's testability path).
        """
        outputs = dict(inputs)
        spans: dict[str, float] = {}
        for name in self._order:
            node = self.nodes[name]
            snap = dict(outputs)  # shallow copy: node fns must not mutate
            t0 = time.perf_counter()
            try:
                if node.timeout_ms is None:
                    outputs[name] = node.fn(snap)
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        fut = pool.submit(node.fn, snap)
                        outputs[name] = fut.result(timeout=node.timeout_ms / 1000)
            except Exception as e:  # noqa: BLE001 — failure is data here
                dt_ms = (time.perf_counter() - t0) * 1000
                spans[name] = dt_ms
                return {"outputs": outputs, "spans": spans,
                        "error": {"node": name, "error": str(e),
                                  "type": type(e).__name__, "span_ms": dt_ms}}
            spans[name] = (time.perf_counter() - t0) * 1000
        return {"outputs": outputs, "spans": spans}


# Spec §7 (home_search) filter shape -> Rust IvfIndex.search kwargs.
#   date_range -> month_range, keywords -> terms,
#   domains / month_range / terms / allow pass through unchanged.
# Unknown keys raise ValueError (fail-fast: surfaces as a retrieve node
# error via DAG.run, never silently ignored). Terms-AND / domains-OR
# semantics stay untouched — they are resolved in Rust.
SPEC_TO_RUST_FILTERS = {"date_range": "month_range", "keywords": "terms"}
RUST_FILTER_KEYS = frozenset({"domains", "month_range", "terms", "allow"})


def translate_filters(filters: dict) -> dict:
    """Map the spec-shaped filter dict to IvfIndex.search kwargs."""
    out: dict = {}
    for k, v in (filters or {}).items():
        rust_key = SPEC_TO_RUST_FILTERS.get(k, k)
        if rust_key not in RUST_FILTER_KEYS:
            raise ValueError(
                f"unknown filter key {k!r}; allowed: "
                f"{{domains, date_range, keywords}} (spec shape) or "
                f"{{domains, month_range, terms, allow}} (Rust shape)")
        out[rust_key] = v
    return out


def build_search_dag(embedder, ann, reranker, store, top_coarse=200, nprobe=8):
    """Assemble embed -> retrieve [filter in retrieve] -> rerank -> snippet.

    Every backend arrives via a constructor arg (mock-friendly). Filters
    are translated to ann.search kwargs via translate_filters — never
    reinterpreted beyond that naming map.

    Sub-span side channel: run_search injects a mutable ``_parts`` dict
    into the DAG inputs; node fns record fetch/model splits there. This
    is the one sanctioned mutation (DAG.run still shallow-copies outputs
    per node; only the shared ``_parts`` dict is written through).
    """
    def embed_fn(x):
        return embedder.encode_queries([x["query"]])[0]

    def retrieve_fn(x):
        t0 = time.perf_counter()
        filters = translate_filters(x.get("filters") or {})
        parts = x.get("_parts")
        if parts is not None:
            parts["filter_translate_ms"] = (time.perf_counter() - t0) * 1000
        ids, scores = ann.search(x["embed"], nprobe=nprobe,
                                 top_k=top_coarse, **filters)
        return [{"id": i, "coarse_score": float(s)}
                for i, s in zip(ids, scores)]

    def rerank_fn(x):
        parts = x.get("_parts")
        t0 = time.perf_counter()
        cands = []
        for c in x["retrieve"]:
            doc = store.get(c["id"])
            text = (doc.get("text", "") if doc else "")
            cands.append({"id": c["id"], "text": text,
                          "coarse_score": c["coarse_score"]})
        fetch_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        ranked, degraded = reranker.rerank(x["query"], cands)
        model_ms = (time.perf_counter() - t1) * 1000
        if parts is not None:
            parts["rerank_fetch_ms"] = fetch_ms
            parts["rerank_model_ms"] = model_ms
        norm = []
        for item in ranked:
            item = dict(item)
            if "score" not in item:
                # Degraded path keeps coarse ANN order and carries no score;
                # fall back to the coarse score so every result has one.
                item["score"] = item.get("coarse_score")
            if item["score"] is not None:
                item["score"] = float(item["score"])
            norm.append(item)
        return {"ranked": norm, "degraded": bool(degraded)}

    def snippet_fn(x):
        parts = x.get("_parts")
        t0 = time.perf_counter()
        top_k = x.get("top_k", 10)
        out = []
        for item in x["rerank"]["ranked"][:top_k]:
            doc = store.get(item["id"])
            if doc and doc.get("text"):
                text = doc["text"]
            else:
                text = item.get("text", "")
            out.append({"id": item["id"],
                        "snippet": (text or "")[:SNIPPET_CHARS],
                        "score": item["score"],
                        "coarse_score": item.get("coarse_score")})
        if parts is not None:
            # Snippet fetch is the store.get loop; formatting is negligible.
            parts["snippet_fetch_ms"] = (time.perf_counter() - t0) * 1000
        return out

    return DAG([Node("embed", embed_fn, []),
                Node("retrieve", retrieve_fn, ["embed"]),
                Node("rerank", rerank_fn, ["retrieve"]),
                Node("snippet", snippet_fn, ["rerank"])])


def run_search(dag, query, filters=None, top_k=10, profile=False):
    """Run the search DAG; never raises for backend failures.

    Returns {"results", "degraded"} plus "profile" ({stage: ms, total_ms})
    when profile=True, plus "error" when a node failed (results [] then).
    """
    t0 = time.perf_counter()
    parts: dict = {}
    out = dag.run({"query": query, "filters": filters or {}, "top_k": top_k,
                   "_parts": parts})
    wall_ms = (time.perf_counter() - t0) * 1000
    err = out.get("error")
    rerank_out = out["outputs"].get("rerank") or {}
    res: dict = {"results": [] if err else list(out["outputs"].get("snippet", [])),
                 "degraded": True if err else bool(rerank_out.get("degraded", False))}
    if err:
        res["error"] = err
    if profile:
        spans = out["spans"]
        # filter_ms is the measured filter-translation time (in-scan
        # intersect itself stays inside ann_ms — the retrieve node owns
        # both, so the waterfall keeps ann as the stage).
        res["profile"] = {
            "embed_ms": spans.get("embed", 0.0),
            "ann_ms": spans.get("retrieve", 0.0),
            "filter_ms": parts.get("filter_translate_ms", 0.0),
            "rerank_ms": spans.get("rerank", 0.0),
            "rerank_fetch_ms": parts.get("rerank_fetch_ms", 0.0),
            "rerank_model_ms": parts.get("rerank_model_ms", 0.0),
            "snippet_ms": spans.get("snippet", 0.0),
            "snippet_fetch_ms": parts.get("snippet_fetch_ms", 0.0),
            "total_ms": wall_ms,
        }
    return res
