# exa_home/mcp_server.py — FastMCP app (home_search, home_contents).
#
# The serving surface of the project. Design notes:
#
# - `fastmcp` is imported LAZILY (inside `_get_app`). A production module
#   that cannot be imported without an optional dependency forces tests to
#   stub `sys.modules` — inverted, and it duplicated our test suite.
# - The serving configuration comes from exa_home.config.OPERATING_POINT by
#   default; main(--config) can replace it with the loaded config so the
#   shipped tool cannot drift from the measured one.
# - `main()` is the runnable entry point: `python -m exa_home.mcp_server
#   --index index/ --port 8080`. HTTP mode is token-authenticated and
#   rate-limited; stdio mode is local-only and needs no token.
from __future__ import annotations

import hmac
import os
import sys
import threading
import time
import unicodedata
from collections import deque
from functools import wraps

from .config import OPERATING_POINT

_app = None
_deps: dict = {}  # wired by wire(); populated by main() or by tests

# The active operating point. `main(--config)` may replace this with the
# loaded config's `operating_point`; tests can call configure_operating_point.
_operating_point: dict = dict(OPERATING_POINT)

# Bound attacker-cost knobs: a caller can never size an absurd request.
MAX_QUERY_CHARS = 4096
MAX_IDS_PER_CONTENTS_CALL = 100
PAGE_SIZE = 20  # home_contents page size (named constant, finding P3-16)


def configure_operating_point(op: dict | None) -> dict:
    """Set the serving operating point (loaded config), returning it."""
    global _operating_point
    if op:
        _operating_point = {**OPERATING_POINT, **op}
    return _operating_point


def wire(embedder=None, ann=None, reranker=None, store=None):
    """Attach the backends. Called by main() at startup, or by tests."""
    _deps.update(embedder=embedder, ann=ann, reranker=reranker, store=store)


def _get_app():
    """Build (once) the FastMCP app. Imported lazily so this module stays
    importable without fastmcp installed."""
    global _app
    if _app is None:
        from fastmcp import FastMCP

        # mask_error_details: never leak exception types/messages/internal
        # paths to MCP clients (findings S4/P2-10).
        _app = FastMCP("exa-home", mask_error_details=True)
        # The tools are plain module-level functions (testable without
        # fastmcp); registering them here keeps the MCP surface identical.
        # `_safe_tool` converts unexpected failures into a redacted result
        # shape instead of a traceback over the wire.
        _app.tool(_safe_tool(home_search))
        _app.tool(_safe_tool(home_contents))
        _app.tool(_safe_tool(home_health))
    return _app


def _safe_tool(fn):
    """Wrap a tool so the MCP boundary never returns internal details."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as e:
            # Validation errors are the caller's fault: keep the message,
            # never the traceback/type internals.
            return {"error": {"code": "invalid_request", "message": str(e)}}
        except Exception:  # noqa: BLE001 — the boundary must not leak
            return {"error": {"code": "internal", "message": "internal error"}}

    return wrapper


def _sanitize_text(text: str) -> str:
    """Strip control/format characters from corpus-derived text.

    Common Crawl text is attacker-controlled. Control characters are
    removed (newline/tab preserved) and the text is marked untrusted at the
    response level (finding S3). This is not a prompt-injection defence by
    itself — the provenance flag tells the client to treat it as data.
    """
    if not text:
        return ""
    return "".join(
        ch
        for ch in text
        if ch in ("\n", "\t") or unicodedata.category(ch) not in ("Cc", "Cf")
    )


def home_search(
    query: str,
    filters: dict | None = None,
    top_k: int | None = None,
    profile: bool = False,
) -> dict:
    """Search the corpus. Returns {results, degraded, profile?, error?}."""
    if not query or not query.strip():
        raise ValueError("query must be non-empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(
            f"query must be <= {MAX_QUERY_CHARS} characters (got {len(query)})"
        )
    op = _operating_point
    if top_k is None:
        top_k = op["k"]
    if top_k <= 0 or top_k > 100:
        raise ValueError("top_k must be in (0, 100]")
    from .orch import build_search_dag, run_search

    # Retrieve at least `top_k` coarse candidates so a caller asking for
    # top_k > top_coarse is not silently capped at top_coarse (finding P2-09).
    top_coarse = max(op["top_coarse"], top_k)
    dag = build_search_dag(
        _deps["embedder"],
        _deps["ann"],
        _deps["reranker"],
        _deps["store"],
        top_coarse=top_coarse,
        nprobe=op["nprobe"],
    )
    # Exact Task 10 arg order; filters=None means {}. Spec-shaped filters
    # ({domains, date_range, keywords}) are translated to Rust kwargs in
    # orch.retrieve via translate_filters — never reinterpreted here.
    res = run_search(dag, query, filters or {}, top_k, profile)
    # run_search results carry {id, snippet, score, coarse_score}; enrich
    # with url/title from the content store for the MCP contract shape.
    store = _deps["store"]
    results = res.get("results", [])
    ids = [r["id"] for r in results]
    if hasattr(store, "get_many"):
        docs = store.get_many(ids)
    else:  # pragma: no cover — injected fakes from older tests
        docs = {i: store.get(i) for i in ids}
    enriched = []
    for r in results:
        doc = docs.get(r["id"])
        item = dict(r)
        item["url"] = doc.get("url", "") if doc else ""
        item["title"] = _sanitize_text(doc.get("title", "")) if doc else ""
        if "snippet" in item:
            item["snippet"] = _sanitize_text(item["snippet"])
        # The text/snippet is corpus-derived, untrusted input. Mark it so a
        # downstream agent does not treat it as instructions (finding S3).
        item["provenance"] = "corpus"
        item["untrusted"] = True
        enriched.append(item)
    res["results"] = enriched
    # Never expose the underlying exception detail over the MCP boundary.
    if isinstance(res.get("error"), dict):
        res["error"] = {"error": "internal error (see server logs)"}
    return res


def home_contents(ids: list[str], max_chars_per_doc: int = 8000) -> dict:
    """Fetch stored text for the given doc ids (paginated, PAGE_SIZE per call)."""
    if not ids or not isinstance(ids, (list, tuple)):
        raise ValueError("ids must be a non-empty list of store keys")
    if len(ids) > MAX_IDS_PER_CONTENTS_CALL:
        raise ValueError(
            f"ids must be <= {MAX_IDS_PER_CONTENTS_CALL} per call (got {len(ids)})"
        )
    if max_chars_per_doc <= 0 or max_chars_per_doc > 20_000:
        raise ValueError("max_chars_per_doc must be in (0, 20000]")
    page, rest = list(ids[:PAGE_SIZE]), list(ids[PAGE_SIZE:])
    store = _deps["store"]
    if hasattr(store, "get_many"):
        docs = store.get_many(page)
    else:  # pragma: no cover — injected fakes from older tests
        docs = {i: store.get(i) for i in page}
    items = []
    for i in page:
        d = docs.get(i)
        if d is None:
            items.append({"id": i, "error": "not_found"})
        else:
            items.append(
                {
                    "id": i,
                    "url": d["url"],
                    "title": _sanitize_text(d.get("title", "")),
                    "text": _sanitize_text(d["text"][:max_chars_per_doc]),
                    # Corpus text is untrusted data, never instructions.
                    "provenance": "corpus",
                    "untrusted": True,
                }
            )
    out = {"items": items}
    if rest:
        out["next_offset"] = len(page)
        out["remaining"] = len(rest)
    return out


class RateLimiter:
    """Sliding-window limiter (calls/minute). Thread-safe."""

    def __init__(self, limit_per_min: int):
        self.limit = max(0, int(limit_per_min))
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self, now: float | None = None) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            cutoff = now - 60.0
            while self._times and self._times[0] < cutoff:
                self._times.popleft()
            if len(self._times) >= self.limit:
                return False
            self._times.append(now)
            return True


def _check_bearer(authorization: str, token: str | None) -> bool:
    """Constant-time bearer check; an empty token disables auth."""
    if not token:
        return True
    return hmac.compare_digest(authorization or "", f"Bearer {token}")


def _auth_middleware_class():
    """Build the FastMCP middleware class lazily (fastmcp may be absent)."""
    from fastmcp.server.dependencies import get_http_headers
    from fastmcp.server.middleware import Middleware

    class AuthRateLimitMiddleware(Middleware):
        def __init__(self, token: str | None = None, rate_limit: int = 0):
            self.token = token
            self.limiter = RateLimiter(rate_limit)

        async def on_call_tool(self, context, call_next):
            if self.token:
                headers = get_http_headers(include={"authorization"})
                if not _check_bearer(headers.get("authorization", ""), self.token):
                    raise PermissionError("unauthorized: missing/invalid bearer token")
            if not self.limiter.allow():
                raise RuntimeError("rate limit exceeded; retry later")
            return await call_next(context)

    return AuthRateLimitMiddleware


def main(argv: list[str] | None = None) -> int:
    """Run the MCP server over stdio (default) or HTTP.

    Usage:
        python -m exa_home.mcp_server --index index/ [--store DIR]
                                    [--transport stdio|http] [--host 127.0.0.1]
                                    [--port 8080] [--token TOKEN]
                                    [--rate-limit 120] [--config cfg.yaml]
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="index/", help="index dir to serve")
    ap.add_argument("--store", default=None, help="content store root")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "http"])
    # Loopback by default: exposing the corpus to the network must be an
    # explicit choice (finding S1).
    ap.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--token",
        default=os.environ.get("EXA_MCP_TOKEN"),
        help="bearer token for HTTP transport (default: $EXA_MCP_TOKEN)",
    )
    ap.add_argument(
        "--rate-limit",
        type=int,
        default=120,
        help="HTTP calls per minute per server (0 disables)",
    )
    ap.add_argument("--config", default=None, help="YAML/JSON config overrides")
    a = ap.parse_args(argv)

    if a.config:
        from .config_loader import load

        cfg = load(a.config)
        configure_operating_point(cfg.get("operating_point", {}))

    op = _operating_point
    from ann_core import IvfIndex

    from .embed import Embedder
    from .rerank import Reranker
    from .store import ContentStore

    wire(
        embedder=Embedder(),
        ann=IvfIndex.load(a.index),
        reranker=Reranker(
            max_pair_tokens=op["max_pair_tokens"],
            batch_size=op["batch_size"],
            max_query_tokens=op["max_query_tokens"],
        ),
        store=ContentStore(a.store or os.path.join(a.index, "store")),
    )
    app = _get_app()
    if a.transport == "http":
        if not a.token:
            print(
                "warning: HTTP transport without --token/EXA_MCP_TOKEN; "
                "the server is unauthenticated",
                file=sys.stderr,
            )
        middleware = []
        if a.token or a.rate_limit:
            mw_cls = _auth_middleware_class()
            middleware.append(mw_cls(token=a.token, rate_limit=a.rate_limit))
        http_app = app.http_app(middleware=middleware or None)
        import uvicorn

        uvicorn.run(http_app, host=a.host, port=a.port)
    else:
        app.run()
    return 0


def home_health() -> dict:
    """Pure, dependency-checking health probe for operators."""
    out = {
        "status": "ok",
        "deps": {
            k: v is not None
            for k, v in {
                "embedder": _deps.get("embedder", None),
                "ann": _deps.get("ann", None),
                "reranker": _deps.get("reranker", None),
                "store": _deps.get("store", None),
            }.items()
        },
        "operating_point": dict(_operating_point),
    }
    if not all(out["deps"].values()):
        out["status"] = "degraded"
    return out


if __name__ == "__main__":
    raise SystemExit(main())
