# MCP server API (`exa_home/mcp_server.py`)

The serving surface. FastMCP app named `exa-home`; `python -m
exa_home.mcp_server --index index/` runs it (stdio by default, `--transport
http` for HTTP). `fastmcp` is imported lazily inside `_get_app()` so the
module stays importable without it.

```bash
python -m exa_home.mcp_server --index index/ [--store DIR] \
    --transport stdio|http --host 127.0.0.1 --port 8080 \
    --token TOKEN --rate-limit 120 --config cfg.yaml
```

| flag | default | meaning |
|------|---------|---------|
| `--index` | `index/` | index dir (loaded via `ann_core.IvfIndex.load`, manifest-verified) |
| `--store` | `<index>/store` | content store root |
| `--transport` | `stdio` | `http` enables bearer auth + rate limiting |
| `--host` | `127.0.0.1` | loopback by default; exposing the server is an explicit choice |
| `--port` | `8080` | HTTP port |
| `--token` | `$EXA_MCP_TOKEN` | bearer token; HTTP without one prints an unauthenticated warning |
| `--rate-limit` | `120` | calls/minute (whole server, sliding window; `0` disables) |
| `--config` | – | YAML/JSON config; its `operating_point` replaces the serving defaults |

The serving operating point comes from `exa_home.config.OPERATING_POINT`
unless `--config` says otherwise (`configure_operating_point`). The loaded
config is strict: unknown keys raise, and `operating_point` may not
contradict `serving`.

## Request bounds

- `home_search`: `query` non-blank and <= 4096 chars; `top_k` in (0, 100];
  coarse retrieval uses `max(top_coarse, top_k)` so asking for more
  results is never silently capped.
- `home_contents`: 1..100 ids per call, rendered 20 (`PAGE_SIZE`) at a time
  with `next_offset` when more follow; `max_chars_per_doc` in (0, 20 000].

## Tools

### `home_search(query, filters?, top_k?, profile=false)`

`{query: str, filters: {domains?, date_range?, keywords?} | null, top_k: int,
profile: bool}`.

Search the corpus through the Canon-lite DAG: embed → retrieve (filters
translated to the Rust backend) → rerank → snippet. Returns
`{results: [...], degraded: bool, profile?: {...}, error?: {...}}`, one
entry per result:

```json
{
  "id": "sha1:...", "url": "https://...", "title": "…",
  "snippet": "…", "score": 1.23, "coarse_score": 4.5,
  "provenance": "corpus", "untrusted": true
}
```

Error shape: `{"error": {"code": "invalid_request" | "internal",
"message": str}}`. `invalid_request` is the caller's fault (bad
query/filter/top_k) and keeps the validation message; `internal` never
leaks exception types, messages or paths (`mask_error_details` + the
`_safe_tool` wrapper).

### `home_contents(ids, max_chars_per_doc=8000)`

`{ids: [str, ...], max_chars_per_doc: int}` → `{items: [...],
next_offset?: int, remaining?: int}`. Items are `{id, url, title, text,
provenance: "corpus", untrusted: true}` or `{id, error: "not_found"}`.

### `home_health()`

`{status: "ok" | "degraded", deps: {embedder, ann, reranker, store: bool},
operating_point: {...}}`. Pure, no model work; useful for operators.

## Untrusted content

Common Crawl text is attacker-controlled. Every corpus-derived string is
stripped of control/format characters (newline/tab preserved) and tagged
`"provenance": "corpus"`, `"untrusted": true`. Treat it as data, never
instructions — the tag is the contract with the client.

## HTTP middleware

`--transport http` installs `AuthRateLimitMiddleware` (a FastMCP
middleware) when a token or rate limit is configured:

- bearer check via `hmac.compare_digest` (constant-time; an empty token
  disables auth) — failures raise `PermissionError`;
- `RateLimiter` (sliding window over `time.monotonic`) — exceeding the
  limit raises `RuntimeError` with a retry hint.

Both surface through `_safe_tool`, so the client sees a redacted error
dict, never a traceback.
