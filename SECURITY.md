# Security Policy

## Supported versions

| Version | Supported          |
|---------|--------------------|
| main    | :white_check_mark: |
| < 0.9.0 | :x:                |

Only the latest `main` branch receives security fixes. Older releases are a
learning artifact and are not supported.

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security vulnerabilities.
Instead, contact the maintainer directly (see the GitHub repository contact /
profile) with a description of the issue, steps to reproduce, and the affected
component.

We will acknowledge receipt within 5 business days and work with you on a
fix and disclosure timeline.

## Known security considerations (read first)

### The MCP server (`exa_home/mcp_server.py`)

- The default bind is **loopback** (`127.0.0.1`). Exposing the server
  beyond the machine is an explicit choice (`--host`).
- HTTP transport is **bearer-token authenticated** (`--token` /
  `EXA_MCP_TOKEN`) and **rate-limited** (sliding window,
  `--rate-limit`, calls/minute per server). Both run as FastMCP
  middleware. `--config` may replace the serving operating point, and tool
  failures are redacted (`mask_error_details` + a wrapper that converts
  internal exceptions into `{"error": {...}}` shapes) so exception types,
  messages and internal paths never cross the boundary.
- If `--token` is not set over HTTP, the server runs **unauthenticated**;
  it prints a warning at startup. Keep it on loopback or set a token.
- `home_contents` is paginated (`PAGE_SIZE = 20` ids per call,
  `max_chars_per_doc` capped at 20 000 chars); `home_search` bounds the
  query length and `top_k`; validation errors are the caller's, internal
  errors are never echoed.

### The corpus is attacker-controlled

`home_search` / `home_contents` return Common Crawl text. Every corpus
string is passed through a control/format-character stripper and tagged
`"provenance": "corpus"`, `"untrusted": true` so a downstream agent can
treat it as data, never instructions. That marking is not a sanitisation
boundary by itself: text content is preserved, the flag is the contract.

### The content store and the index manifest

- `store/offsets.json` shard paths are contained under the store root
  (absolute paths and `..` traversal entries are dropped), and every
  record carries a content digest: a record mutated in place under a known
  id resolves to a **miss**, not substituted content.
- `index/manifest.json` is an **integrity** check (per-file sha256,
  fail-fast verify on load), not an **authenticity** check: whoever can
  replace the index files can replace the manifest. Only serve index
  directories you trust — or hook `verify_manifest_signature` to an
  out-of-band signature/trust anchor.
- The content store (`store/`) is deliberately excluded from the manifest
  (it is corpus, and hashing it made every load O(corpus)).

### Pipeline provenance gates

`scripts/run_pipeline.py` re-uses artifacts only when the recorded input
identity still matches: the WET download set (exact URL match, foreign
files pruned), `docs.jsonl` (WET fingerprint sidecar), and
`q.jsonl`/`gt.jsonl` (docs/vecs/ids/nq fingerprint sidecar). WET downloads
are gzip-verified (CRC/ISIZE) before being treated as complete.

The 2026-11-12 audit's full findings list and fix status live in
[`zenno/plans/2026-11-12-exa-at-home-fix-all-audit-findings.md`](zenno/plans/2026-11-12-exa-at-home-fix-all-audit-findings.md).
