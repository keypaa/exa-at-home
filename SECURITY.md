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

- The FastMCP server (`exa_home/mcp_server.py`) exposes `home_search`,
  `home_contents`, and `home_health`. The current build ships **no auth
  token, no rate limiter, and no `--bind`/`--host` flag**; the default bind
  is loopback (`127.0.0.1`), but `FASTMCP_HOST=0.0.0.0` exposes all
  interfaces. Do not expose the server to untrusted networks until auth and
  rate limiting are added (tracked in `SECURITY.md` / the audit report).
- The content store (`exa_home/store.py`) trusts `store/offsets.json`; an
  attacker-controlled index directory can lead to path traversal. Only serve
  index directories you trust.
- The corpus is Common Crawl (attacker-controlled web text). `home_contents`
  returns raw document text with no sanitization or provenance marking —
  treat returned text as untrusted (prompt-injection risk).

See the full audit report (`/tmp/opencode/audit/AUDIT_REPORT.md` in the
maintainer's checkout) for the complete findings list.
