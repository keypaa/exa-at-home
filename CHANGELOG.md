# Changelog

## [Unreleased]

### Fixed
- Review-hardening pass (2026-11-13): HTTP auth/rate limiting now runs as
  Starlette ASGI middleware (real 401/429 responses); store misses on an
  unchanged shard do not rescan it; `force_full` scans drop stale ids;
  mixed-digest `offsets.json` fails closed; manifest keys are contained in
  both the Python and Rust loaders and the build-only `centroids.npy` is
  excluded; DAG/reranker timeouts retire only their own pool and never cancel
  concurrent callers' queued work; `run_e3` provenance is fail-closed;
  `--no-store --store` is rejected.
- Track `Cargo.lock` so Rust dependency builds are reproducible (was gitignored).
- `AnnIndex::new` now caps matrix elements (`MAX_MATRIX_ELEMENTS`) and returns a
  clean `ValueError` instead of aborting the process on an absurd strided shape.
- `ContentStore._load_persisted_index` contains `offsets.json` shard paths under
  the store root (no absolute/`..` traversal).
- `ContentStore._shard` reads `config.SHARD_COUNT` instead of a hardcoded `64`.
- Store concurrency/latency (audit P1/P2/P5/P6/P10): split the global lock so
  cold reads no longer serialize; a miss for a newly appended doc scans only
  the appended tail, and a miss on an unchanged shard does not rescan it at
  all; each record carries a build-time content digest so an
  in-place same-id mutation resolves to a miss instead of substituted content;
  bulk loads keep one append handle per shard; added `get_many`; the RAM LRU is
  now byte-budgeted.
- Ingest (P3): dedup is a temp SQLite table instead of an unbounded in-RAM set;
  duplicate/too-short ordering is preserved.
- Config (P4): `config_loader` is strict — unknown/typo'd keys and bad hardware
  values raise `ValueError` (no silent default deferral) and `operating_point`
  may not contradict `serving`; the MCP server consumes the loaded config via
  `--config` / `configure_operating_point`.
- MCP server (S1/S3/S4/P2-10): loopback bind by default; HTTP transport adds a
  constant-time bearer check (`--token` / `EXA_MCP_TOKEN`) and a sliding-window
  rate limiter (`--rate-limit`) as Starlette ASGI middleware (real 401/429
  responses, not a FastMCP hook); `mask_error_details` plus
  a tool wrapper redact internal errors into an `{"error": {code, message}}`
  shape; corpus strings are stripped of control characters and tagged
  `provenance="corpus"` / `untrusted=true`; `home_contents` paginates
  (`PAGE_SIZE=20`) and caps `max_chars_per_doc` at 20 000; the DAG/reranker
  timeout pools are reused and replaced instead of leaking one pool per timeout
  (both gained `close()`).
- Pipeline provenance gates (C1/C2/C3): `stage_download` reuses a WET dir only
  when its file set matches the requested URLs (foreign files pruned);
  `stage_ingest` writes/verifies a WET-fingerprint sidecar; `stage_queries`
  writes/verifies a docs/vecs/ids/nq fingerprint sidecar. Stale artifacts are
  regenerated instead of silently reused.
- WET downloads (C9) verify gzip integrity (CRC/ISIZE) before skipping a file
  and after downloading one; corrupt files are removed, not cached.
- `run_e3` (C10) validates `--docs` against the index's recorded source
  (sha256, size for legacy indexes) and fingerprints the target filename by
  corpus hash; `infer_docs_path` prefers corpus-named docs over the root file.
- Index rebuilds no longer accumulate content store (disk hygiene):
  `build_index` prunes the derived store (`ContentStore.reset()`, shards +
  `offsets.json` only — foreign files untouched) before repopulating it, and
  `--no-store` prunes a stale store dir left by a previous build so a dir can
  never outlive the index it belongs to. `--no-reset-store` keeps the
  append/merge semantics as an explicit opt-out.

### Performance
- Embedding: the production path tokenizes each chunk once (no duplicate
  `_token_lengths` pass), sorts by token length, and pads each batch to its
  own longest member instead of the whole chunk; `max_seq_length` is a
  parameter recorded in the resume state/provenance.
- Store/index: `ContentStore.put_line` stores the original JSONL line
  byte-for-byte; `build_index` uses it, removing a full JSON parse→dump
  round-trip over the corpus.
- Index: `assign(device="auto"|"cuda"|"cpu")` runs the centroid matmul on the
  GPU when available (5M/K=30k CPU is ~38 TFLOP); `train_centroids` exposes
  `--n-init`/`--max-iter` for the K-means quality/speed A/B.
- Added `docs/SCALE_PLAN.md`: mmap + shard architecture and the stage-by-stage
  optimization backlog for 50M → 10B docs.

### Security
- `index/manifest.json` is documented (and hooked via
  `verify_manifest_signature`) as an integrity check, not an authenticity
  check: serving an index from an untrusted location needs an out-of-band
  trust anchor. `SECURITY.md` now describes the shipped auth/rate-limit,
  corpus-provenance and trust-boundary posture.

### Docs
- Docs-truth pass (2026-11-13): contradictions between README / WORKPLAN /
  HANDOFF / ROADMAP / EXPERIMENTS / the vision spec resolved against the
  repo as it actually is — 1M numbers restated as box-only and *not*
  reproducible (0 one-million rows in `runs.jsonl`), WORKPLAN TODO items 3
  and 4 marked done (gates ship in `scripts/gate.py`; the 1M session plan
  exists), item 5 done on the box but not reproducible, item 6 partly done
  (sweep cells recorded, no committed ledger rows), test counts corrected
  (225 passed / 7 deselected; 29 Rust tests; 7 `cloud_only`), ledger row
  count (32), the 10M-vs-5M stretch conflict settled on the spec (5M), the
  dangling `VISION.md` acceptance reference mapped to `ROADMAP.md`, README's
  stdio/HTTP MCP examples fixed (`--port` removed from stdio; `--help`
  removed from the HTTP line; gate-vs-integrity duty stated), and
  `cloud_only` CI coverage described accurately (`|| true`, no GPU/box on
  hosted runners).
- README: the "one config" paragraph now states the reality — `config.py` is
  the single source of truth and `config_loader.py` (Lane B) is a strict,
  optional hardware-profile adaptation layer consumed only via `--config`;
  serving/auth flags documented; test counts corrected.
- Added `docs/MCP_API.md` (tool schema, flags, untrusted-content contract),
  `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`.
- `SECURITY.md` rewritten around the shipped posture (loopback default,
  bearer + rate-limit middleware, redacted errors, corpus provenance,
  provenance-gated reuse, gzip-verified WET downloads).
- Removed the duplicated paragraph; removed the misleading
  `eval/e3_result_box_2026-10-25.json` (a copy of the toy result).
- Marked the 1M box-performance numbers as uncommitted/unreproducible from
  this repo.
- Added `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  `CHANGELOG.md`.

### Tests
- Store: fail-closed on in-place record substitution; concurrent cold reads
  overlap (no global-lock serialization); rescan bounded to the appended tail;
  an unchanged shard is not rescanned on a miss; byte-budgeted LRU;
  `get_many`; bulk handle reuse.
- Config: unknown keys / bad hardware values / `operating_point`-vs-`serving`
  conflicts raise; the loaded operating point drives the served defaults.
- Ingest: dedup correctness + disk-backed structure over 60k records.
- Pipeline: provenance gates for download / ingest / queries (regenerate on
  mismatch); download rejects corrupt same-size gzip.
- MCP: error redaction, untrusted/provenance flags, operating-point
  consumption, rate limiter, bearer check, CLI flags; the HTTP transport is
  pinned with real 401/200/429 requests through `app.http_app()`.
- Fake-backend parity (`tests/test_synth_parity.py`): 256-D embeddings,
  binary-dot similarity ranking, Rust filter semantics, degraded-rerank
  contract.
- ANN recall (`tests/test_ann_recall.py`): exhaustive IVF equals brute force
  on unseen queries; measured recall bars and nprobe monotonicity.
- Gate exit codes (`tests/test_gate_exit.py`): 0 pass / 1 hard-gate
  regression / 2 usage error; a zero p50 is a measurement, an unavailable G2
  is reported as skipped.
- Index manifest excludes the content store at any depth and the build-only
  `centroids.npy`; manifest keys are contained under the index in both the
  Python and Rust loaders.
- `e2e_latency.py` CLI defaults (`max_pair_tokens`, `batch_size`) now read from
  `config.OPERATING_POINT` instead of hardcoded 160/128.
- `build_filter.main` accepts `verbose`/`progress_every` parameters (was a
  `NameError` when imported).
- `measure_recall` raises `ValueError` on duplicate `query_id` in predictions
  and does not exhaust generator inputs.
- `build_index.py` loads vectors with `np.load(..., mmap_mode="r")`.
- `embed_resumable.py` accepts a completed final state (rows == n_docs) without
  requiring `--force`.
- `run_e3.py` guards the empty-question-set `ZeroDivisionError`;
  `ground_truth.py` warns when the random `--nq` ground-truth path is used.
- `gate.py` explicitly reports skipped G1/G2 gates (`skipped`, `skip_reason`,
  `skipped_gates`) instead of silently omitting them.
- `mcp_server.py` `home_search` retrieves at least `top_k` coarse candidates
  (no silent cap at `top_coarse`).
- Ground-truth title decoding: percent-encoding decoded, control characters
  stripped.
- CI: `ruff check` / `ruff format --check` now green; added a
  `workflow_dispatch`/schedule-gated `cloud_only` job with `permissions` and
  `timeout-minutes`.
- New regression test: `AnnIndex` rejects absurd strided shapes (no SIGABRT).
- Removed tautological assertions in `test_config_loader.py` / `test_scaffold.py`.
- `ContentStore.reset()` prunes only the store's own files (foreign files and
  nested dirs survive); index rebuilds leave a store identical in size to a
  fresh build of the same corpus; `--no-store` removes a stale store dir;
  `--no-reset-store` keeps append semantics.

## [0.9.0] - 2026-10-13

Initial public release. See `pyproject.toml`.
