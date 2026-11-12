# Changelog

## [Unreleased]

### Fixed
- Track `Cargo.lock` so Rust dependency builds are reproducible (was gitignored).
- `AnnIndex::new` now caps matrix elements (`MAX_MATRIX_ELEMENTS`) and returns a
  clean `ValueError` instead of aborting the process on an absurd strided shape.
- `ContentStore._load_persisted_index` contains `offsets.json` shard paths under
  the store root (no absolute/`..` traversal).
- `ContentStore._shard` reads `config.SHARD_COUNT` instead of a hardcoded `64`.
- Index manifest excludes the content store at any depth and is written with
  `encoding="utf-8"`.
- `e2e_latency.py` CLI defaults (`max_pair_tokens`, `batch_size`) now read from
  `config.OPERATING_POINT` instead of hardcoded 160/128.
- `build_filter.main` accepts `verbose`/`progress_every` parameters (was a
  `NameError` when imported).
- `measure_recall` raises `ValueError` on duplicate `query_id` in predictions.
- `build_index.py` loads vectors with `np.load(..., mmap_mode="r")`.
- `embed_resumable.py` accepts a completed final state (rows == n_docs) without
  requiring `--force`.
- `run_e3.py` guards the empty-question-set `ZeroDivisionError` and warns when
  the random `--nq` ground-truth path is used.
- `gate.py` explicitly reports skipped G1/G2 gates (`skipped`, `skip_reason`,
  `skipped_gates`) instead of silently omitting them.
- `mcp_server.py` `home_search` retrieves at least `top_k` coarse candidates
  (no silent cap at `top_coarse`).
- Ground-truth title decoding: percent-encoding decoded, control characters
  stripped.
- CI: `ruff check` / `ruff format --check` now green; added a
  `workflow_dispatch`/schedule-gated `cloud_only` job with `permissions` and
  `timeout-minutes`.

### Docs
- Fixed test counts (117 pytest / 28 cargo, not 77/13) and `cloud_only` count (7, not 6).
- Removed duplicated paragraph; removed misleading `eval/e3_result_box_2026-10-25.json` (copy of toy result).
- Marked the 1 M box-performance numbers as uncommitted/unreproducible from this repo.
- Added `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`.

### Tests
- New regression test: `AnnIndex` rejects absurd strided shapes (no SIGABRT).
- Removed tautological assertions in `test_config_loader.py` / `test_scaffold.py`.

## [0.9.0] - 2026-10-13

Initial public release. See `pyproject.toml`.
