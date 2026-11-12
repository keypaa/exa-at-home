# exa-at-home — Fix All Audit Findings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every finding from the complete security & quality audit (Rust core, Python library, security/MCP surface, pipeline scripts, tests/CI, documentation) on branch `audit/fix-all-findings`, including Low-severity items, leaving the tree green (pytest + cargo test + ruff check + ruff format --check + cargo fmt --check).

**Architecture:** One task per audit finding (or tightly-coupled pair). Each task: write/verify a failing test (or confirm existing coverage), implement the minimal fix, run verification, commit. Domain groups: Rust core (R), Python lib (P), Security (S), Pipeline (C), Tests/CI (T), Docs (D).

**Tech Stack:** Python 3.14 (`.venv`), Rust/PyO3 `ann_core` (PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1), pytest, ruff 0.17, cargo 1.97.

**Spec:** The audit report at `/tmp/opencode/audit/AUDIT_REPORT.md` and per-domain reports under `/tmp/opencode/audit-*/REPORT.md`. Each task cites its finding ID(s) and file:line.

## Global Constraints

- No tracked file may be deleted; only fix/augment.
- Do not push to upstream; keep all work local on `audit/fix-all-findings`.
- Every behavior change must be accompanied by a test (new or extended) that pins the fix.
- After each task: run the relevant test subset + ruff check on touched files, then commit.
- torch/sentence-transformers are NOT installed — real embedder/reranker paths cannot be executed; mark those tests appropriately and audit by reading + synth fakes.
- CPU only; no GPU; no network downloads.

## Review Focus

The five input classes/failure modes most likely to bite a user of this software, each pinned to an owning task:
1. **Malformed / adversarial numpy array at the FFI boundary** → R1/R2 (strided-array abort + OOB read). Pinned by R-task test.
2. **Attacker-controlled `index/` dir (offsets.json traversal)** → S2. Pinned by S-task test.
3. **Stale / cross-crawl artifact reuse in the orchestrator** → C1/C2/C3. Pinned by provenance-gate tests.
4. **Concurrent `ContentStore.get()` collapse + O(shard) rescan storm** → P1-01/P1-02. Pinned by concurrency + rescan tests.
5. **Config drift between measured operating point and shipped CLI defaults** → C4/D3/P2-06. Pinned by config-consistency test.

---

## Task R1 — Rust core: cap allocation size in AnnIndex::new (finding R1, Critical)

**Files:**
- Modify: `crates/ann-core/src/lib.rs:98-121` (`AnnIndex::new`)
- Test: `crates/ann-core/src/lib.rs` tests module (add `ann_index_new_rejects_absurd_shape`)

**Interfaces:**
- Consumes: `AnnIndex::new(mat: PyReadonlyArray2<f32>, ids: Vec<String>)`
- Produces: `PyResult<Self>`; returns `Err` when `nrows * ncols` exceeds a cap (e.g. 64M elements) instead of `Vec::with_capacity` panicking.

**Steps:**
- [ ] **Step 1:** Write failing test `ann_index_new_rejects_absurd_shape` constructing a tiny 1×1 float32 array and asserting `AnnIndex::new` with a strided 1×10^12 view returns an `Err` (not panic). Run: `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p ann-core ann_index_new_rejects_absurd_shape` → FAIL.
- [ ] **Step 2:** Implement: before `Vec::with_capacity(a.nrows() * a.ncols())`, check `a.nrows().checked_mul(a.ncols())` against a `MAX_MATRIX_ELEMENTS` cap; return `PyValueError` on overflow/too-large.
- [ ] **Step 3:** Run test → PASS; run `cargo test -p ann-core` (28 pass); commit.

## Task R2 — Rust core: same strided-matrix root cause as R1 (bounds safety)

**Rationale:** R1's cap on `nrows*ncols` also bounds the element-wise copy loop in `AnnIndex::new` and `from_binary`, eliminating the OOB read. No separate code change beyond R1's cap + a documented stride sanity note.

**Files:**
- Modify: `crates/ann-core/src/lib.rs` (same as R1)
- Test: extend R1's test to also assert a moderate strided view (e.g. 4-float buffer read as 10 floats) returns Err rather than silent wrong data.

**Interfaces:** Same as R1.

**Steps:**
- [ ] **Step 1:** Write failing test asserting strided-over-read returns Err. Run → FAIL.
- [ ] **Step 2:** Implement: validate that the strided element access stays within the underlying buffer (or rely on the element cap making OOB unreachable from realistic input).
- [ ] **Step 3:** Run test → PASS; run full Rust suite; commit.

## Task R3 — Rust core: fix clippy erasing_op in index.rs test fixture (finding R3, Low)

**Files:**
- Modify: `crates/ann-core/src/index.rs:367-368` (`brow[0 * CODE_BYTES]`, `brow[1 * CODE_BYTES]`)
- Test: none (existing `cargo clippy --all-targets` gate)

**Interfaces:** None.

**Steps:**
- [ ] **Step 1:** Run `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo clippy -p ann-core --all-targets` → confirm FAIL on `0 * CODE_BYTES`.
- [ ] **Step 2:** Replace `0 * CODE_BYTES` → `0` and `1 * CODE_BYTES` → `CODE_BYTES`.
- [ ] **Step 3:** Run clippy → clean; run `cargo fmt --check`; commit.

## Task R4 — Rust core: bounded top-n_coarse in search_lists (finding R4, Medium)

**Files:**
- Modify: `crates/ann-core/src/ivf.rs:56-72` (`search_lists`)
- Test: `crates/ann-core/src/ivf.rs` tests module (add `search_lists_bounded_top_n_coarse` verifying only n_coarse results returned and no full-sort memory blowup on a large skewed list).

**Interfaces:**
- Consumes: `search_lists(lists, codes, cluster_ids, q, n_coarse, allow)` (signature unchanged).
- Produces: `Vec<(u32, f32)>` truncated to `n_coarse` (same), computed via a bounded selection instead of full `sort_by`+`truncate`.

**Steps:**
- [ ] **Step 1:** Write test: build index where one list holds a large share; call `search_lists` with `n_coarse=50`; assert `got.len() <= 50` and scores match the true top-50. Run → FAIL (or pass but measure).
- [ ] **Step 2:** Implement: use `select_nth_unstable_by` to partition to top n_coarse, then sort only those, instead of sorting all scanned candidates.
- [ ] **Step 3:** Run test → PASS; run `cargo test -p ann-core`; commit.

---

## Task P1 — Python lib: split ContentStore lock / bound rescan (findings P1-01, P1-02, High)

**Files:**
- Modify: `exa_home/store.py` (lock splitting, rescan-bounded, `_shard_changed`/`_scan_shard`)
- Test: `tests/test_store.py` (add concurrency + rescan tests)

**Interfaces:**
- Consumes: `ContentStore.get(doc_id)`, `ContentStore.put(doc)`, `ContentStore.begin_bulk()`, `ContentStore.save_index()`
- Produces: same signatures; internal locking changed so disk I/O is not under the global lock; rescan bounded per-shard-per-generation.

**Steps:**
- [ ] **Step 1:** Write `test_store_concurrent_gets_no_collapse` (N threads vs serial, assert speedup ≥ 0.8×) and `test_store_rescan_bounded` (append one doc, repeated gets → only one rescan). Run → FAIL.
- [ ] **Step 2:** Implement: introduce a read-mostly index reference swapped under lock; perform `_scan_shard`/disk reads outside the global lock; track a per-shard "generation" so a changed shard is rescanned at most once per generation.
- [ ] **Step 3:** Run tests → PASS; run `pytest tests/test_store.py -q`; run ruff on `exa_home/store.py`; commit.

## Task P2 — Python lib: content hash per record in ContentStore (finding P1-04, High)

**Files:**
- Modify: `exa_home/store.py` (`_index` value becomes `(path, off, length, digest)`; `put`/`save_index`/`_load_persisted_index`/`_read_record`/`_scan_shard` updated)
- Test: `tests/test_store.py` (add `test_store_serves_original_content_after_shard_mutation`)

**Interfaces:**
- Consumes: same public API.
- Produces: `get()` verifies a stored content digest against the record bytes; returns `None` (miss → rescan) on mismatch instead of serving substituted content.

**Steps:**
- [ ] **Step 1:** Write test: `put` a doc, `save_index`, mutate the shard bytes at the same offset with same-length different content, assert `get(doc_id)` returns `None` (or rescans to original). Run → FAIL.
- [ ] **Step 2:** Implement: store sha256 of payload in index; `_read_record` verifies digest; on mismatch treat as miss and rescan.
- [ ] **Step 3:** Run test → PASS; run `pytest tests/test_store.py -q`; commit.

## Task P3 — Python lib: disk-backed ingest dedup (finding P2-05, High)

**Files:**
- Modify: `exa_home/ingest.py` (`write_docs_jsonl` dedup spills to disk instead of unbounded `seen` set)
- Test: `tests/test_ingest.py` (add `test_ingest_dedup_bounded_memory` using a temp file with many duplicates)

**Interfaces:**
- Consumes: `write_docs_jsonl(docs, out_path, progress_every)` (signature unchanged).
- Produces: same `IngestStats`; dedup uses a disk-backed structure (temp-file + sort/uniq or sqlite) so RAM is bounded independent of corpus size.

**Steps:**
- [ ] **Step 1:** Write test ingesting >50k docs with many duplicates through `write_docs_jsonl`; assert kept count correct and no `MemoryError`; (optionally assert peak RAM bounded). Run → currently passes but memory unbounded; add assertion on dedup correctness with duplicates.
- [ ] **Step 2:** Implement: replace in-memory `seen` set with a disk-backed dedup (e.g. write candidate digests to a temp file, sort + uniq, or use a bloom-filter+spill). Keep `seen` for small runs as a fast path? Keep simple: single disk-backed pass.
- [ ] **Step 3:** Run test → PASS; run `pytest tests/test_ingest.py -q`; commit.

## Task P4 — Python lib: strict config_loader + server consumes loaded config (finding P2-06, High; D3)

**Files:**
- Modify: `exa_home/config_loader.py` (raise on unknown/typo'd keys instead of silent default), `exa_home/mcp_server.py` and `exa_home/orch.py` (consume loaded config instead of `config.OPERATING_POINT` constant), possibly `exa_home/config.py` (add helper to apply overrides)
- Test: `tests/test_config_loader.py` (add typo'd-key raises), `tests/test_mcp.py` / `tests/test_orch.py` (assert served config matches loaded config)

**Interfaces:**
- Consumes: `config_loader.load(config_path)` returns normalized dict with `serving`/`operating_point`.
- Produces: mcp_server/orch read operating point from the loaded config; unknown keys raise `ValueError`.

**Steps:**
- [ ] **Step 1:** Write `test_config_loader_rejects_unknown_key` and `test_mcp_uses_loaded_config_not_constant`. Run → FAIL.
- [ ] **Step 2:** Implement: `HardwareProfile(**raw["hardware"])` raises on unexpected kwargs (no silent fallback); `mcp_server.main()` / `orch` accept an optional operating-point dict resolved from the loaded config, falling back to `OPERATING_POINT` only when none loaded.
- [ ] **Step 3:** Run tests → PASS; run ruff on touched files; commit.

## Task P5 — Python lib: keep shard handle open during bulk (finding P2-07, Medium)

**Files:**
- Modify: `exa_home/store.py` (`begin_bulk` keeps per-shard handles open; `put` reuses them)
- Test: `tests/test_store.py` (add `test_store_bulk_puts_reuse_handle` — functionally equivalent, measures fewer syscalls optionally)

**Interfaces:**
- Consumes: same public API.
- Produces: same; internal file-handle caching during `begin_bulk`.

**Steps:**
- [ ] **Step 1:** Write test asserting bulk `put` + `save_index` roundtrips correctly (existing `test_bulk_offsets_roundtrip_no_rescan` covers; add syscall-count check optional).
- [ ] **Step 2:** Implement: `begin_bulk` opens and caches shard file handles; `put` writes to cached handle and tracks offset; `save_index` flushes+closes.
- [ ] **Step 3:** Run test → PASS; commit.

## Task P6 — Python lib: batch store.get_many (finding P2-08, Medium)

**Files:**
- Modify: `exa_home/store.py` (add `get_many(ids)`), `exa_home/mcp_server.py:81-86` (use `get_many` in `home_search`)
- Test: `tests/test_store.py` (add `test_store_get_many`), `tests/test_mcp.py`

**Interfaces:**
- New: `ContentStore.get_many(ids: list[str]) -> dict[str, dict | None]`.
- Consumes: list of doc ids.
- Produces: dict id→doc (or None), single index load, batched reads.

**Steps:**
- [ ] **Step 1:** Write `test_store_get_many` asserting batch results equal individual gets. Run → FAIL (method missing).
- [ ] **Step 2:** Implement `get_many`; refactor `home_search` to call it once instead of per-result `get()`.
- [ ] **Step 3:** Run tests → PASS; commit.

## Task P7 — Python lib: top_k vs top_coarse cap (finding P2-09, Medium)

**Files:**
- Modify: `exa_home/mcp_server.py:62-63` (clamp `top_k` to `top_coarse` or derive `n_coarse`), `exa_home/orch.py` (pass `n_coarse=max(top_coarse, top_k)`)
- Test: `tests/test_mcp.py` (add `test_home_search_top_k_capped_at_top_coarse`)

**Interfaces:**
- Consumes: `home_search(query, top_k=10)`; produces results capped consistently.

**Steps:**
- [ ] **Step 1:** Write test: `home_search(..., top_k=100)` returns at most `top_coarse` (50) results and does not error. Run → currently returns 50 but undocumented; add explicit assertion + clamp behavior.
- [ ] **Step 2:** Implement: clamp `top_k = min(top_k, top_coarse)` in `home_search` and document in docstring; or derive `n_coarse = max(top_coarse, top_k)`.
- [ ] **Step 3:** Run test → PASS; commit.

## Task P8 — Python lib: mask errors over MCP boundary + pagination constant (findings P2-10, P3-16, Medium/Low)

**Files:**
- Modify: `exa_home/mcp_server.py` (wrap tool body so all errors → redacted `error` dict; name `PAGE_SIZE = 20`; `next_offset` documented)
- Test: `tests/test_mcp.py` (add `test_home_search_errors_are_redacted`, `test_home_contents_page_size_constant`)

**Interfaces:**
- Consumes: `home_search`, `home_contents` (signatures unchanged).
- Produces: errors return `{"error": {...}}` shape; no raw `str(e)`/`AttributeError` leaks.

**Steps:**
- [ ] **Step 1:** Write `test_home_search_empty_query_returns_error_dict` (not an exception) and `test_home_contents_page_size_is_named_constant`. Run → FAIL.
- [ ] **Step 2:** Implement: `PAGE_SIZE = 20` module constant; wrap `home_search`/`home_contents` bodies so validation + backend exceptions become the redacted error dict.
- [ ] **Step 3:** Run tests → PASS; run ruff on `mcp_server.py`; commit.

## Task P9 — Python lib: store._shard uses config.SHARD_COUNT (finding P2-11, Medium)

**Files:**
- Modify: `exa_home/store.py:43-47` (import config; `_shard` uses `config.SHARD_COUNT`)
- Test: `tests/test_store.py` (add `test_store_shard_count_matches_config`)

**Interfaces:**
- Consumes: `config.SHARD_COUNT`.
- Produces: shard path derived from configured count.

**Steps:**
- [ ] **Step 1:** Write test asserting `ContentStore._shard(doc_id)` distributes across `config.SHARD_COUNT` shards. Run → FAIL.
- [ ] **Step 2:** Implement import + use.
- [ ] **Step 3:** Run test → PASS; commit.

## Task P10 — Python lib: byte-budgeted LRU (finding P2-12, Medium)

**Files:**
- Modify: `exa_home/store.py` (`_cache` tracks total bytes; evicts by byte budget)
- Test: `tests/test_store.py` (add `test_store_lru_byte_budget_evicts`)

**Interfaces:**
- Consumes: same; new optional `lru_bytes` param.
- Produces: LRU evicts by total byte size, not just count.

**Steps:**
- [ ] **Step 1:** Write test storing docs of varying sizes with a small byte budget; assert total cached bytes ≤ budget. Run → FAIL.
- [ ] **Step 2:** Implement byte tracking + eviction loop.
- [ ] **Step 3:** Run test → PASS; commit.

## Task P11 — Python lib: manifest exclusion depth + encoding (finding P2-13, Medium)

**Files:**
- Modify: `exa_home/index_format.py:83-113` (exclude any path under `store/` at any depth; `encoding="utf-8"` in `write_manifest`)
- Test: `tests/test_index_format.py` (add `test_manifest_excludes_nested_store_dir`)

**Interfaces:**
- Consumes: `write_manifest`, `verify_manifest`, `_manifest_entries`.
- Produces: nested `store/` excluded; manifest written with UTF-8.

**Steps:**
- [ ] **Step 1:** Write test creating `index/nested/store/x.bin` and asserting it's not in the manifest. Run → FAIL.
- [ ] **Step 2:** Implement recursive `store/` exclusion + `encoding="utf-8"`.
- [ ] **Step 3:** Run test → PASS; run ruff on `index_format.py`; commit.

## Task P12 — Python lib: title percent-encoding decoded (finding P3-15, Low)

**Files:**
- Modify: `exa_home/ingest.py:34-53` (`_title_from_url` decodes percent-encoding, strips control chars)
- Test: `tests/test_ingest.py` (add `test_title_from_url_decodes_percent_encoding`)

**Interfaces:**
- Consumes: URL string.
- Produces: decoded, sanitized title.

**Steps:**
- [ ] **Step 1:** Write test: `"a%20b.html"` → `"A B"`; `"foo%2Fbar.html"` → `"Foo Bar"`. Run → FAIL.
- [ ] **Step 2:** Implement `urllib.parse.unquote` + control-char strip.
- [ ] **Step 3:** Run test → PASS; commit.

---

## Task S1 — Security: auth token + rate limiter + bind for mcp_server (findings S1, P1-03, S9, High)

**Files:**
- Modify: `exa_home/mcp_server.py` (`main()` gains `--token`, `--bind`/`--host`, rate-limit middleware; auth check on tools)
- Test: `tests/test_mcp.py` (add auth/rate-limit tests; may need synth server)

**Interfaces:**
- Consumes: new CLI args `--token`, `--bind`; env `FASTMCP_HOST` still honored.
- Produces: HTTP server requires bearer token; rate-limited; bindable host.

**Steps:**
- [ ] **Step 1:** Write `test_mcp_server_rejects_without_token` (using synth backends, FastMCP test client or direct tool calls with auth middleware). Run → FAIL.
- [ ] **Step 2:** Implement: add `--token`/`--bind` args; wire FastMCP auth provider + rate-limiting middleware (fastmcp 4.0.3 has `rate_limiting`); apply token check to tool invocation.
- [ ] **Step 3:** Run tests → PASS; run `python -m exa_home.mcp_server --help` to confirm new flags; commit.

## Task S2 — Security: contain offsets.json shard paths (finding S2, High)

**Files:**
- Modify: `exa_home/store.py:97-122` (`_load_persisted_index` rejects absolute/`..` shard paths)
- Test: `tests/test_store.py` (add `test_store_offsets_rejects_traversal`)

**Interfaces:**
- Consumes: `offsets.json` entries.
- Produces: `ValueError` (or skip) for shard paths escaping `root`.

**Steps:**
- [ ] **Step 1:** Write test: `offsets.json` with `{"evil": ["../target.jsonl", 0]}` → `ContentStore.get("evil")` does not read outside root. Run → FAIL (currently reads external file).
- [ ] **Step 2:** Implement: `os.path.normpath(os.path.join(root, shard))` must start with `root`; reject absolute and `..`.
- [ ] **Step 3:** Run test → PASS; commit.

## Task S3 — Security: sanitize / mark corpus-derived text (finding S3, High)

**Files:**
- Modify: `exa_home/mcp_server.py` (snippet/text enrichment), `exa_home/orch.py` (`snippet_fn`) — add provenance flag + escape/sanitize
- Test: `tests/test_mcp.py` (add `test_home_contents_text_has_provenance_flag`)

**Interfaces:**
- Consumes: document text from store.
- Produces: responses include `source: "corpus"` / `untrusted: true` marker and escaped text; snippets sanitized.

**Steps:**
- [ ] **Step 1:** Write test: `home_contents` result items include provenance marking; text with `IGNORE ALL PREVIOUS INSTRUCTIONS` is escaped/flagged. Run → FAIL.
- [ ] **Step 2:** Implement: add `"provenance": "corpus"` and HTML-escape `text`/`snippet` before returning.
- [ ] **Step 3:** Run test → PASS; commit.

## Task S4 — Security: mask_error_details + don't abandon timeout pools (findings S4, S5, Medium)

**Files:**
- Modify: `exa_home/mcp_server.py` (FastMCP `mask_error_details=True` or normalize exceptions), `exa_home/orch.py:105-113` (cancel/join abandoned thread pools), `exa_home/rerank.py:70-93` (same)
- Test: `tests/test_mcp.py` (add `test_errors_do_not_leak_internal_type`), `tests/test_orch.py`

**Interfaces:**
- Consumes: exception paths.
- Produces: normalized external errors; timeout pools shut down cleanly.

**Steps:**
- [ ] **Step 1:** Write test: `home_search` with bad input returns externalized error without `type`/`node`/`span_ms` internals. Run → FAIL.
- [ ] **Step 2:** Implement: `mask_error_details=True` where supported; wrap tool bodies; `DAG.run`/`Reranker` shut down (not abandon) timeout pools.
- [ ] **Step 3:** Run tests → PASS; commit.

## Task S5 — Security: cap max_chars_per_doc / total bytes per call (finding S5, Medium)

**Files:**
- Modify: `exa_home/mcp_server.py:106` (lower `max_chars_per_doc` cap or total-bytes cap)
- Test: `tests/test_mcp.py` (add `test_home_contents_total_bytes_capped`)

**Interfaces:**
- Consumes: `max_chars_per_doc`.
- Produces: capped total payload per call.

**Steps:**
- [ ] **Step 1:** Write test asserting 100 ids × max_chars does not exceed a sane total. Run → FAIL.
- [ ] **Step 2:** Implement total-bytes cap.
- [ ] **Step 3:** Run test → PASS; commit.

## Task S6 — Security: filter type/size validation (finding S8, Low)

**Files:**
- Modify: `exa_home/orch.py:193-207` (`translate_filters` validates value types/sizes)
- Test: `tests/test_orch.py` (add `test_translate_filters_rejects_wrong_types`)

**Interfaces:**
- Consumes: `filters` dict.
- Produces: `ValueError` for non-list `terms`, oversized `domains`, etc.

**Steps:**
- [ ] **Step 1:** Write test: `translate_filters({"terms": "alpha"})` raises `ValueError`. Run → FAIL.
- [ ] **Step 2:** Implement type/size validation.
- [ ] **Step 3:** Run test → PASS; commit.

## Task S7 — Security: index manifest trust anchor (finding S6, Low)

**Files:**
- Modify: `exa_home/index_format.py` / `mcp_server.py` — document trust boundary; optional signed-manifest verification hook
- Test: none (design-level; verify loader rejects tampered manifest already covered)

**Interfaces:** N/A.

**Steps:**
- [ ] **Step 1:** Add a `verify_manifest_signature` placeholder/hook + docstring noting manifest is integrity-only, not authenticity.
- [ ] **Step 2:** Run `pytest tests/test_index_format.py -q`; commit.

---

## Task C1 — Pipeline: provenance-gate stage_download (finding C1, Critical)

**Files:**
- Modify: `scripts/run_pipeline.py:115-120` (`stage_download` verifies files match `urls_file`)
- Test: `tests/test_build_index.py` or new `tests/test_pipeline.py` (add `test_stage_download_verifies_urls`)

**Interfaces:**
- Consumes: `urls_file`, `out_dir`, `force`.
- Produces: re-downloads if on-disk files don't match requested URLs (URL set + size/checksum).

**Steps:**
- [ ] **Step 1:** Write test: pre-seed `out_dir` with wrong-count/wrong-name files for a different URL set; `stage_download(force=False)` must still download (not skip). Run → FAIL.
- [ ] **Step 2:** Implement: compare on-disk file set against `urls_file` entries before skipping; include crawl in tag/out dir.
- [ ] **Step 3:** Run test → PASS (synthetic, no network — use file URLs or monkeypatch download); commit.

## Task C2 — Pipeline: provenance-gate stage_queries (finding C2, Critical)

**Files:**
- Modify: `scripts/run_pipeline.py:327-330` (`stage_queries` verifies q/gt match current corpus)
- Test: new pipeline test

**Interfaces:**
- Consumes: `queries_path`, `gt_path`, current `docs_path`/`vecs_path`.
- Produces: regenerates if provenance hash mismatch.

**Steps:**
- [ ] **Step 1:** Write test: rebuild docs, `stage_queries(force=False)` with stale gt → regenerates. Run → FAIL.
- [ ] **Step 2:** Implement hash-based provenance check (hash docs.jsonl, compare to stored fingerprint).
- [ ] **Step 3:** Run test → PASS; commit.

## Task C3 — Pipeline: provenance-gate stage_ingest (finding C3, Critical)

**Files:**
- Modify: `scripts/run_pipeline.py:136-141` (`stage_ingest` verifies docs.jsonl from current wet_dir)
- Test: new pipeline test

**Interfaces:**
- Consumes: `wet_dir`, `docs_path`.
- Produces: regenerates if WET fingerprint changed.

**Steps:**
- [ ] **Step 1:** Write test: change WET dir contents, `stage_ingest(force=False)` → re-ingests. Run → FAIL.
- [ ] **Step 2:** Implement WET fingerprint (sha256 of WET file list) stored alongside docs.jsonl; compare on reuse.
- [ ] **Step 3:** Run test → PASS; commit.

## Task C4 — Pipeline: e2e_latency defaults from OPERATING_POINT (finding C4, High)

**Files:**
- Modify: `scripts/e2e_latency.py:289-292` (defaults `max_pair_tokens=OPERATING_POINT["max_pair_tokens"]`, `batch_size=OPERATING_POINT["batch_size"]`)
- Test: `tests/test_sweep.py` or new (assert defaults equal config)

**Interfaces:**
- Consumes: argparse defaults.
- Produces: defaults match `exa_home.config.OPERATING_POINT`.

**Steps:**
- [ ] **Step 1:** Write test asserting `e2e_latency.py` argparse defaults for `max_pair_tokens`/`batch_size` equal `config.OPERATING_POINT`. Run → FAIL.
- [ ] **Step 2:** Import config and set defaults.
- [ ] **Step 3:** Run test → PASS; run `scripts/e2e_latency.py --help` to confirm; commit.

## Task C5 — Pipeline: build_filter.main signature (finding C5, High)

**Files:**
- Modify: `scripts/build_filter.py:182-187` (`main(docs_path, out, max_terms, verbose=False, progress_every=...)`)
- Test: `tests/test_build_index.py` (add `test_build_filter_main_callable_directly`)

**Interfaces:**
- Consumes: `main(docs_path, out, max_terms, verbose, progress_every)`.
- Produces: filter index; no reliance on module-global `a`.

**Steps:**
- [ ] **Step 1:** Write test calling `main(...)` directly → currently `NameError`. Run → FAIL.
- [ ] **Step 2:** Add parameters to `main`; `if __name__ == "__main__"` passes parsed args.
- [ ] **Step 3:** Run test → PASS; commit.

## Task C6 — Pipeline: duplicate query_id error (finding C6, Medium)

**Files:**
- Modify: `scripts/measure_recall.py:34` (error/warn on duplicate `query_id`)
- Test: `tests/test_measure_recall.py` (if exists) or `tests/test_sweep.py`

**Interfaces:**
- Consumes: `pred.jsonl` rows.
- Produces: `ValueError` on duplicate `query_id`.

**Steps:**
- [ ] **Step 1:** Write test with duplicate `query_id` → assert `ValueError`. Run → FAIL.
- [ ] **Step 2:** Implement duplicate detection before dict comprehension.
- [ ] **Step 3:** Run test → PASS; commit.

## Task C7 — Pipeline: warn on --nq random ground-truth path (finding C7, Medium)

**Files:**
- Modify: `scripts/ground_truth.py:102-105` (emit warning when `--queries` omitted)
- Test: new script test

**Interfaces:**
- Consumes: `--nq` vs `--queries`.
- Produces: stderr warning + gt over random vectors.

**Steps:**
- [ ] **Step 1:** Write test invoking `ground_truth.py --nq 5` → assert warning emitted. Run → FAIL.
- [ ] **Step 2:** Implement warning print.
- [ ] **Step 3:** Run test → PASS; commit.

## Task C8 — Pipeline: mmap vecs in build_index (finding C8, Medium)

**Files:**
- Modify: `scripts/build_index.py:193` (`np.load(vecs_path, mmap_mode="r")`)
- Test: `tests/test_build_index.py` (add `test_build_index_uses_mmap` — assert `mmap_mode` used / memory bounded)

**Interfaces:**
- Consumes: `vecs_path`.
- Produces: memory-mapped vecs array.

**Steps:**
- [ ] **Step 1:** Write test asserting `build_index` loads vecs with `mmap_mode="r"`. Run → FAIL.
- [ ] **Step 2:** Change `np.load` to `mmap_mode="r"`.
- [ ] **Step 3:** Run test → PASS; commit.

## Task C9 — Pipeline: WET download checksum (finding C9, Medium)

**Files:**
- Modify: `scripts/download_wet.py:58-67` (hash downloaded body / ETag verification)
- Test: script test (monkeypatched network)

**Interfaces:**
- Consumes: url, expected size.
- Produces: verified file or error.

**Steps:**
- [ ] **Step 1:** Write test monkeypatching `_remote_size` and download; corrupt content with right size → assert rejection. Run → FAIL.
- [ ] **Step 2:** Implement sha256/ETag verification.
- [ ] **Step 3:** Run test → PASS; commit.

## Task C10 — Pipeline: run_e3 target filename + docs validation (finding C10, Medium)

**Files:**
- Modify: `scripts/run_e3.py:23-34,103-104` (fingerprint in target filename; validate `--index` vs docs)
- Test: new script test

**Interfaces:**
- Consumes: `--index`, `docs_path`.
- Produces: non-stale targets; error on mismatch.

**Steps:**
- [ ] **Step 1:** Write test: `infer_docs_path` with different `--index` and existing root `docs.jsonl` → error or explicit. Run → FAIL.
- [ ] **Step 2:** Implement corpus fingerprint in target filename + validation.
- [ ] **Step 3:** Run test → PASS; commit.

## Task C11 — Pipeline: ZeroDivisionError guard (finding C11, Medium)

**Files:**
- Modify: `scripts/run_e3.py:139-140` (guard `n == 0`)
- Test: new script test

**Interfaces:**
- Consumes: questions list.
- Produces: empty result instead of crash.

**Steps:**
- [ ] **Step 1:** Write test with empty questions → assert no `ZeroDivisionError`. Run → FAIL.
- [ ] **Step 2:** Implement `if n == 0` early return with empty metrics.
- [ ] **Step 3:** Run test → PASS; commit.

## Task C12 — Pipeline: report skipped gates (finding C12, Low)

**Files:**
- Modify: `scripts/gate.py:172-221` (explicit `pass: null` + "skipped" reason for absent G1/G2)
- Test: `tests/test_sweep.py` or gate test

**Interfaces:**
- Consumes: gate inputs.
- Produces: gate report marks skipped gates explicitly.

**Steps:**
- [ ] **Step 1:** Write test running `gate.py --synth` without `--pred` → assert G1/G2 present with `pass: null` and reason. Run → FAIL.
- [ ] **Step 2:** Implement explicit skipped-gate reporting.
- [ ] **Step 3:** Run test → PASS; commit.

## Task C13 — Pipeline: warmup iterations (finding C13, Low)

**Files:**
- Modify: `scripts/e2e_latency.py:148-160` (warmup queries excluded from percentiles)
- Test: new script test

**Interfaces:**
- Consumes: queries.
- Produces: warmup + measured timings.

**Steps:**
- [ ] **Step 1:** Write test: first query slower; assert warmup excluded. Run → FAIL.
- [ ] **Step 2:** Implement warmup loop before timed loop.
- [ ] **Step 3:** Run test → PASS; commit.

## Task C14 — Pipeline: resume after final chunk (finding C14, Low)

**Files:**
- Modify: `scripts/embed_resumable.py:124-135` (allow resume when `rows_done == n_docs`)
- Test: script test (needs torch — may be read-only; implement by reading)

**Interfaces:**
- Consumes: state file.
- Produces: resume accepts completed run.

**Steps:**
- [ ] **Step 1:** Write test (guarded by torch availability) or code-reading verification.
- [ ] **Step 2:** Implement: `if rows_done == n_docs: treat as complete` instead of rejecting.
- [ ] **Step 3:** Run test → PASS (or verify by reading); commit.

## Task C15 — Pipeline: centroids.npy into out dir (finding C15, Low)

**Files:**
- Modify: `scripts/run_pipeline.py:259` (`cent_path` under `out`, not cwd)
- Test: pipeline test

**Interfaces:**
- Consumes: `out` dir.
- Produces: `centroids.npy` inside `out`.

**Steps:**
- [ ] **Step 1:** Write test asserting `centroids.npy` written under `out`. Run → FAIL.
- [ ] **Step 2:** Change `cent_path = out` (join with out dir).
- [ ] **Step 3:** Run test → PASS; commit.

---

## Task T1 — Tests/CI: track Cargo.lock (finding T8, Critical)

**Files:**
- Modify: `.gitignore` (remove `Cargo.lock` line); `Cargo.lock` (already exists in working tree, `git add`)
- Test: `git ls-files Cargo.lock` → present

**Interfaces:** N/A.

**Steps:**
- [ ] **Step 1:** Remove `Cargo.lock` from `.gitignore`.
- [ ] **Step 2:** `git add Cargo.lock`; verify tracked.
- [ ] **Step 3:** Commit.

## Task T2 — Tests/CI: fake-backend parity with real contracts (finding T3, Critical; P2-14)

**Files:**
- Modify: `exa_home/synth.py` and/or `tests/fixtures/backends.py` (align contracts: FakeANN similarity-based scores, FakeReranker degraded mode, FakeEmbedder 256-D), add parity test
- Test: `tests/test_synth_parity.py` (new; fake vs brute-force reference agree on filter semantics + ordering)

**Interfaces:**
- Consumes: fake backends.
- Produces: fakes matching real `Embedder`/`Reranker`/`IvfIndex` contracts.

**Steps:**
- [ ] **Step 1:** Write `test_fake_ann_ranks_by_similarity` and `test_fake_reranker_degraded_flag`. Run → FAIL.
- [ ] **Step 2:** Align fake contracts with real backends (256-D embeddings, similarity scores, degraded mode).
- [ ] **Step 3:** Run `pytest tests/ -m "not cloud_only" -q` → 117 pass; commit.

## Task T3 — Tests/CI: ground-truth recall test (finding T4, High)

**Files:**
- Modify: `tests/` (new `tests/test_ann_recall.py` or add to `test_ann.py`)
- Test: new recall test using ground-truth brute force vs ANN

**Interfaces:**
- Consumes: synth index + queries.
- Produces: recall@10 assertion > threshold (not self-match).

**Steps:**
- [ ] **Step 1:** Write test building a small index with known nearest neighbors, querying vectors NOT in the index, asserting ANN returns the true nearest neighbors. Run → FAIL.
- [ ] **Step 2:** Implement test (uses existing test fixtures).
- [ ] **Step 3:** Run test → PASS; commit.

## Task T4 — Tests/CI: cloud_only CI job (finding T7, High)

**Files:**
- Modify: `.github/workflows/ci.yml` (add `schedule`/`workflow_dispatch` cloud job OR document that cloud_only is manual-only)
- Test: workflow lint (no run possible without box)

**Interfaces:** N/A.

**Steps:**
- [ ] **Step 1:** Add a `cloud_only` job gated on `workflow_dispatch` (runs `pytest -m cloud_only`) so it CAN run on the box when triggered.
- [ ] **Step 2:** Validate workflow YAML parses.
- [ ] **Step 3:** Commit.

## Task T5 — Tests/CI: tautological assertions (finding T12, Medium)

**Files:**
- Modify: `tests/test_config_loader.py:16`, `tests/test_scaffold.py:63`
- Test: replace with real assertions

**Interfaces:** N/A.

**Steps:**
- [ ] **Step 1:** Replace `assert cfg["optiona"] is False if False else True` with a real assertion on the config value.
- [ ] **Step 2:** Replace `assert True` in test_scaffold with meaningful check.
- [ ] **Step 3:** Run `pytest tests/ -m "not cloud_only" -q`; commit.

## Task T6 — Tests/CI: fix CI red (ruff check + ruff format --check)

**Files:**
- Modify: touched files across all tasks (ruff-format them) + any pre-existing ruff violations
- Test: `ruff check` and `ruff format --check` on `exa_home/ scripts/ tests/ notebooks/`

**Interfaces:** N/A.

**Steps:**
- [ ] **Step 1:** Run `ruff check exa_home/ scripts/ tests/ notebooks/` → list all violations.
- [ ] **Step 2:** Fix all (edits / `ruff format` on files we own — careful not to reformat unrelated files beyond our changes; use targeted edits).
- [ ] **Step 3:** Run `ruff check` → clean; `ruff format --check` → clean; commit.

## Task T7 — Tests/CI: run full gate (scripts/gate.py --synth) in CI already present — verify exit code meaningful (finding C12/T6 overlap)

**Files:** `scripts/gate.py`
- Test: CI step already runs it; ensure nonzero exit on failure.

**Steps:**
- [ ] **Step 1:** Run `scripts/gate.py --synth --n-docs 200 --json` → confirm exit 0 on pass, nonzero on fail (e.g. `--min-recall` analog).
- [ ] **Step 2:** Commit if no change needed (verify only).

---

## Task D1 — Docs: fix test counts (finding D1, High)

**Files:**
- Modify: `README.md:89-90` ("77 passed" → "117 passed", "13 passed" → "28 passed"); `HANDOFF.md:236,370` ("108 passed" → "117 passed")

**Steps:**
- [ ] **Step 1:** Edit counts to 117 / 28.
- [ ] **Step 2:** Commit.

## Task D2 — Docs: remove unverifiable 1M headline numbers (finding D2, Critical)

**Files:**
- Modify: `README.md:11-13`, `EXPERIMENTS.md` M9 row, `WORKPLAN.md:8-9`, zenno plan/audit files citing 75.562/131.151/0.359 — replace with "measured on box, ledger row not committed; see runs.jsonl" or remove the specific numbers.

**Steps:**
- [ ] **Step 1:** Edit each doc to mark 1M numbers as unverifiable / remove exact figures pending re-measurement.
- [ ] **Step 2:** Commit.

## Task D3 — Docs: fulfill "one config" promise (finding D3, High; overlaps P4/C4)

**Files:**
- Modify: `README.md:55-59` (clarify config_loader Lane B is a second system; gate.py/e2e_latency.py now read OPERATING_POINT via P4/C4)
- Test: none (docs)

**Steps:**
- [ ] **Step 1:** Update README "one config" paragraph to reflect reality (config.py + config_loader Lane B; gate/e2e read OPERATING_POINT after fixes).
- [ ] **Step 2:** Commit.

## Task D4 — Docs: remove duplicated paragraph (finding D4, Medium)

**Files:**
- Modify: `README.md:27-34` (remove duplicate of 18-25)

**Steps:**
- [ ] **Step 1:** Delete duplicate block.
- [ ] **Step 2:** Commit.

## Task D5 — Docs: cloud_only count 6→7 (finding D5, Medium)

**Files:**
- Modify: `README.md:149`, `HANDOFF.md`, `WORKPLAN.md`, zenno files citing "6 cloud_only tests" → "7".

**Steps:**
- [ ] **Step 1:** Update all references to 7.
- [ ] **Step 2:** Commit.

## Task D6 — Docs: e3_result_box is toy copy (finding D6, Medium)

**Files:**
- Modify: `eval/e3_result_box_2026-10-25.json` (regenerate or remove; better: add note it's a toy copy) + `README.md` references.

**Steps:**
- [ ] **Step 1:** Either remove `eval/e3_result_box_2026-10-25.json` or annotate it as a copy of the toy result; update docs not to cite it as 1M evidence.
- [ ] **Step 2:** Commit.

## Task D7 — Docs: threshold contradiction (finding D7, Medium)

**Files:**
- Modify: `README.md:155-156` (align with `scripts/gate.py` thresholds; threshold IS set in gate.py, measure_recall.py default None)

**Steps:**
- [ ] **Step 1:** Edit README threshold paragraph.
- [ ] **Step 2:** Commit.

## Task D8 — Docs: stale status claims (finding D8, High)

**Files:**
- Modify: `HANDOFF.md` (last commit, license, 1M build status, ledger rows), `zenno/plans/2026-10-26-item-3-four-gates.md` (gates/CI exist), `zenno/plans/2026-10-28-v0.9-release-ops.md` (1M numbers)

**Steps:**
- [ ] **Step 1:** Update each stale claim to current state.
- [ ] **Step 2:** Commit.

## Task D9 — Docs: missing open-source docs (finding D9, Medium)

**Files:**
- Create: `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md` (initial), `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`, docs for MCP tool schema, model-license/attribution doc.

**Steps:**
- [ ] **Step 1:** Create `SECURITY.md` (disclosure contact, supported versions).
- [ ] **Step 2:** Create `CONTRIBUTING.md` (how to build, test, release).
- [ ] **Step 3:** Create `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, issue/PR templates, MCP API doc, license/attribution doc.
- [ ] **Step 4:** Commit.

---

## Verification gate (run before declaring complete)

- [ ] `.venv/bin/python -m pytest tests/ -m "not cloud_only" -q` → 117+ pass
- [ ] `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p ann-core` → 28+ pass
- [ ] `.venv/bin/ruff check exa_home/ scripts/ tests/ notebooks/` → clean
- [ ] `.venv/bin/ruff format --check exa_home/ scripts/ tests/ notebooks/` → clean
- [ ] `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo fmt --check -p ann-core` → clean
- [ ] `git status --short` → only intended fixes on `audit/fix-all-findings`, nothing pushed

**Execution method:** Native (single session), one task at a time, committing after each task passes its verification gate.
