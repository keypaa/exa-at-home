# Item 2 — Resumable chunked embed: design

## Problem
`stage_embed` in `scripts/run_pipeline.py` streams the corpus in 8192-doc
chunks and writes a memmap, but the commit is `os.replace` only at the **end**.
A 1M embed is ~19 min, 5M ~1.5 h. A SIGKILL / spot-preemption / box wipe
at minute 94 throws it all away and restarts from zero. The deliverable is a
restart that resumes from the **last fully committed chunk**, not restarts.

## Constraints (inherited)
- Whole corpus is never held in RAM (streaming boundary = `EMBED_CHUNK`).
- Vectors never all fit in RAM as Python lists (memmap, `w+`).
- No reduction in throughput (item 1b/M9 measured 901 docs/s at bs=1024;
  resuming must add, not multiply, passes over the data).
- Reuse gate already provenance-based (`source_identity` in build_index): if
  source changes mid-flight, we start over — no silent mismatch.
- The old `_skip(vecs_path, force)` reuse path checks mere existence; with
  resume there is now a 3rd state ("partial") that must be detectable.

## Design

### Files on disk during a run
```
vecs.npy                final, only renamed in place on success
vecs.npy.part           memmap w+, final shape (n_docs, dim), pre-allocated
vecs.npy.state.json     {"n_docs": N, "chunk": 8192, "batch": 1024,
                        "source_sha256": "<sha256 of docs.jsonl>",
                        "rows": R, "done": false}
ids.json                final, only renamed in place on success
ids.json.part           partial JSON array written incrementally
ids.json.state          same R; kept separate so the two artifacts can't
                        drift out of sync on a crash
```

### State machine
```
NOT_STARTED  -> state file absent, no .part
IN_PROGRESS  -> state rows R ∈ [0, N), .part valid, ids.part valid
COMPLETE     -> vecs.npy + ids.json present, state file removed
STALE        -> state exists but n_docs/source_sha mismatch -> wipe & restart
```

### Resume (the invariant)
On `stage_embed(docs, vecs, ids, *, force)`:

1. `n = count_lines(docs)` (fast, O(N) but small compared to forward pass).
2. `src_sha = sha256(docs)` streamed.
3. If `force`: delete `*.part*`, state, final targets; continue fresh.
4. Load state if present:
   - if `n != state.n_docs` or `src_sha != state.source_sha256`
     or `state.rows % state.chunk != 0`
     or `R > n` -> delete state + parts + fail fast (provenance mismatch,
       loud, per assemble rule: never reuse a mismatched corpus silently).
   - else open memmap in `'r+'`, seek ids.part to end, set `start_row = R`.
5. Otherwise create `.part` with `open_memmap(..., shape=(n, dim), mode='w+')`,
   write `[`, state rows=0.
6. For each chunk `[start_row::EMBED_CHUNK]`:
   - embed chunk (`emb.encode_docs` with the production batch)
   - write into `mm[rows:rows+len]`, `mm.flush()`
   - append ids to `ids.part`, `flush()` + `os.fsync`
   - update state rows -> atomic replace (write tmp then `os.replace`)
   - only after the atomic state-replace does the next chunk start.
7. On last chunk (rows == n):
   - close mm, `os.replace(.part, vecs.npy)`
   - append `]` to ids.part, `os.replace(ids.part, ids.json)`
   - `os.replace(state, state + ".done")` then delete state (keep done marker
     for 24 h so an accidental re-invocation is a no-op not a re-embed)
8. `_log("embed", ...)` with rows and resume flag.

### Crash semantics (the actual point)
- The only "load-bearing" writes are the memmap data (which lands under the
  page cache and is flushed per chunk) and the **atomic** state rename.
- A crash inside a chunk re-runs that chunk only (`start_row` unchanged until
  the state file commits). Memmap writes for the failed chunk are overwritten
  by the re-run.
- A crash inside the final rename is non-fatal: the `.done` marker or the
  presence of a complete `vecs.npy` + `ids.json` (shape verified) makes the
  next invocation a no-op that only verifies shape and source sha. No re-embed.
- Power loss before the page-cache flush loses at most one chunk; that chunk
  is re-embedded. Acceptance budget: ≤ 8192 docs redone, not the whole corpus.
- Counterfactual vs. today: a 5M run that dies at hour 1h30 loses 8k docs
  instead of 5M.

### Resume on a different machine
Not supported: memmap plus chunk-size binding. The design assumes the same
`EMBED_CHUNK`, `EMBED_BATCH`, source file, and build root. If those differ,
the state check fails and the run restarts. Documented, not prohibited.

### Interaction with run_pipeline.py and D4
- `stage_embed(..., resume=True)` default in the box path; `--no-resume` forces
  the legacy all-or-nothing path for A/B measurement (item 9 wants both).
- `run_pipeline.py toy` gains `--resume` (default true); `run_pipeline.py
  full` keeps `--force` overriding resume.
- The `_skip(vecs_path, force)` reuse check becomes:
  `exists(vecs_path) and exists(state.done) and source_sha matches -> reuse`.
  A partial build is *never* reused silently.
- Source sha is computed once per invocation and stays in the state; if
  `docs.jsonl` changed during a run, the second chunk aborts with a
  fail-fast error rather than corrupting the index.

### Coverage cost
- +~40 lines in `stage_embed` plus a small `.state.py` helper.
- `open_memmap` reuse: the only new error modes are (a) corrupt state json
  (treat as "restart", never silently trust it) and (b) a partial memmap that
  claims rows R but the file is truncated##################
  (compare `mm.shape[0] >= R`; fail fast on mismatch).
- No new Python deps; `hashlib` and `json` already in stdlib.

## Milestone 9 dependency
This item gives the *survivable* version of the embed. It does not raise the
throughput — M9 manual forward did that. It makes the box capable of starting
an 18-hour run without babysitting.

## What we measure (before / after)

### Before (legacy, today)
- 1M simulated, warm cache: ~19 min wall-clock end-to-end.
- Inject `kill -9 <pid>` at a known wall time (e.g. after 120 s).
  - Expected outcome: **no `vecs.npy`, no `ids.json`**, run incomplete.
  - Time lost: full 120 s.
  - Rerun: from 0.

### After (resumable)
- Same, kill at 120 s.
  - Expected outcome: state file shows `rows ≈ 120s × 901 docs/s ≈ 108k`
    (within one 8192-doc chunk of where it was killed, ≤ 8192 docs redone).
  - `vecs.npy.part` exists, valid memmap, dims match.
  - `stage_embed --resume` (or plain rerun, default resumes) picks up and
    finishes; total wall-clock ≤ 19 min + 1 chunk of rework.
  - No duplicate chunk (verify `rows` monotonically increases in
    `state` across the resumed run — a second killing then restarting at
    240 s must show rows at the second point, not overlap with the first).
  - Final `vecs.npy` shape `(n_docs, 256)` and `ids.json` length match; a
    random row's bytes match a re-embedding of that doc (single-doc sanity).
  - `manifest.json` / `source_identity` of the resulting index still
    validates against the original `docs.jsonl`.

### Acceptance criteria
1. Kill at **120 s** + resume + finish → **≤ 1 chunk of wasted work**
   (measured by `rows` in the state file before vs. after kill).
2. Kill at **120 s** + resume produces a complete final index whose gate
   latency is within **+2 ms** of the no-kill run (the re-run is not
   slower enough to matter).
3. `run_pipeline.py toy --resume` succeeds twice in a row on the same
   corpus (idempotency) with 0 changed rows.
4. A corrupted-state-file run **fails fast** rather than silently producing
   an index from scratch or reusing a mismatched base.

### What NOT to do in this item
- Do not change the embedder or batching (M9 is locked).
- Do not change centroid training, the IVF build, or ground truth.
- Do not introduce a new dependency (no sqlite, no msgpack).
- Do not "clean up" the streaming shape: `EMBED_CHUNK` stays 8192 so the
  resume boundary is identical to the memory-containment boundary.

## Ledger row (fill on the box)
```
item=2, metric=wasted_docs_on_kill, before=N (full run), after=<=8192,
config={n_docs, chunk=8192, batch=1024, sha_of_docs, kill_at_s=120, resumed=True}
```
plus throughput `docs_per_s` to confirm no regression vs M9.

## Next items after this
- item 3: gates as one command (using the same state/journal pattern)
- item 4: 1M session plan (can now be written because the build is survivable)
- item 5: the 1M build itself
PY