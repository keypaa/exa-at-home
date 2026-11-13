# Scale plan — 5M → multi-billions, without a rewrite

This is the engineering map for taking `exa-at-home` from the current
5M-doc single index to **1B–10B docs** while keeping the codebase modular
and the MCP surface stable. The rule is: **add modules and a shard layer;
do not rewrite the core index format or the pipeline.**

## 1. Where the current design stops

The 5M path is a single `index/` directory loaded by
`crates/ann-core/src/index.rs::load`:

| Artifact | Current loader | Limit |
|---|---|---|
| `codes.bin` (32 B/row) | `fs::read` → `Vec<[u8; 32]>` | 1B = 32 GB RAM; 10B = 320 GB |
| `lists.bin` (postings) | `fs::read` → `Vec<Vec<u32>>` | 1B ≈ 4 GB + per-list overhead; 10B impossible |
| `doc_ids.json` | parsed into `Vec<String>` | 1B ids ≈ 20–40 GB + parse time |
| `centroids.f32` | `Vec<f32>` | K=1M → 1 GB; okay, but K-means cost grows |
| build | one process | 1B docs needs a distributed build |
| ground truth | `vecs @ q` over all rows | 200 queries × 1B = impossible |
| MCP server | one `IvfIndex` | no shard routing |

The first three are the hard wall. The fix is not a bigger machine; it is
**mmap + sharding**.

## 2. Target architecture (modular)

### 2.1 Index format v2 (same directory, new files)

Keep the v1 files for compatibility; add:

```
index/
  manifest.json          # version 2, shard-aware
  centroids.f32          # global route centroids (K_route × 256)
  shard-0000/
    manifest.json        # version 2, per-shard hashes/counts
    centroids.f32        # shard-local centroids (K_s × 256)
    codes.bin            # mmap'd, n_s × 32
    lists.bin            # mmap'd; offset table in lists.idx
    lists.idx            # u64 offsets: K_s + 1 entries
    doc_ids.bin          # u32 offsets + UTF-8 bytes (or Arrow IPC)
  shards.json            # [{dir, first_row, n_docs, sha256}]
```

- **mmap everything**: `memmap2` in Rust; `Loaded` borrows the maps instead
  of owning `Vec`s. `search_lists` takes `&[[u8; 32]]` and a posting slice
  `&[u32]`; no full-file read.
- **`doc_ids.bin`** replaces `doc_ids.json` for large indexes; the Python
  verifier checks its sha256; a small `doc_ids.json` remains for the toy.
- **`lists.idx`** lets the loader seek list `c` without scanning from zero.
- The manifest stays the integrity anchor; v1 loaders keep working.

### 2.2 Shard layer (`exa_home/shards.py`)

A thin Python module, no core changes:

```
ShardSet(root)
  .search(vec, nprobe, n_coarse, filters) -> [(id, score)]
  .contents(ids) -> {id: doc}
```

- Loads `shards.json`; each shard is an `IvfIndex`.
- A query is routed to the `n_shard_probe` nearest shards using the global
  `centroids.f32` (`route` over shard centroids), then each selected shard is
  searched and the top `n_coarse` merged.
- `contents` maps an id to its shard by the shard id ranges (or a small
  `id → shard` index).
- The MCP server gains an optional `--shards <dir>`; without it, it keeps
  using a single `IvfIndex` (no API change).

This is the key modularity property: the existing `IvfIndex`, store, reranker,
DAG, and MCP tools do not change; `ShardSet` is a new consumer.

### 2.3 Build (`scripts/build_shards.py`)

- Split `ids.json` / `vecs.npy` into row ranges (e.g. 8M rows/shard at 1B).
- Train a **global route** K-means on a sample (K_route ≈ 1024–4096).
- For each shard: assign rows to shard-local centroids (K_s ≈ sqrt(n_s) ×
  constant), write the v2 files, verify, write `shards.json`.
- Shards build independently → process pool / multi-machine. The current
  `build_index.main` stays the single-index builder; `build_shards` calls it
  per shard with a shard-local output dir.

### 2.4 Ground truth / recall

- Exact `vecs @ q` only on a **sample** (e.g. 1M rows) or one shard.
- `measure_recall` gains `--sample N` / `--shard`; the gate reports the
  sample size. The full-corpus exact recall is no longer the metric at 1B.

### 2.5 Scale ladder

| Docs | Layout | K_route | Shards | Notes |
|---|---|---|---|---|
| 5M | single index | 10k–30k | 1 | current |
| 50M | 8 shards | 4k | 8 | ~6M rows/shard |
| 500M | 64 shards | 4k | 64 | ~8M rows/shard |
| 1B | 128 shards | 4k | 128 | ~8M rows/shard |
| 10B | 1,024 shards | 8k | 1,024 | ~10M rows/shard; distributed build |

Per-shard K should keep `nprobe × (n_s / K_s)` in the same range as the
5M operating point (~hundreds to low thousands of docs scanned).

## 3. Optimization backlog (stage by stage)

### Ingest (biggest 5M cost)
- **Parallel WET parsing**: process pool over WET files; workers parse and
  write per-file JSONL; parent merges/dedups. The current
  `exa_home/ingest.py:62-181` is single-process.
- **`fastwarc` backend**: Rust WARC parser, optional import with `warcio`
  fallback. Add as an optional dependency; keep the `warcio` path for Tier-0.
- **Cheaper dedup**: Bloom filter / 64-bit digest set in front of SQLite;
  the 160 GB box makes the old RAM argument obsolete.
- **`orjson`** for the output line; or copy the raw WET JSON line.
- **Target**: 1M ingest 876 s → 300–450 s; 5M < 10–15 min.

### Index
- **GPU assign** (done on this branch): `assign(device="cuda")`; 5M/K=30k
  CPU ~38 TFLOP → seconds on the GPU.
- **Store raw line** (done): `ContentStore.put_line`; no parse→dump.
- **K-means `--n-init`** (done): 1 vs 3; measure centroid quality/G1.
- **mmap + lists.idx + doc_ids.bin** (v2): removes the RAM wall.
- **Shard build**: process pool / multi-machine.

### Download
- Keep-alive / connection reuse (130 TLS handshakes).
- Stream gzip verification (no second full read).
- Optional parallel ranges per file; more workers only if the 403 budget allows.

### Embed
- Single tokenization + per-batch padding (done).
- `max_seq_length` 512→256 (quality-gated; ~2×).
- Batch 2048/4096; `inference_mode`, `bf16`, TF32.
- ONNX/TensorRT (2–3×; larger project).

### Gate / sweep
- Approximate/sample recall at 1B.
- Parallel combos; fewer queries for the smoke gate.

## 4. Rules

1. **No format break without a version bump.** v1 loaders stay; v2 is additive.
2. **The MCP tool API does not change.** `--shards` is an internal detail.
3. **One knob per experiment.** Every optimization gets a before/after row in
   `EXPERIMENTS.md` and a full-config ledger row.
4. **Measure at the target scale, or label it a projection.** 1B numbers are
   not 5M numbers.
5. **Prefer mmap/zero-copy over "just add RAM".** The 160 GB box is not a
   10B machine.

## 5. First implementation slice (after the 5M run)

1. `exa_home/shards.py` + `ShardSet` tests (two tiny shards).
2. `scripts/build_shards.py` (row-range split, per-shard `build_index`).
3. Rust mmap for `codes`/`lists`/`doc_ids` behind schema v2.
4. MCP `--shards` wiring; benchmark 5M single vs 5×1M shards.
5. 50M shard run; then 500M/1B.