# Exa-at-Home: Web-Scale Vector Search + Low-Latency MCP — Design Spec

- Date: 2026-09-13
- Status: draft, pending user review
- Path: `zenno/specs/2026-09-13-exa-at-home-design.md` (per project CLAUDE.md override)
- Goal: reproduce Exa-style low-latency neural search at home on a single cloud box, to learn how real systems get their speed.

## 1. Context and sources

Inspired by Exa's MCP and two engineering posts:

- "Building web-scale vector DB": custom vector DB, Matryoshka embeddings truncated to 256 dims (~20x memory saving), binary quantization 16-bit float to 1-bit (~16x further), hybrid binary-doc / float-query dot-product scoring with length-4 subvector lookup tables (LUTs) in CPU registers, C++ hot path, K-means clustering into 100,000 clusters with nearby-cluster search (~1000x throughput), inverted indexes for metadata filtering, full-precision rerank to recover recall. Claimed: billions of vectors in under 100ms, >500 QPS, memory footprint smaller than a gaming PC, CPU-optimized, no GPU for ANN.
- "Composing a search engine": "Canon" DAG orchestrator — query preprocessing, multi-index retrieval (web, inverted, news), fusion (e.g. RRF), content fetch, rerank, snippet extraction, safety filtering; parallel node execution by DAG dependencies, memoization for diamond deps, lazy/pull evaluation with cancellation propagation, durable execution with per-node retry, built-in per-node observability (timing, I/O, failures).

## 2. Goals and success criteria

- G1: `home_search` p50 < 100ms end to end on the cloud box (RTX PRO 6000 Blackwell 96GB VRAM, 160GB RAM), corpus 1–5M pages from a Common Crawl slice.
- G2: Every Exa optimization implemented as a measurable milestone (binary quant, IVF routing, LUT scoring, rerank recovery), each with before/after latency + recall numbers (latency waterfall).
- G3: MCP surface `home_search` + `home_contents` (namespaced to avoid confusion with real Exa tools), with a `profile: true` flag returning per-stage timings.
- G4: Local laptop stays usable — anything heavy is tagged cloud-only; local CI is seconds on any CPU.
- G5: Architecture docks future tools (answer-with-citations, similar-pages) as new DAG nodes without rework.

## 3. Non-goals (for this spec)

- Training our own embedding model. We use frozen `Snowflake/snowflake-arctic-embed-m-v2.0` (native MRL-256, ~99% retention, Apache-2.0; ST query prompt — verify doc-side prefix on-box). Truncation to 256 dims is a config choice, not a training run. Optional future stretch: fine-tune a Matryoshka head on our corpus to feel the truncation/recall trade-off.
- Web-scale billions of docs, live crawling, freshness pipelines, safety filtering, auth/billing/QPS hardening. Single-box, static-slice, learning build.
- GPU-first ANN (cuVS/CAGRA). Deliberately CPU-first for ANN to stay faithful to Exa; GPU is reserved for query embedding + rerank.

## 4. Architecture

Five online stages with strict budgets summing under 100ms p50:

```
query → [embed 5–10ms, GPU] → [ANN retrieve 10–20ms, Rust CPU] → [filter/fuse, Rust/Python] → [rerank 20–40ms, GPU] → [extract/snippets, Python]
```

- Ingest (offline): Common Crawl WET slice → clean text → Arctic-m embed (native 256 dims) → binary quant → IVF assignment → Parquet shards + inverted filter index + float store (rerank) + content store.
- Serve (online): Python MCP server + Canon-lite DAG orchestrator fans out to persistent GPU embedder (no cold loads), Rust ANN core (FFI/Unix socket), cross-encoder reranker (top ~200 → top 10), content store for `contents`.
- Instrumentation: every stage emits timing spans from day one. The latency waterfall is a first-class output.

Latency budgets (p50 targets): embed 5–10ms, ANN 10–20ms, filter/fuse <5ms, rerank 20–40ms, snippet/format <10ms, orchestrator overhead <5ms. Total < 100ms.

Capacity sanity: 5M docs × 256 bits ≈ 160MB binary matrix; full-precision float store for rerank ≈ 5GB; centroids + inverted index small. Fits trivially in 160GB RAM with room for LRU caches.

## 5. Components

| # | Component | Lang / where | Job | Interface |
|---|-----------|--------------|-----|-----------|
| 1 | Ingester | Python, offline | WET files → text extraction → clean JSONL (WARC+Trafilatura is a documented upgrade) | WET paths in → `docs.jsonl` (id, url, title, text, date) out |
| 2 | Embedder | Python + GPU, offline + online | Frozen Arctic-m-v2.0, native MRL-256, ST query prompt (verify doc prefix on-box); bulk offline, persistent-model single-query online | texts in → float32[256] out |
| 3 | ANN core | Rust, online (built offline) | Binary-quantized IVF: 1-bit/dim, ~10–30k K-means centroids for 5M docs, length-4 subvector LUT dot-product scoring, top-k over routed clusters | query vec + nprobe in → candidate ids + coarse scores out |
| 4 | Filter index | Rust, alongside ANN | Inverted index (domain/date/keyword), applied during cluster scan (Exa's graph-vs-IVF filtering argument) | filter expr in → allowed-doc bitset out |
| 5 | Reranker | Python + GPU, online | Cross-encoder over top ~200 → top 10, from stored full-precision vectors/text; recovers quant recall loss | candidate ids + query in → ranked top-10 out |
| 6 | Content store | Disk + RAM LRU | Full extracted text by doc id, serves `contents` path | ids in → texts out |
| 7 | DAG orchestrator | Python (Canon-lite) | Nodes embed → retrieve → filter → rerank → snippet; parallel fan-out, memoization, cancellation propagation, per-node spans; future tools dock as nodes | query/toolspec in → results + spans out |
| 8 | MCP server | Python (FastMCP) | `home_search`, `home_contents`, `profile` flag | MCP tools in/out (see §7) |

## 6. Data flow

Offline (ingest once, versioned `index/` dir, rebuildable/diffable):

```
WET slice → Ingester (text JSONL) → Embedder bulk (float256) → binarize (1-bit)
  → K-means assign → write: binary shards + float store + centroids + inverted index + content store
```

Online `home_search`:

```
query → orchestrate → embed (~5ms) → ANN scan routed clusters w/ LUT (~15ms, filter bitset in-scan)
  → top-200 coarse → rerank one GPU batch (~30ms) → top-10 + snippets + spans
```

Online `home_contents`:

```
ids → content store (RAM LRU → SSD) → texts. No GPU/ANN. Target p50 < 20ms.
```

Funnel principle (from Exa): expensive full-precision work touches only ~200 docs, never 5M. IVF + binary coarse pass exist to make that funnel cheap.

## 7. MCP surface

Namespaced `home_*` to avoid confusion with real Exa tools in the same client.

- `home_search(query: str, filters?: {domains?, date_range?, keywords?}, top_k?: int = 10, profile?: bool = false)` → `{results: [{id, url, title, snippet, score}], profile?: {embed_ms, ann_ms, filter_ms, rerank_ms, snippet_ms, total_ms}, degraded?: bool}`.
- `home_contents(ids: string[], max_chars_per_doc?: int)` → `{items: [{id, url, title, text}],}` paginated, never silently truncated.
- Errors: standard MCP errors for invalid args; rerank timeout falls back to coarse ANN order with `degraded: true` rather than failing.

## 8. Error handling

- ANN core: checksum index shards at startup; corrupt → fail fast, never serve silent partial recall. Clamp `nprobe`; empty filter results → `[]`.
- GPU services: model load failure → refuse startup (no cold-load-at-query-time path). Rerank timeout (>100ms) → coarse-order fallback + `degraded: true`.
- Ingest: per-doc failures logged + skipped with counters; per-shard checkpoints for resume.
- Orchestrator: node failure cancels downstream (cancellation propagation), keeps upstream cache; per-node timeouts; errors include the failing span so the waterfall shows where it broke.
- MCP: empty query / unknown ids → MCP errors; oversized contents → pagination.

## 9. Testing

- Recall harness (correctness anchor): brute-force exact top-10 for ~1k sample queries as ground truth; every index change reports `recall@10` with p50/p99; recall below threshold fails CI regardless of speed.
- Latency benches: Rust core benches (single-query, concurrency 1/8/64) + end-to-end p50/p99 per commit; `profile` spans must sum to wall time within tolerance.
- Optimization experiments: each trick lands as a milestone with a before/after record:

| Milestone | Change | p50 before → after | recall@10 before → after | Notes |
|-----------|--------|--------------------|--------------------------|-------|
| M1 baseline | float brute force, serial | | | correctness ref |
| M2 | binary quantization | | | |
| M3 | IVF routing (nprobe sweep) | | | record nprobe/recall curve |
| M4 | LUT dot-product scoring | | | |
| M5 | rerank top-200 → top-10 | | | recall recovery |
| M6 | filter-in-scan + orchestration parallel | | | |

- MCP contract tests: arg validation, degraded flag, pagination — mocked backends, fast.
- Tiered scale (laptop-safe):
  - Tier 0 (local CI default): 100–500 docs, synthetic random vectors, no model, no WARC. Seconds on any CPU.
  - Tier 1 (local optional): ~1k docs, real Trafilatura, stub/hashed embeddings. <1 min, no model download/GPU.
  - Tier 2 (cloud-only): 10k toy shard with real embeddings → full 1–5M build. Real GPU/model/latency. Tagged so it never runs locally.

## 10. Milestones (build order)

1. Ingester + content store on a tiny WET sample (Tier 0/1).
2. Embedder wiring (Arctic-m-v2.0, native MRL-256) — cloud.
3. ANN core M1→M4 with recall harness + benches.
4. Filter index + in-scan filtering.
5. Reranker (M5) + degraded fallback.
6. Orchestrator + MCP (`home_search`, `home_contents`, `profile`).
7. Full 1–5M ingest + p50 < 100ms validation + waterfall write-up.
8. (Optional) answer/similar nodes; Matryoshka fine-tune stretch.

## 11. Pre-implementation decisions (resolved 2026-09-13 via bounded research)

- ~~Exact embedding model pick~~ → `Snowflake/snowflake-arctic-embed-m-v2.0` (native MRL-256, ~99% retention, Apache-2.0). RESOLVED.
- ~~Rust↔Python bridge choice~~ → PyO3 via maturin (in-process FFI, zero IPC overhead). RESOLVED.
- ~~Cross-encoder reranker model pick within rerank budget~~ → `cross-encoder/ms-marco-MiniLM-L6-v2` (200 pairs ~6–10ms at batch 128 fp16, inside 40ms slice; Apache-2.0). RESOLVED.
- ~~Common Crawl slice selection~~ → `CC-MAIN-2026-34`, WET files (~130 for 1M / ~620 for 5M English pages), dedup by `content_digest` then URL. Locked pending on-box toy validation.
