# HANDOFF — exa-at-home

**For a fresh Claude Code session.** Read this file top to bottom, then
read `ROADMAP.md` and `WORKPLAN.md`. Together they are the complete state
of the project, what it is, what works, what is open, and what to do next.

Last updated: 2026-10-24. Latest commit: `ff2b551`.

---

## 1. What this project is

A **learning build** of Exa-style low-latency neural search, reproduced at
home on a single cloud box (RTX Pro 6000, 160GB RAM, 503GB disk), to learn
how real systems get their speed. It is a CPU-first binary-quantized vector
DB + cross-encoder rerank + MCP server, where every optimization lands as a
**measured before/after experiment**. The artifacts are `EXPERIMENTS.md`
(numbers), `LEARNINGS.md` (principles), and `WORKPLAN.md` (state + open
decisions).

- GitHub: https://github.com/keypaa/exa-at-home
- Local: `/home/keypaa/Projects/exa_at_home_exp`
- License: none specified (private learning repo)

**Ultimate goal (4 conditions, all must hold):**
1. It works — full funnel over 1M+ real Common Crawl pages, served over MCP,
   p50 inside 100ms.
2. Every decision is measured — no design choice exists because it felt right.
3. It is professional — modular, linted, CI-gated, tested, documented.
4. The knowledge is written down — LEARNINGS.md holds principles that outlive
   the code.

**Scale ladder:** 1M (milestone) → 5M (headline) → 10M (stretch). 20M is out
on RAM/disk, not GPU. Web-scale stays out of scope.

---

## 2. Architecture

```
Common Crawl WET slice
  → ingest (digest + URL dedup, clean) → docs.jsonl
  → mxbai-embed-large-v1, MRL-256 → vecs.npy
  → sign-bit binarize (256f → 32B, LSB-first via np.packbits bitorder="little")
     + MiniBatchKMeans → index/
       centroids.f32 (K×256) / codes.bin (n×32) / lists.bin (K posting lists)
       / doc_ids.json / manifest.json (sha256 per file, store/ excluded)
       filter/  (roaring bitmaps: domains/dates/terms.{bin,json})
       store/   (content shards + offsets.json byte index)
  → serve: embed → Rust IVF+LUT scan → top-50 → MiniLM-L6 rerank → top-10
  → home_search / home_contents (FastMCP)
```

**Funnel principle:** expensive full-precision work touches ~200 docs, never 5M.
**5M docs:** ~160MB binary codes + content store (no float store — the
reranker re-reads text from the store; byte-offset index makes seeks O(1)).

**Funnel knobs (all in `exa_home/config.py`):**
- `nprobe` (default 8): clusters probed per query. Knee at 8 on real data.
- `top_coarse` (default 50): candidates handed to the reranker.
- `max_pair_tokens` (default 64): passage tokens per rerank pair.
- `batch_size` (default 32): reranker batch.
- `budget_ms` (default 100): total p50 budget.

**Key latency model:** ANN scans `nprobe × (N/K)` docs per query. At 5M with
K=10k, that's 4,000 docs/query — the scan eats the budget. K must grow
with N to keep it flat. This is the scaling law.

**Exa's documented stack** (from their blog, our spec §1): binary quant →
IVF + LUT → inverted indexes for metadata filtering → full-precision rerank
to recover recall. Our repo reproduces all of these, and Exa 3.0 (Oct 19 2026)
added a larger embedding model, Dynamic Highlights (95% fewer tokens), billions
more documents, and a Lite/Auto cost tier — which maps onto our coarse/rerank
funnel.

---

## 3. Repo layout

```
exa_home/        Python: ingest, store, embed, rerank, orch (Canon-lite DAG), MCP,
                 config (the one operating point) + synth (Tier-0 fakes)
crates/ann-core/ Rust (PyO3): binary quant, LUT scoring, IVF, roaring filters,
                 index loader, manifest verify
scripts/         Pipeline: select_slice, download_wet, train_centroids,
                 build_index, build_filter, ground_truth, measure_recall,
                 e2e_latency, sweep (knob-grid bench), score_relevance,
                 run_pipeline (one-command slice→gate) + box_setup.sh + jsonl.py
notebooks/       m5_gate.py — literate gate notebook (molab entry point)
tests/           Tier-0 suite (synth, no model) + cloud_only markers
eval/            Relevance eval: review set + LLM-judge labels + rubric + E3
docs/RUNBOOK.md  Exact cloud commands: slice → toy → 1M/5M → validation
EXPERIMENTS.md   M1–M6 + M5b–M5h before/after table (the learning artifact)
LEARNINGS.md     14 principles (the transferable knowledge)
ROADMAP.md       The map: done / now / next / ultimate goal
WORKPLAN.md      Protocol + ordered TODO + open decisions + the four gates
zenno/           Design spec + implementation plan (historical, banner-marked)
```

---

## 4. What works (proven, with numbers)

| Component | State | Evidence |
|---|---|---|
| Rust ANN core | Solid | 28 Rust tests: binary quant, LUT (33×), IVF, roaring filters, manifest verify, partition checks, NaN-safe, non-contiguous numpy |
| Pipeline | Works end to end | ingest 81k docs → embed → K=1024 centroids → index → gate → recall; ~6 min on the box |
| Measurement harness | Works | `sweep.py` (4-D grid, timestamped artifacts, full-config ledger), `e2e_latency` (p50/p99 + waterfall integrity), `score_relevance` (judged precision@10) |
| Notebook (molab) | Works | self-bootstrapping, literate, Plotly, ran the whole session unattended-ish |
| CI | Green | ruff 0.17, pytest, cargo test, cargo fmt — every push |
| Ledger | 21 rows | 5 with full config + recall; 16 older notebook rows lack knobs |

### The numbers

- **Toy gate (81k docs):** p50 47.7ms vs 100ms budget — PASS. Breakdown: embed 8.7, ann 5.1, rerank 32.7 (fetch 5.2 / model 28.2), snippet 0.02.
- **Reranker decision (E3, 50 questions):** target recall@10 0.740 = 0.740 (tie), MRR 0.641 → 0.712 (+0.071, p=0.18, not significant), 0 degraded. **Kept.**
- **Exact-id recall:** coarse 0.4015 (nprobe=8), 0.4240 (nprobe=16), reranked 0.2885. The reranker's earlier "loss" was a query-type artifact — see LEARNINGS §4/§13.
- **Binary ceiling:** 0.444 (full scan vs float brute force) — matches M2's synth 0.4460. The representation tops out near 44% id-agreement.
- **Relevance eval (M5f):** coarse 0.350 vs reranked 0.380 precision@10 (3 better / 1 worse / 7 tied, n=10, LLM judge).
- **M8 embed throughput:** 442 docs/s at 81k. fp16 + batch 32→4096 gave +3.4% total. The embedder is **CPU-bound, not GPU-bound** — the tokenizer is the wall.

### What is measured but weak

- 16 of 21 ledger rows carry no knobs and no recall (provenance-poor).
- The M5c/M5d recall curves are prose-only (no committed artifacts).
- The headline "46.5ms / 0.2895" is pre-ledger; the ledger's actual best is 43.7ms / 0.2885.
- The reranker is kept on a favourable-but-not-significant MRR (p=0.18, n=50).

---

## 5. What is broken or unproven

- **MCP server**: runnable now (`python -m exa_home.mcp_server --index index/`), but never benchmarked against a real index. No latency/recall numbers for the shipped surface.
- **Corpus mismatch**: the toy came from the *unranked* slice → spam-heavy, multilingual, 73.7% mostly-ASCII. A 1M density-ranked build is English-only. Toy ≠ 1M corpus.
- **6 `cloud_only` tests never run anywhere** — no cloud CI job.
- **The recall threshold**: the README claims "below-threshold recall fails CI" — no threshold is set, no CI step runs it. The gate is currently manual/cloud-only.
- **Scale blockers closed** (filter build RAM, posting lists, ground truth, store hashing) — see the audit wave. Remaining: `stage_ingest`'s URL-dedup set (~660MB at 5M URLs, inherent to URL-second dedup).
- **The 1M build itself**: unstarted.

---

## 6. Key decisions (with reasoning)

| Decision | Choice | Why |
|---|---|---|
| Embedder | `mxbai-embed-large-v1`, MRL-256 | Native MRL + `truncate_dim=256` + `prompt_name="query"`, Apache-2.0, no custom modeling code (no `trust_remote_code`). Arctic-m-v2.0 unloadable (xformers assert). |
| Reranker | `ms-marco-MiniLM-L6-v2` | Same NDCG as L12 at ~2x throughput; measured 0.59 ms/pair. **Kept** after E3 — no harm on question-like queries, favourable direction (p=0.18, not significant). |
| ANN | Rust IVF + 64×16 LUT nibble scoring | 33x over naive bit loop (`lut_scan_2k` 60µs). |
| Bridge | PyO3 via maturin (in-process FFI) | Zero per-query IPC overhead. |
| Corpus | WET files, `CC-MAIN-2026-34` | Pre-extracted text; density-ranked via the columnar index (explicit per-file list — CloudFront 404s directory globs). Toy uses the unranked slice. |
| Store | Byte-offset index at build time | Killed a 1400ms cold-read wall misattributed to the GPU. |
| Scale ladder | 1M → 5M (headline) → 10M (stretch) | 5M fits comfortably in 160GB RAM / 503GB disk; 20M is out on RAM/disk, not GPU. |
| Embed throughput | fp16 + batch 1024 (measured: +3.4%) | The GPU is not the bottleneck — the CPU tokenizer is. Pre-tokenizing offline is the real lever. |
| Reranker decision | Pre-registered rule: win → keep, tie → drop to flag, lose → drop | E3 fired on MRR → keep. The rule prevented post-hoc rationalization. |

**The big correction:** the "10.7x rerank lift" was an artifact of a deliberately
inverted baseline (MockReranker reversed the candidate list, scoring the *worst*-10
of top-50, not the top-10). The honest coarse baseline is 0.4015; the reranker
moves it to 0.2885. The reranker's earlier "harm" was a query-type artifact:
it demotes embedding-space neighbours (which exact-id overlap rewards) on
document-prefix queries, but promotes the right page to #1 on real questions.

**The second correction:** the 0.40 exact-id recall is not "poor quality" — it's
the quantization ceiling (binary full-scan agreement with float brute force =
0.444, matching M2's synth 0.4460). The user-facing metric (target-doc recall)
is 0.74. The representation, not the serving stack, limits quality.

---

## 7. The measurement protocol

**Design → Decide → Measure.** No new work starts until the current item is
designed, decided, *measured*, and recorded. "Measured" means a number in
`runs.jsonl` (with full config) or a row in `EXPERIMENTS.md` — not a
plausible-sounding claim in chat.

Rules learned the hard way:
1. Every knob in a ledger row, or the row means nothing.
2. Measure a baseline with an instrument that isn't sabotaged.
3. When a component measures badly, check whether the measurement is in the
   component's domain before cutting it.
4. A negative result is a result — publish it.
5. "Not significant" is an answer; write down the sample size that would settle it.
6. One knob per row, or the row means nothing.
7. Do the science where the science is welcome (attended, incremental, committed).

**Instruments:**
- `scripts/sweep.py` — knob grid, per-combo gate + preds + recall, timestamped
  artifacts, full-config ledger rows.
- `scripts/e2e_latency.py` — p50/p99 per stage + waterfall integrity check.
- `scripts/score_relevance.py` — judged precision@10 from a review set.
- `eval/E3_PROTOCOL.md` — the D1 experiment design + cells.
- `runs.jsonl` — the source of truth for every measurement.

**Four gates (defined, not yet implemented as a command):**

| Gate | Metric | Toy baseline | Rule at 1M |
|---|---|---|---|
| G1 — ANN fidelity | exact-id recall@10, 200 prefix queries | coarse 0.4015 / reranked 0.2885 | no more than −0.10 |
| G2 — answer coverage | target-doc recall@10, 50 questions | 0.740 | no more than −0.10 |
| G3 — latency | total p50 over 200 queries | 47.7ms | ≤ 100ms |
| G4 — relevance (noisy) | judged precision@10, n=10 | coarse 0.350 / reranked 0.380 | do-not-regress only |

---

## 8. Environment and setup

**Box (molab):** 4 CPU, 160GB RAM, 503GB disk, RTX Pro 6000 (96GB VRAM).
Python 3.13 (not 3.14 — the box uses 3.13, which avoids the PyO3 ABI gate).

**Setup (every wiped session):**
```bash
git clone https://github.com/keypaa/exa-at-home.git && cd exa-at-home
bash scripts/box_setup.sh    # rust + deps + ann_core, idempotent, PEP-668 aware
```

**Toolchain:**
- `maturin develop` from `crates/ann-core` (not the workspace root)
- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` needed on Python 3.14; not on 3.13
- CI runs on Python 3.13 + Rust stable

**Molab rules:** no SSH, no alt UIs, no background jobs, no keep-alives,
no public endpoints. Stay attended, commit + push before leaving (the box
wipes). The notebook (`notebooks/m5_gate.py`) is the sanctioned interface:
literate markdown + library imports + Plotly graphs.

---

## 9. How to run things

**Local (Tier-0, seconds, any CPU):**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
cd crates/ann-core && maturin develop && cd ../..
python -m pytest tests/ -m "not cloud_only" -q   # 108 passed
cargo test -p ann-core                            # 28 passed
```

**Cloud box (toy → gate, ~30 min attended):**
```bash
python scripts/run_pipeline.py toy        # 4 WET → index/ + q.jsonl + gt.jsonl
python scripts/e2e_latency.py --ann ivf --index index/ --queries q.jsonl \
  --k 10 --budget-ms 100 --embedder real --reranker real \
  --top-coarse 50 --max-pair-tokens 64 --batch-size 32 \
  --max-query-tokens 32 --ledger runs.jsonl
python scripts/sweep.py --index index/ --queries q.jsonl --gt gt.jsonl \
  --nprobe 8 --top-coarse 50 --max-pair-tokens 32 64 --batch-size 32 \
  --out sweeps/ --ledger runs.jsonl
python scripts/measure_recall.py --gt gt.jsonl --pred sweeps/<ts>/pred_np8_tc50_pt64_b32.jsonl
```

**The notebook:** `marimo edit notebooks/m5_gate.py` — synth mode (seconds)
then toy mode (~30 min attended). Interactive knobs: `top_coarse`/`nprobe`
sliders re-run the gate live.

---

## 10. Immediate next steps (in order)

1. **Item 1b — pre-tokenize offline.** The embedder is CPU-bound (442 docs/s,
   96GB GPU at 1.1GB). Tokenize all docs once, save ids, then the GPU does
   pure forward passes. Measure before/after. This is the lever that
   determines the scale ceiling.
2. **Item 2 — resumable chunked embed.** Per-chunk part files so an
   interrupted 1M embed resumes. (37 min of GPU is currently all-or-nothing.)
3. **Item 3 — the four gates.** Implement as one command with exit code.
4. **Item 4 — 1M session plan.** Written, like E3's. Pre-registered
   acceptance criteria.
5. **Item 5 — 1M build.** Unranked first-130 WET, same rule as the toy.
6. **Item 6 — 1M sweep.** nprobe × top_coarse × K. The scale curve.
7. **Item 7 — 5M build** (stretch, if 1M is clean).
8. **Item 8 — `lite`/`auto` serving tier.** Coarse-only vs coarse+rerank,
   one switch, one measured frontier. D1 just made this concrete.
9. **Item 9 — MCP server benchmarked against a real index.**

**Corpus decision (D2):** unranked first (same rule as toy) for the scale
comparison; density-ranked English as a separate run. Toy ASCII fraction:
73.7%.

---

## 11. Pitfalls (learned the hard way)

1. **`runs.jsonl` gets deleted by validation.** Every `rm -f runs.jsonl`
   in a validation script nukes it. Back up before destructive validation.
2. **The box has no GitHub access.** Download artifacts via the browser;
   merge locally.
3. **`maturin develop` must run from `crates/ann-core`**, not the workspace
   root (root `Cargo.toml` is workspace-only).
4. **PyO3 0.22 vs Python 3.14**: set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.
   (The box runs 3.13, so this isn't needed there.)
5. **`import scripts` works because `scripts/` is a real package** — don't
   add `sys.path.insert(0, ROOT)` hacks (they're dead code).
6. **The embedder is CPU-bound, not GPU-bound.** fp16 + big batches gave +3.4%.
   The real lever is pre-tokenizing offline.
7. **`text[:12000]` is a dead knob** — mxbai truncates to 512 tokens
   (~2000 chars). Either raise `max_seq_length` or admit the slice does
   nothing.
8. **The reranker's `--embedder arctic` flag is an alias for mxbai.** The
   ledger records "arctic" for mxbai runs. Flag, don't trust.
9. **`top_k` means two things** in the Rust binding (coarse count vs final
   count). Renamed to `n_coarse` in `IvfIndex.search`. `AnnIndex` still
   uses `top_k` (final count). Don't confuse them.
10. **The test suite's `cloud_only` marker** skips 6 tests locally. Run with
    `-m "not cloud_only"` always.
11. **`ruff` 0.17.0 is pinned** — local and CI must match. Use `.venv/bin/ruff`.
12. **`cargo fmt --check` runs in CI** on `crates/ann-core`. Run it before
    pushing Rust changes.
13. **The notebook's cells share one namespace** (marimo). A variable defined
    in one cell must be declared as a parameter in cells that use it.
14. **`mo.stop` is interactive-only** — it doesn't stop execution in headless
    export. Use `if/else` for headless-safe branching.
15. **Dropdown `dict` options resolve differently in headless vs interactive**
    — use list options.
16. **`runs.jsonl` is tracked evidence** — don't delete it during validation.
17. **The `zenno/` plan is historical** — execute RUNBOOK.md, not the plan.

---

## 12. Key artifacts (committed, reproducible)

| Artifact | What it is | Where |
|---|---|---|
| `runs.jsonl` | 21-row ledger: p50/p99 + stages + breakdown + recall + full config | repo root |
| `sweeps/20261011-151336/` | Pair-tokens sweep: grid.json + 2 pred files + sweep.log | repo |
| `sweeps/20261021-161945/` | Batch-size sweep: grid.json + 3 pred files + sweep.log | repo |
| `eval/review_toy_20261021.json` | 10 queries + 200 blinded entries (A/B shuffled) | repo |
| `eval/labels_llm_judge_20261021.json` | 200 labels + rubric (LLM judge) | repo |
| `eval/e3_result_toy_20261021.json` | E3 result: recall@10 0.740/0.740, MRR 0.641/0.712, +0.071 | repo |
| `eval/questions_toy_20261021.json` | 50 grounded questions | repo |
| `eval/E3_PROTOCOL.md` | The D1 experiment design + cells | repo |
| `EXPERIMENTS.md` | M1–M6 + M5b–M5h before/after table | repo |
| `LEARNINGS.md` | 14 principles | repo |
| `ROADMAP.md` | The map: done / now / next / ultimate goal | repo |
| `WORKPLAN.md` | Protocol + TODO + open decisions + four gates | repo |
| `HANDOFF.md` | This file | repo |
| `gt.jsonl`, `pred.jsonl`, `q.jsonl` | 200 queries + ground truth + predictions | repo |

---

## 13. Open decisions

- **D1 — reranker: DECIDED 2026-10-21: KEEP.** E3 (50 questions): recall@10
  0.740 = 0.740, MRR 0.641 → 0.712 (+0.071, p=0.18, not significant),
  0 degraded. Pre-registered rule fired on MRR. The earlier "it hurts" reading
  was a query-type artifact (document prefixes vs real questions). Follow-up:
  a bigger, human-judged eval would either confirm the +0.07 or dissolve it.
- **D2 — corpus for the 1M build: DECIDED.** Same selection rule as the toy
  (unranked first-N WET) so the scale comparison isolates scale. The
  density-ranked English slice is a separate, separately reported run.
- **D3 — 1M scale blockers: CLOSED.** Filter build 21× less RAM, posting
  lists 86× faster, ground truth 37× faster, sweep runs queries once.
  Remaining: `stage_ingest`'s URL-dedup set (~660MB at 5M URLs, inherent).
- **D4 — recall threshold: NOT SET.** The README claims "below-threshold
  recall fails CI" — no threshold, no CI step. Either wire it or delete the
  claim.
- **D5 — the serving tier (`lite`/`auto`):** not built. D1 just made it
  concrete.

---

## 14. How to run things

**Local (Tier-0, seconds, any CPU):**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
cd crates/ann-core && maturin develop && cd ../..
python -m pytest tests/ -m "not cloud_only" -q   # 108 passed
cargo test -p ann-core                            # 28 passed
```

**Cloud box (toy → gate, ~30 min attended):**
```bash
python scripts/run_pipeline.py toy
python scripts/e2e_latency.py --ann ivf --index index/ --queries q.jsonl \
  --k 10 --budget-ms 100 --embedder real --reranker real \
  --top-coarse 50 --max-pair-tokens 64 --batch-size 32 \
  --max-query-tokens 32 --ledger runs.jsonl
python scripts/sweep.py --index index/ --queries q.jsonl --gt gt.jsonl \
  --nprobe 8 --top-coarse 50 --max-pair-tokens 32 64 --batch-size 32 \
  --out sweeps/ --ledger runs.jsonl
python scripts/measure_recall.py --gt gt.jsonl \
  --pred sweeps/<ts>/pred_np8_tc50_pt64_b32.jsonl
```

**The notebook:** `marimo edit notebooks/m5_gate.py` — synth (seconds) then
toy (~30 min attended). Interactive knobs: `top_coarse`/`nprobe` sliders
re-run the gate live.

---

## 15. The learning principles (LEARNINGS.md)

1. The bottleneck is never where the spans say — until you've profiled.
2. A parked Minor on the hot path is a Major with a delay fuse.
3. Measure the unit you claim to optimize.
4. Exact-id recall is the harshest lens — report lift, not just absolutes.
5. Know your harness failure modes — we measured one pred five times.
6. Funnels are non-monotonic — and a timeout can masquerade as a curve.
7. Find knees, don't chase maxima.
8. One knob per row, or the row means nothing.
9. At 1M scale, models are fixed cost — time is the constraint, not memory.
10. Build-time work beats query-time work, always.
11. Do the science where the science is welcome.
12. A negative result is a result — publish it.
13. A metric can punish a component for doing its job — change the query,
    not the metric.
14. Pre-register the decision rule, not just the experiment.

---

## 16. Immediate next step (post-1 M)

Use **Lane C work already on branch PRs**, not code changes mid-run:
- `feat/config-loader-seam` (core contract for one YAML, awaiting merge gate)
- `perf/ingest-disk-url-dedup` (storage-safe, tested)
- `refactor/store-thread-safety` (merged but re-pushed)

The M1 policy is **publish first, scale later**: record the -0.10 drop rule,
the tc100 over-budget gate, and the G2 re-derivation set before starting 5 M.

(Any leftover “fix `gate.py` option” patching on the 1 M data must target
the 2026-10-25 corpus, not the 81 k artifact set.)

The E3 protocol (for the reranker decision, already decided) lives in
`eval/E3_PROTOCOL.md`. The next experiment that needs the box is the
pre-tokenize diagnostic — I'll write the cell when asked.
