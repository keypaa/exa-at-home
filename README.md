# exa-at-home

Reproduce Exa-style low-latency neural search at home: a CPU-first,
binary-quantized vector DB + cross-encoder rerank + MCP server, where every
optimization lands as a measured before/after experiment.

Target box: RTX PRO 6000 Blackwell (96GB VRAM), 160GB RAM. ANN stays on CPU
on purpose — faithful to how Exa does it; the GPU serves query embedding +
rerank only.

**Toy result (81k real Common Crawl pages, CC-MAIN-2026-34): p50 43.7ms vs
the 100ms budget — PASS.** Ledger-backed (`runs.jsonl`, 2026-10-21, np8 /
tc50 / pt64 / b32): **p50 43.67ms, recall@10 = 0.2885.**

**On question-like queries** (50 natural questions, known target pages):
target recall@10 **0.740** either way, target MRR@10 **0.641 → 0.712** with
the cross-encoder, no degradations — so the reranker **stays** (the
pre-registered rule fired; the effect is favourable but not significant at
n=50). The earlier "the reranker hurts" reading came from measuring it with
document prefixes instead of questions — see [`EXPERIMENTS.md`](EXPERIMENTS.md)
M5/M5f/M5h and [`LEARNINGS.md`](LEARNINGS.md) §4, §13–14. The plan lives in
[`WORKPLAN.md`](WORKPLAN.md).

## How it works

```
Common Crawl WET slice
  → ingest (dedup, clean) → docs.jsonl
  → mxbai-embed-large-v1, MRL-256 → vecs.npy
  → sign-bit binarize (256f → 32B) + MiniBatchKMeans → index/
      centroids.f32 / codes.bin / lists.bin / doc_ids.json
      filter/ (roaring domain/date/term bitmaps)
      store/  (content shards + offsets.json byte index)
  → serve: embed → Rust IVF+LUT scan → top-50 → MiniLM-L6 rerank → top-10
  → home_search / home_contents (FastMCP, runnable: python -m exa_home.mcp_server)
```

Funnel principle: expensive full-precision work touches ~200 docs, never 5M.
5M docs need ~160MB of binary codes; there is no float store — the
reranker re-reads text from the content store (byte-offset index makes
those seeks O(1)).

**Every knob lives in one place:** [`exa_home/config.py`](exa_home/config.py)
holds the operating point (`nprobe 8, top_coarse 50, max_pair_tokens 64,
batch_size 32, budget 100ms`), the model names, the K values and the
crawl. The library, the scripts, the notebook and the MCP server all read
it — which is how the shipped tool is prevented from drifting away from
the configuration we actually measured.

## Layout

```
exa_home/        Python: ingest, store, embed, rerank, orch (Canon-lite DAG), MCP,
                 config (the one operating point) + synth (Tier-0 fakes)
crates/ann-core/ Rust (PyO3): binary quant, LUT scoring, IVF, roaring filters
scripts/         Pipeline: select_slice, download_wet, train_centroids,
                 build_index, build_filter, ground_truth, measure_recall,
                 e2e_latency, sweep (knob-grid bench), score_relevance,
                 run_pipeline (one-command slice→gate) + box_setup.sh
notebooks/       m5_gate.py — literate gate notebook (molab entry point)
eval/            relevance eval: review set + LLM-judge labels + rubric
tests/           Tier-0 suite (synth, no model) + cloud_only markers
docs/RUNBOOK.md  Exact cloud commands: slice → toy → 1M/5M → validation
EXPERIMENTS.md   M1–M6 before/after table (the learning artifact)
zenno/           Design spec + implementation plan
```

## Quickstart

**Local (Tier-0, seconds, any CPU):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1  # only on Python 3.14 + PyO3 0.22
cd crates/ann-core && maturin develop && cd ../..
python -m pytest tests/ -m "not cloud_only" -q   # 77 passed
cargo test -p ann-core                            # 13 passed
```

Anything needing a model, GPU, or >1k docs is `@pytest.mark.cloud_only`
and never runs locally. The notebook's `synth` mode also runs anywhere:

```bash
marimo edit notebooks/m5_gate.py
```

**Cloud box (toy → gate, ~30 min attended):** terminal setup is one command
(`bash scripts/box_setup.sh` — rust + deps + `ann_core`, idempotent), then
follow [`docs/RUNBOOK.md`](docs/RUNBOOK.md) §§0–6, or stepwise through the
notebook in `toy` mode. One-command equivalent:

```bash
python scripts/run_pipeline.py toy        # 4 WET → index/ + q.jsonl + gt.jsonl
python scripts/e2e_latency.py --ann ivf --index index/ --queries q.jsonl \
  --k 10 --budget-ms 100 --embedder real --reranker real \
  --top-coarse 50 --max-pair-tokens 64 --batch-size 32 \
  --max-query-tokens 32 --ledger runs.jsonl
# the gate writes NO pred file — the recall number needs the serving stack
# (RUNBOOK §5): this writes sweeps/<ts>/pred_np8_tc50_pt64_b32.jsonl
python scripts/sweep.py --index index/ --queries q.jsonl --gt gt.jsonl \
  --nprobe 8 --top-coarse 50 --max-pair-tokens 64 --batch-size 32 \
  --out sweeps/ --ledger runs.jsonl
python scripts/measure_recall.py --gt gt.jsonl \
  --pred sweeps/<ts>/pred_np8_tc50_pt64_b32.jsonl
```

## The notebook (molab)

`notebooks/m5_gate.py` is the box entry point, built for hosted rules:
markdown reasoning, library imports, Plotly waterfall/recall graphs, no
background jobs. It self-bootstraps (clone-if-missing, environment status
table, one-click rust + deps + `ann_core` build) and assumes the workspace
is wiped each session — all paths are repo-anchored, every stage rebuilds
in attended cells with progress ticks. Mirror the repo → save a copy →
`synth` to prove the environment (seconds) → `toy` for the real rebuild
(~30 min: 4 WET → 81k docs → gate + recall). Interactive knobs:
`top_coarse` / `nprobe` sliders re-run the gate live. The run ledger goes
to `runs.jsonl` — **commit it + `q/gt/pred.jsonl` and push before leaving**
(small files; `index/` and vectors stay local).

## Key decisions

| Choice | Pick | Why |
|---|---|---|
| Embedder | `mxbai-embed-large-v1`, MRL-256 | Native MRL + `truncate_dim=256` + `prompt_name="query"`, Apache-2.0, no custom modeling code (no `trust_remote_code`); Arctic-m-v2.0 unloadable (xformers assert) |
| Reranker | `ms-marco-MiniLM-L6-v2` | Same NDCG as L12 at ~2x throughput; **measured 0.59 ms/pair** (29.6ms for 50 pairs) — the vendor "6–10ms for 200 pairs" figure is stale, and the `<40ms for 200 pairs` budget test is still an open deferral (RUNBOOK §6a) |
| ANN | Rust IVF + 64×16 LUT nibble scoring | 33x over naive bit loop (`lut_scan_2k` 60µs) |
| Bridge | PyO3 via maturin (in-process FFI) | Zero per-query IPC overhead |
| Corpus | WET files, `CC-MAIN-2026-34` | Pre-extracted text; density-ranked via the columnar index (explicit per-file list — CloudFront 404s directory globs). **The toy corpus was built from the UNRANKED slice** (no `content_languages=eng` gate) → spam-heavy and multilingual; a density-ranked 1M slice is English-only, so toy numbers do not transfer |
| Store | Byte-offset index at build time | Killed a 1400ms cold-read wall misattributed to the GPU |

## Testing tiers

- **Tier-0** (local CI): 100–500 synth docs, hash embeddings, numpy/Rust fakes. Seconds.
- **Tier-2** (`cloud_only`): 10k toy → 1–5M builds. Real GPU/model/latency.
  These 6 tests are **not run by any CI job** (they need the box) — see
  WORKPLAN.md.

Recall gate: every index change reports recall@10 vs brute-force ground
truth (`ground_truth.py --queries`, aligned to the gate queries — the
random-vector path scores 0.0000 by construction). `measure_recall.py
--min-recall` exits 1 below a threshold, but **the threshold is not set
yet and no CI step runs it** — the gate is currently manual/cloud-only.

## Status / roadmap

- [x] M1–M4 (Tier-0 synth + Rust benches) · M5/M5c/M5d/M5e/M5f/M5g on the 81k toy
- [x] Store offset-index at build time, observability split, run ledger
- [x] Columnar slice fix, RUNBOOK↔pipeline alignment, literate notebook
- [x] Audit: 3 read-only audits → 3 fix waves (Rust FFI, serving path, pipeline) + a docs truth pass
- [x] Relevance eval (10 queries, 200 LLM-judge labels) — the reranker is *not* validated
- [ ] **Decide the reranker's fate** (WORKPLAN D1) — needs a bigger, human-judged eval
- [ ] 1M density-ranked build (top-130 WET, K=10k) + gate, after the scale blockers
- [ ] Waterfall tolerance 10%+2ms; optional answer/similar DAG nodes

Working plan and open decisions: [`WORKPLAN.md`](WORKPLAN.md).
Spec: `zenno/specs/2026-09-13-exa-at-home-design.md` · historical plan:
`zenno/plans/2026-09-13-exa-at-home.md`
