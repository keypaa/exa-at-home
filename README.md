# exa-at-home

Reproduce Exa-style low-latency neural search at home: a CPU-first,
binary-quantized vector DB + cross-encoder rerank + MCP server, where every
optimization lands as a measured before/after experiment.

Target box: RTX PRO 6000 Blackwell (96GB VRAM), 160GB RAM. ANN stays on CPU
on purpose — faithful to how Exa does it; the GPU serves query embedding +
rerank only.

**Toy result (81k real Common Crawl pages, CC-MAIN-2026-34): p50 46.5ms vs
the 100ms budget — PASS, with recall@10 = 0.2895 and a 10.7x rerank lift.**
Independently reproduced from a wiped box through the notebook alone
(2026-10-07): **p50 47.7ms PASS, recall@10 = 0.2885.**
Details in [`EXPERIMENTS.md`](EXPERIMENTS.md).

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
  → home_search / home_contents (FastMCP)
```

Funnel principle: expensive full-precision work touches ~200 docs, never 5M.
5M docs fit in ~160MB of binary codes + ~5GB float store.

## Layout

```
exa_home/        Python: ingest, store, embed, rerank, orch (Canon-lite DAG), MCP
crates/ann-core/ Rust (PyO3): binary quant, LUT scoring, IVF, roaring filters
scripts/         Pipeline: select_slice, download_wet, train_centroids,
                 build_index, ground_truth, measure_recall, e2e_latency,
                 run_pipeline (one-command slice→gate)
notebooks/       m5_gate.py — literate gate notebook (molab entry point)
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

**Cloud box (toy → gate, ~30 min attended):** follow
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) §§0–6, or stepwise through the
notebook in `toy` mode. One-command equivalent:

```bash
python scripts/run_pipeline.py toy        # 4 WET → index/ + q.jsonl + gt.jsonl
python scripts/e2e_latency.py --ann ivf --index index/ --queries q.jsonl \
  --k 10 --budget-ms 100 --embedder real --reranker real \
  --top-coarse 50 --max-pair-tokens 64 --batch-size 32 \
  --max-query-tokens 32 --ledger runs.jsonl
python scripts/measure_recall.py --gt gt.jsonl --pred pred.jsonl
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
| Embedder | `mxbai-embed-large-v1`, MRL-256 | Only open candidate with published 256-dim quality; Arctic-m unloadable (xformers assert) |
| Reranker | `ms-marco-MiniLM-L6-v2` | Same NDCG as L12 at ~2x throughput; 200 pairs ~6–10ms |
| ANN | Rust IVF + 64×16 LUT nibble scoring | 33x over naive bit loop (`lut_scan_2k` 60µs) |
| Bridge | PyO3 via maturin (in-process FFI) | Zero per-query IPC overhead |
| Corpus | WET files, `CC-MAIN-2026-34` | Pre-extracted text; density-ranked via the columnar index (explicit file list — CloudFront 404s globs) |
| Store | Byte-offset index at build time | Killed a 1400ms cold-read wall misattributed to the GPU |

## Testing tiers

- **Tier-0** (local CI): 100–500 synth docs, hash embeddings, numpy/Rust fakes. Seconds.
- **Tier-1** (local optional): ~1k docs, real parsing, stub embeddings. <1 min.
- **Tier-2** (`cloud_only`): 10k toy → 1–5M builds. Real GPU/model/latency.

Recall gate: every index change reports recall@10 vs brute-force ground
truth (`ground_truth.py --queries`, aligned to the gate queries — the
random-vector path scores 0.0000 by construction). Below-threshold recall
fails CI no matter how fast.

## Status / roadmap

- [x] M1–M5 + M6-partial on the 81k toy (46ms PASS, 0.29 recall)
- [x] Store offset-index at build time, observability split, run ledger
- [x] Columnar slice fix, RUNBOOK↔pipeline alignment, literate notebook
- [x] Fresh-box reproduction via notebook alone (47.7ms PASS, 0.2885 recall)
- [ ] 1M density-ranked build (top-130 WET, K=10k) + gate
- [ ] Offset-index load path hardening at 1M scale, waterfall tolerance 10%+2ms
- [ ] (Optional) answer/similar DAG nodes, Matryoshka fine-tune stretch

Spec: `zenno/specs/2026-09-13-exa-at-home-design.md` · Plan:
`zenno/plans/2026-09-13-exa-at-home.md`
