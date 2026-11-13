# exa-at-home

Reproduce Exa-style low-latency neural search at home: a CPU-first,
binary-quantized vector DB + cross-encoder rerank + MCP server, where every
optimization lands as a measured before/after experiment.

Target box: RTX PRO 6000 Blackwell (96GB VRAM), 160GB RAM. ANN stays on CPU
on purpose — faithful to how Exa does it; the GPU serves query embedding +
rerank only.

**1 M box measurement (2 576 778 indexed docs, 130 WET, K=10 000, cc-main 2026-34):
total p50 75.562 ms against the 100 ms budget — PASS; p99 131.151 ms.** These are
**box measurements, not repo evidence**: their ledger rows are **not committed** to
`runs.jsonl` (which currently holds 0 one-million rows), so they are **not
reproducible from this repo as it stands**. Re-run the 1 M build and commit the
ledger rows before citing these as reproduced results.
G1 exact-id recall@10 **0.359** stayed inside the pre-registered −0.10 drop budget
(81 k reranked baseline 0.2885 → limit 0.1885). There is **no 1 M relevance (G4) or
answer-coverage (G2) number** — the 50-question set has never been re-anchored into
the 1 M haystack (see [`eval/E3_PROTOCOL.md`](eval/E3_PROTOCOL.md) §D1). Toy numbers
below are on the toy corpus and must not be read as 1 M evidence.

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

Serving notes: loopback by default; `--transport http` adds a bearer token
(`--token` / `EXA_MCP_TOKEN`) and a per-minute rate limit
(`--rate-limit`), and `--config` may replace the serving operating point.
Full tool schema and flags: [`docs/MCP_API.md`](docs/MCP_API.md); the
threat model and trust boundaries are in
[`SECURITY.md`](SECURITY.md). Two optional config layers exist:
`config.py` is the single source of truth for the operating point, and
`config_loader.py` (Lane B) adapts presets to a hardware profile when you
pass `--config`.

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

**There is a second, optional config layer.**
[`exa_home/config_loader.py`](exa_home/config_loader.py) (Lane B) reads a
YAML/JSON hardware profile (`tiny` / `rtx-3090-classic` /
`rtx-pro-6000-classic`) and maps it onto preset batch/chunk/`top_coarse`
values. It is an *adaptation* layer, not a second source of truth:
`config.py:OPERATING_POINT` remains what the serving path, `gate.py` and
`e2e_latency.py` read by default, and the MCP server only swaps it for the
loaded profile when you pass `--config`. Since the 2026-11-12 audit the
loader is strict — unknown or typo'd keys (and a `hardware` block with bad
types) are rejected with a `ValueError` instead of silently deferring to
the defaults, and an explicit `operating_point` may not contradict
`serving`. That strictness is what makes a config file trustworthy enough
to hand to the server.

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
python -m pytest tests/ -m "not cloud_only" -q   # 174 passed, 7 deselected
cargo test -p ann-core                            # 28 passed
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
| Reranker | `ms-marco-MiniLM-L6-v2` | Same NDCG as L12 at ~2x throughput; **measured 0.59 ms/pair** (29.6ms for 50 pairs) — the vendor "6–10ms for 200 pairs" figure is stale, and the `<40ms for 200 pairs` budget test is still an open deferral (RUNBOOK §6, item (a)) |
| ANN | Rust IVF + 64×16 LUT nibble scoring | 33x over naive bit loop (`lut_scan_2k` 60µs) |
| Bridge | PyO3 via maturin (in-process FFI) | Zero per-query IPC overhead |
| Corpus | WET files, `CC-MAIN-2026-34` | Pre-extracted text; density-ranked via the columnar index (explicit per-file list — CloudFront 404s directory globs). **The toy corpus was built from the UNRANKED slice** (no `content_languages=eng` gate) → spam-heavy and multilingual; a density-ranked 1M slice is English-only, so toy numbers do not transfer |
| Store | Byte-offset index at build time | Killed a 1400ms cold-read wall misattributed to the GPU |

## Testing tiers

- **Tier-0** (local CI): 100–500 synth docs, hash embeddings, numpy/Rust fakes. Seconds.
- **Tier-2** (`cloud_only`): 10k toy → 1–5M builds. Real GPU/model/latency.
  These 7 tests need the box (GPU + model weights + Common Crawl access). CI
  has a weekly / `workflow_dispatch` `cloud_only` job, but it runs on a
  GitHub-hosted runner without GPU or box access and is `|| true` — so
  cloud_only **results require on-box validation**; only the Tier-0 gates
  are enforced on every push.

Recall gate: every index change reports recall@10 vs brute-force ground
truth (`ground_truth.py --queries`, aligned to the gate queries — the
random-vector path scores 0.0000 by construction). `measure_recall.py
--min-recall` exits 1 below a threshold (default `None`; `scripts/gate.py`
implements G1–G4 with hardcoded thresholds and CI runs `gate.py --synth`)
— the full real-index gate is manual/cloud-only.

## Status / roadmap

- [x] M1–M4 (Tier-0 synth + Rust benches) · M5/M5c/M5d/M5e/M5f/M5g on the 81k toy
- [x] Store offset-index at build time, observability split, run ledger
- [x] Columnar slice fix, RUNBOOK↔pipeline alignment, literate notebook
- [x] Audit: 3 read-only audits → 3 fix waves (Rust FFI, serving path, pipeline) + a docs truth pass
- [x] Relevance eval (10 queries, 200 LLM-judge labels) — the reranker is relevance-neutral at
  n=10 (0.350 coarse vs 0.380 reranked precision@10, 3 better / 1 worse)
- [x] **Decide the reranker's fate** (WORKPLAN D1, 2026-10-21) — **KEEP**. 50 question-based eval:
  target recall@10 0.740 = 0.740; MRR 0.641 → 0.712 (+0.071, p=0.18, n.s.).
  Question-vs-prefix behaviour (not a coefficient of the default) determines it.
- [x] **1 M scale exercise complete on the box** (2026-10-25) — unranked 130 WET, 2.58 M docs,
  K=10 k, G1 reranked **0.359** (−0.0705 vs 81 k 0.2885), G3 **p50 75.562 / p99 131.151 ms**,
  `top_coarse` 100 is over-budget. G2 at 1 M is **unmeasured (0.00 / 0.00 is "not re-anchored",
  not a quality claim)**: the 50 committed questions are not each re-anchored into the 1 M
  haystack. **The ledger rows for this run are not committed — see "Measurement status"**.
- [ ] Waterfall tolerance 10%+2ms; optional answer/similar DAG nodes

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Running the MCP server

```bash
# stdio (recommended for local agents)
python -m exa_home.mcp_server --index index/

# HTTP — loopback by default; --token/EXA_MCP_TOKEN adds bearer auth,
# --rate-limit caps calls/minute
python -m exa_home.mcp_server --index index/ --transport http --port 8080
```

Tool schema, bounds and flags: [`docs/MCP_API.md`](docs/MCP_API.md). Serve an
index that has passed `scripts/gate.py` — the server itself only verifies
`index/manifest.json` **integrity** (per-file sha256, fail-fast), it does not
re-run the gate. Model weights are cached under `$HF_HOME` /
`~/.cache/huggingface`; a cold start downloads ~1.3 GiB for mxbai-embed and
~90 MB for the reranker.

## Measurement status

- **1 M results are box measurements, not repo evidence.** The 1 M build ran on the box
  2026-10-25 (G1 `0.359`, G3 p50 `75.562` ms / p99 `131.151` ms — see `EXPERIMENTS.md` M9),
  but **its ledger rows are not committed to `runs.jsonl`**, which currently holds 0
  one-million-row entries. Until those rows (or a fresh 1 M run) are committed, the numbers
  are **not reproducible from this repo** — re-run and commit before citing them as evidence.
  G2 (target-doc coverage) remains **non-comparable** at 1 M until the fixed 50-question
  set is re-anchored into the 1 M haystack (`eval/E3_PROTOCOL.md` §D1); no 5 M numbers
  exist yet.

- **5 M runs are deferred** until every PR in the 1 M hardening queue is clean and one
  **soft gate threshold** document is merged with the measured nprobe curve.

Working plan and open decisions: [`WORKPLAN.md`](WORKPLAN.md).
The map — what we did, what we're doing, what's next, where it's going: [`ROADMAP.md`](ROADMAP.md).
Spec: `zenno/specs/2026-09-13-exa-at-home-design.md` · historical plan:
`zenno/plans/2026-09-13-exa-at-home.md`
