# Item 4 — 1M session plan (design)

The 1M build is too expensive, too easy to improvise, and too hard to re-run to
justify ad-hocery. This is the **single paste I run on the box**, in order,
with pre-registered acceptance criteria — the way E3 was run.

## Preconditions
- `git pull` tip matches the commit I paste from (record the SHA at the top).
- `bash scripts/box_setup.sh` done; `ann_core` imports; `torch.cuda.is_available()`.
- `runs.jsonl` double-checked backed up (the `runs.jsonl` deletion pitfall).
- `eval/e3_result_toy_20261021.json` present (G2 baseline).
- No stale `index/` from the toy build (reuse gate would block it, but we force fresh anyway).

## Steps (paste exactly)
```bash
git reset --hard <SHA>
bash scripts/box_setup.sh
python scripts/run_pipeline.py full --wet-files 130 --k 10000 --verbose --force
python scripts/e2e_latency.py --ann ivf --index index_130wet/ \
  --queries q_130wet.jsonl --k 10 --budget-ms 100 --embedder real \
  --reranker real --top-coarse 50 --max-pair-tokens 64 --batch-size 32 \
  --max-query-tokens 32 --ledger runs.jsonl
python scripts/gate.py --index index_130wet/ --queries q_130wet.jsonl \
  --gt gt_130wet.jsonl \
  --pred sweeps/<ts>/pred_np8_tc50_pt64_b32.jsonl \
  --e3-result eval/e3_result.json --budget-ms 100 --json --ledger runs.jsonl
python scripts/sweep.py --index index_130wet/ --queries q_130wet.jsonl \
  --gt gt_130wet.jsonl --nprobe 8 16 --top-coarse 50 100 \
  --max-pair-tokens 64 --batch-size 32 --ledger runs.jsonl
python scripts/run_e3.py     # writes eval/e3_result_1m.json (use manual naming)
```

## What each step proves
| step | produces | feeds |
|---|---|---|
| full | `index_130wet/` + `q_130wet.jsonl` + `gt_130wet.jsonl` | G1, G2, G3 baselines |
| e2e_latency | waterfall + p50/p99 (G3 hand-run) | sanity |
| `gate.py` (×1) | one fail-fast check + ledger row | G3 |
| sweep | nprobe×top_coarse curves + recall (G1) | nprobe choice at scale |
| run_e3 | `eval/e3_result_1m.json` with target recall@ 10 and MRR | G2 |

## Pre-registered acceptance criteria
Before pasting, these four must hold or the session is a *measurement*, not a publish:
1. **G3** on the 1M index: `p50 ≤ 100 ms` → gate succeeds. If not, the latency number is the result; we move the knob, not the budget.
2. **G1** (exact-id recall@10, reranked + coarse) drops no more than 0.10 versus the **same corpus's own** measured baseline (coarse ~0.40, reranked ~0.28).
3. **G2** (target-doc recall on the same 50 E3 questions against the 1M haystack) stays ≥ 0.640 (−0.10 from the committed toy 0.740; or vs the box-measured 0.68 if that is the honest baseline, flagged).
4. **G4** do-not-regress only: `precision@10` reranked ≥ 0.28 coherent within the n=10 noise; advisory.
5. **Interrupt resilience**: kill point. The ingest and embed runs are not allowed to restart from zero (item 1b + item 2 are both verified on this box) — if killed and resumed, the resumed wall-clock must not be worse than (1 chunk) + a few minutes.
6. **A full run survives end-to-end with no manual intervention** (autonomy, 0 edits mid-flight; every manual touch outside a pre-registered knob must be logged as an intervention in the next WORKPLAN PR).

## Abort conditions (when to stop, not improvise)
- G3 > 100 ms on the first sweep: stop; adjust K or nprobe **before** finishing the build.
- A crash in resumable embed leaves `*.state.json` that disagrees with the byte
  size of `docs.jsonl`: recompute the sha, but **do not** silently restart if it
  matches.
- The 1M box runs out of disk (5M eats 5 GB → 1M ≈ 1 GB + 8 GB downloads).
  Check `df` after every step; if over 450 GB, abort before 5M planning.
- `golden.sh` (does not exist yet — added here so I do not call something that
  does): there is no "golden" script. If one is needed, add it in this same session.

## Cost model (one page)
- 130 WET × 1.3MB ≈ 170 MB downloaded.
- 81k docs in ~4 WET → 1.04M at ~26×; fits the 503 GB disk and 160 GB RAM
  with margin.
- Embed wall-clock ~19 min at 901 docs/s; index ~2 min; sweep ~4 min; gate ~2 min.
- Total attended session: ~1 hour, within the "attended" rule of LEARNINGS §11.

## Artifacts to download before leaving
- `runs.jsonl` (the ledger — back it up)
- `index_130wet/manifest.json` (size, K=10000, sha)
- `eval/e3_result_1m.json`
- the `sweeps/<ts>/` dir for the ITM session
- (mcp tier-9 target is not in this plan; that is item 9)

## What this plan is NOT (scope defense)
- No `lite`/`auto` tier (D5, item 8).
- No content cleaning, language filtering, or re-ranking changes. The 1M
  comparison must isolate *scale* only — same code, same corpus rule as the toy.
- No 5M target. That happens only if item 4 (this item) completes under the
  protocol (gate passes) and item 6's sweep confirms the `nprobe × (N/K)`
  scaling law at 1M.
PY