# Item 3 — The four gates as one command (design)

## Why
WORKPLAN §"The gates" defines G1–G4, but only in prose, with **no threshold**, **no command**,
and no CI wired to any of them. README claims "below-threshold recall fails CI" — a false
claim (D4 open). This item makes the gate real: one entry point, one exit code, four rows in
the ledger.

## The four gates (frozen from WORKPLAN)

| Gate | Metric | Toy baseline | Rule at 1M |
|------|--------|--------------|------------|
| G1 — ANN fidelity | exact-id recall@10 (200 prefix queries) | coarse 0.4015 / reranked 0.2885 | drop at most **0.10** vs its own corpus baseline |
| G2 — answer coverage | target-doc recall@10 (50 E3 questions) | 0.740 | drop at most **0.10** vs 0.740 — this is the scale test |
| G3 — latency | total p50 over 200 queries | 47.7 ms | **≤100 ms** |
| G4 — relevance | judged precision@10 (LLM, n=10) | coarse 0.350 / reranked 0.380 | do-not-regress only — n=10 cannot gate |

Baseline is **corpus-local**, not toy-fixed: a 1M build records its own baseline G1(reranked)
and G3 and then compares against that. This is what makes G1 a *scale* test rather than
a rebuild test.

## Command shape
```
scripts/gate.py --index <dir>/ --queries q.jsonl --gt gt.jsonl \
    --pred-g1 <tc50 pred> --pred-g2 <e3 pred or run> \
    --queries-e3 eval/questions_toy_20261021.json \
    --budget-ms 100 --ledger runs.jsonl
```
Single exit code, single row appended. All four gates are computed in process (no shelling).

## Threshold and drop-budget contract (decisions)

- **G3** is an *absolute* bound: `total_ms_p50 <= 100`. Fails (exit 1) if exceeded. No
  drop budget — the 100ms is the product, not a relative change.
- **G1** is *relative*: `G1_reranked corpus - G1_reranked baseline <= 0.10`. The baseline row
  must be carried with the gate result (a recorded G1 baseline is a prerequisite).
- **G2** is *relative against 0.740*: `G2 >= 0.740 - 0.10`.
- **G4** is *do-not-regress within ±noise*: neither side may drop more than what n=10 can
  resolve. With ±0.1 at n=10, gate as: `precision@10 reranked >= 0.380 - 0.1` AND
  `precision@10 coarse >= 0.350 - 0.1`. Mark G4 as **advisory only** (warn, don't fail) until
  the n=50 eval is added.
- Exit code 0 = all hard gates (G1, G2, G3) pass; exit code 1 = at least one hard gate
  failed; exit code 2 = usage/input error. G4 only ever prints a warning.

## Dependencies (what the gate consumes)
- G1: `gt.jsonl` (partition-based top-k) + sweep pred file from the same queries.
  Both already produced by `run_pipeline.py toy/full`.
- G2: the committed E3 question/answer pairs against the **same 50 target ids** against the
  1M haystack. Requires stage_queries-scale machinery (already has `--queries`). Mode:
  coarse vs reranked on the same 50-candidate pool — reuses `eval/E3_PROTOCOL` cells.
- G3: `e2e_latency.py --profile` style per-stage waterfall.
- G4: the review set + judge pipeline (`score_relevance.py`); runs as a **separate**
  grid (n=10) and is reported, not gated.

## Tier-0 support (must run under `-m "not cloud_only"`)
- Use `HashEmbedder`, `MockReranker`, `NumpyFakeANN` for G1 (synth ANN baseline),
  G3 (latency bound trivially passes; sanity check only), and leave G2/G4 cloud-only with
  skip-markers and explicit skip reasons so that the command's exit code is still 0 on
  Tier-0 (Tier-0 is a smoke test, not the real gate).

## Ledger row
```
gate_run: {g1_coarse, g1_reranked, g1_drop_budget, g1_baseline_ref,
           g2_target_recall, g2_baseline, g2_drop_budget,
           g3_p50_ms, g3_p99_ms, g3_budget_ms,
           g4_coarse_p10, g4_reranked_p10, g4_advisory,
           index_dir, sha, n_docs, k, nprobe, top_coarse,
           max_pair_tokens, batch_size, n_queries, wall_ms}
```
Full config on the same row (item measurement protocol rule: every knob in a row).

## Design decisions left open for review
1. **D4 — threshold magnitude.** WORKPLAN suggests −0.10 drop budgets for G1/G2.
   But the 81k binary-quantization ceiling is 0.444 while exact-id recall sits at 0.4015;
   a 0.10 allowed slip at 1M probably assumes rank stabilization that we have not measured.
   Options: ship −0.10 as pre-registered (then `--delta` flag for future), or raise the
   allowed slip to −0.20 with a unit test that records which fires. Pre-register now; the
   number is cable.
2. **G4 → exit code.** The n=10 review set cannot reliably discriminate; keeping G4
   advisory matches WORKPLAN ("do-not-regress only — n=10 cannot gate"). When the **n=50 E3
   question eval is extended to judge responses**, G4 may promote to hard (binary judgement,
   not LLM-only).

## Implementation sketch (4 tasks)
1. `scripts/gate.py` CLI + G1/G3 compute + exit codes. Tier-0 smoke test.
2. Add G2 compute against E3 questions id-aligned with the corpus.
3. Wire G4 as advisory + reuse `score_relevance.py` outputs.
4. CI step: Tier-0 smoke + nightly/cloud full gate on the small toy slice.
   First wire in `.github/workflows/` (none exists today — the repo has no CI at all).

## What we do NOT touch in this item
- No model, no indexing, no ANN algorithm changes (measure against what shipped).
- No threshold retrofitting: once G3/p99 gate in the first row is written, the
  threshold stands unless a code review explicitly overrides it (D1's discipline).
- G2 target-doc ids are fixed by E3: five target docs that must remain graph-reachable.

## Ledger entry to append on the box (copy once real run lives)
```
gate_run: {g1_coarse, g1_reranked, g1_drop_budget, g1_baseline_ref,
           g2_target_recall, g2_baseline, g2_drop_budget,
           g3_p50_ms, g3_p99_ms, g3_budget_ms,
           g4_coarse_p10, g4_reranked_p10, g4_advisory,
           index_dir, sha, n_docs, k, nprobe, top_coarse,
           max_pair_tokens, batch_size, n_queries, wall_ms}
```

## Next (only after the box run lands)
- Record the 1M `gate_run` row.
- Use the G3 p99 to tune the reranker batch ceiling before the 5M.
- Begin item 4 (session plan, like E3) — the workplan clearly says this and every future
  measurement has to rest on the gate running green.
PY