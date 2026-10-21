# Work plan — one thing at a time

**Protocol (binding).** No new work starts until the current item is
designed, decided, *measured*, and recorded. "Measured" means a number in
`runs.jsonl` (with full config) or a row in `EXPERIMENTS.md` — not a
plausible-sounding claim in chat.

**Order of work**

| # | Item | State | Measure / acceptance |
|---|------|-------|---------------------|
| 1 | Audit (correctness / docs / architecture) | done | 3 read-only audits, findings triaged |
| 2 | Fix wave 1: Rust core (NaN, partition, manifest, naming) | done | 20 Rust tests, 116 Python tests green |
| 3 | Fix wave 2: serving path (timeouts, store, filters) | done | regression tests per bug |
| 4 | Fix wave 3: pipeline (download corruption, dedup, streaming, p50) | done | regression tests per bug |
| 5 | **Corrected measurements** (M5f relevance eval) | done | coarse 0.350 vs reranked 0.380 (n=10, noise) |
| 6 | Cleanup wave 4: dead code, duplication, config, tests | in progress | 116 tests stay green; LOC down; no behaviour change |
| 7 | Cleanup wave 5: docs truth pass | pending | every doc number re-derived from an artifact |
| 8 | **Decide the reranker's fate** | pending | design → measure: does it earn 33ms? (M5f says no, n=10) |
| 9 | 1M build | pending | needs a decision on the corpus (English-only density-ranked) and on the reranker |
| 10 | Serving tier (`lite` vs `auto`, Exa-3.0-style) | pending | the frontier plot the reranker question demands |

**Open decisions (block the next step)**

- **D1 — keep the cross-encoder?** It costs 33ms of a 48ms budget, loses 11
  points of exact-id recall (0.4015 → 0.2885), ties on source-page metrics,
  and is +0.03 (noise) on topical relevance (M5f). Options: (a) drop it by
  default, keep it behind a flag; (b) keep it, and find the metric that
  justifies it; (c) keep it only for a "deep" tier. **Needs a better eval
  than n=10 before deciding.**
- **D2 — the corpus.** The toy came from the unranked slice (no English
  filter, spam-heavy, multilingual). A 1M density-ranked slice is
  English-only. Toy numbers do not transfer; state this before comparing.
- **D3 — the 1M scale blockers.** `build_filter` holds a token set per doc
  (~50GB at 5M); `build_index` posting lists are O(n·K); `stage_ingest` holds
  every URL. Fix before the 1M build, not during it.

**Measurement discipline**

- Every gate/sweep run appends to `runs.jsonl` with the **full config**
  (`nprobe`, `top_coarse`, `max_pair_tokens`, `batch_size`, `n`).
- Notebook rows that lack knobs are provenance-poor and are labelled as
  such; they cannot support a before/after claim.
- Numbers quoted in `README.md` / `EXPERIMENTS.md` must be re-derivable from
  a committed artifact (`runs.jsonl`, `sweeps/*/`, `eval/*`). The docs audit
  found several that were not — those are the ones that got corrected.
