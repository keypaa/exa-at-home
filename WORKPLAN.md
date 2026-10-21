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
| 6 | Cleanup wave 4: dead code, duplication, config, tests | done (scripts wave still landing) | 108 Python + 28 Rust tests green, ruff clean, CI green; behaviour change limited to the `top_k`→`n_coarse` rename (a loud error, not a silent alias) |
| 7 | Cleanup wave 5: docs truth pass | done | every doc number re-derived from an artifact; the historical plan is banner-marked |
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

### D1 experiment design (E3) — question-grounded retrieval eval

**The hypothesis being tested.** The cross-encoder is trained on
question→passage (MS MARCO). Every query we have ever measured it with was
NOT a question: 200-char document prefixes (the sweep) or URL slugs (M5f).
If the reranker is evaluated out of its training distribution, then "it
loses 11 points of exact-id recall" may be an artifact of the *queries*,
not a property of the model. E3 removes that confound.

**The design.**
- 50 (query, target document) pairs. The target document is sampled from
  the corpus; the query is a *question* written about that document's
  content (grounded, natural language, in-distribution for the reranker).
- The question set is written once and committed as an artifact
  (`eval/questions_*.json`), so the experiment is reproducible and
  re-runnable by anyone.
- Metric: **target-document recall@10** (is the page the question was
  written about in the top-10?) and **MRR of the target**, coarse vs
  reranked, same 50-candidate pool, paired per query.
- Secondary (optional, human): topical labels on the non-target results —
  that is where a reranker can show a difference the target metric cannot.

**Why this design and not a bigger sweep.**
- A bigger sweep of the same metric cannot settle it: the effect to resolve
  is ~3 points, and at n=10 queries a sign test has no power.
- The paired design removes query-difficulty variance: every query goes to
  both systems, so win/loss/tally is a direct comparison.
- The target-document metric has a known correct answer, so it needs no
  human labels to be unambiguous.
- It tests the actual hypothesis (query distribution) instead of
  re-measuring the known result.

**Decision rule — fixed BEFORE running, to avoid moving the goalposts.**
- E3 shows the reranker winning on target recall@10 or MRR → **keep it**
  (option b); the lesson is "the metric and the query type were wrong", and
  the serving default stays as measured.
- E3 shows a tie → **drop it from the default path** (option a) and expose
  it as an opt-in flag: it costs 33ms and no measurement supports it.
- E3 shows the reranker losing → **drop it** (option a). The project
  becomes a pure binary-IVF engine: ~14ms p50, 0.40 exact-id recall, one
  less moving part. That is a legitimate outcome, not a failure.
- A win **only for question-like queries** (option c, a "deep" tier) is a
  third possible answer and would justify keeping the model behind a
  quality tier.

**Cost.** ~30 min to write the 50 questions (once, reusable), ~5 min of box
time; the eval reuses the existing sweep/serve machinery.

**Measurement discipline**

- Every gate/sweep run appends to `runs.jsonl` with the **full config**
  (`nprobe`, `top_coarse`, `max_pair_tokens`, `batch_size`, `n`).
- Notebook rows that lack knobs are provenance-poor and are labelled as
  such; they cannot support a before/after claim.
- Numbers quoted in `README.md` / `EXPERIMENTS.md` must be re-derivable from
  a committed artifact (`runs.jsonl`, `sweeps/*/`, `eval/*`). The docs audit
  found several that were not — those are the ones that got corrected.
