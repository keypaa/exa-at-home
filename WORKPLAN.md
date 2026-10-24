# Work plan — one thing at a time

**Protocol (binding).** No new work starts until the current item is
designed, decided, *measured*, and recorded. "Measured" means a number in
`runs.jsonl` (with full config) or a row in `EXPERIMENTS.md` — not a
plausible-sounding claim in chat.

**Current focus:** get the system to 1M documents under the 100ms budget,
with the embed path fast enough and safe enough to make 5M reachable.

The map of the whole project: [`ROADMAP.md`](ROADMAP.md).

---

## TODO — the next steps, in order

| # | Item | State | Done when |
|---|------|-------|-----------|
| 1 | **Embed throughput**: fp16 + real GPU batch + length bucketing + explicit `max_seq_length` | **done (M8)** | measured 442 → 457 docs/s (+3.4%); **the embedder is CPU-bound, not GPU-bound** (fp16 + 32→1024 batches barely move it). The real lever is pre-tokenizing offline; see item 1b. |
| 1b | **Embed diagnostic + pre-tokenize**: profile one batch (torch.profiler) to settle CPU-tokenize vs transfer, then tokenize once offline and save token ids → pure-GPU forward passes | to do | docs/s measured; the 1M embed drops from ~37 min to a few minutes |
| 2 | **Resumable chunked embed** (per-chunk part files, resume from the last complete chunk) | to do | an interrupted 1M embed resumes instead of restarting |
| 3 | **The four gates** (below) — thresholds + one command | to do | `scripts/gate.py` exits non-zero on a regression; drop budgets pre-registered |
| 4 | **1M session plan** (written, like E3's) | to do | one paste, pre-registered acceptance criteria, no improvisation |
| 5 | **1M build** (unranked first-130 WET — the same selection rule as the toy) | to do | index/ built, gate run, 81k-vs-1M comparison reported. **Ingest is now the long pole** (2473 docs/s = 6.7 min @1M) — parallelize before 5M |
| 5b | **Ingest throughput**: parse in worker processes, write shards instead of one file | to do | docs/s measured; 5M ingest under ~5 min |
| 6 | **1M sweep**: nprobe × top_coarse × K — the scale curve | to do | curves in `runs.jsonl`; the `nprobe × (N/K)` latency model checked against measurement |
| 7 | **5M build** (stretch) | to do | only if 1M is clean and the box has the disk |
| 8 | **`lite`/`auto` serving tier** | to do | one switch, both paths benchmarked, frontier plotted |
| 9 | MCP server benchmarked against a real index | to do | p50 + recall for the shipped surface, not just the harness |

**Done so far:** M1–M6 (the ANN core + the serving stack), the measurement
instruments (sweep / gate / judge / ledger), three audits → three fix
waves, D1 (the reranker decision: **kept**), the cleanup wave (config
unification, dead code, scale blockers closed), and the docs truth pass.

---

## The gates (item 3) — "valid recall" means a thresholded, repeatable check

Four gates, one command, an exit code. Each has a **toy baseline** and a
**pre-registered drop budget**: the corpus grows, so absolute numbers
shift — the budget is the rule.

| Gate | Metric | Toy baseline | Rule at 1M |
|---|---|---|---|
| **G1 — ANN fidelity** | exact-id recall@10 on the 200 prefix queries (`gt.jsonl`) | coarse 0.4015 / reranked 0.2885 | must not drop more than **0.10** vs the same corpus's own baseline |
| **G2 — answer coverage** | target-doc recall@10 on the committed 50 questions (E3 set) | 0.740 | must not drop more than **0.10** vs 0.740 |
| **G3 — latency** | total p50 over 200 queries | 47.7ms | ≤ **100ms** (unchanged) |
| **G4 — relevance (noisy)** | judged precision@10 (LLM, n=10) | coarse 0.350 / reranked 0.380 | do-not-regress only — n=10 cannot gate |

G2 is what makes 1M a *scale* test rather than a rebuild: **the same 50
questions and the same 50 target pages**, against a 12× larger haystack.
Same eval set, more distractors — that is the experiment.

---

## The scale ladder (D2b) — what is actually reachable

Memory/time model for the box (4 CPU, 32 GiB RAM, RTX Pro 6000):

| Docs | vectors (fp32 256d) | WET download | content store | embed @2500 docs/s¹ | RAM risk |
|---|---|---|---|---|---|
| 81k (today) | 0.08 GB | 0.3 GB | 0.08 GB | **3.5 min** (measured, 389 docs/s) | none |
| 1M | 1.0 GB | ~8 GB | ~1 GB | **38 min** (at today's 389 docs/s; ~10 min after 1b) | none |
| **1M** | 1.0 GB | ~8 GB | ~1 GB | see above | none |
| **5M** | 5.1 GB | ~40 GB | ~5 GB | see above | comfortable |
| 10M | 10.2 GB | ~80 GB | ~10 GB | see above | comfortable (160GiB) |
| 20M | 20.4 GB | ~160 GB | ~20 GB | see above | safe (160GiB RAM, 503GB disk) |

¹ after item 1 — today's 442 docs/s on 3GB of a 96GB GPU leaves 5–8× on
the table.

**So: 1M is the milestone, 5M is the headline, 10M is a stretch**, and 20M
is out on this box (RAM and disk, not GPU). That is a "mini-Exa" worth
having; web-scale stays explicitly out of scope.

**The latency model that matters at scale:** the ANN scans
`nprobe × (N/K)` documents per query. At 81k/K=1024 that is 632 docs; at
1M/K=10k it is 800; at 5M/K=10k it is **4,000** — the scan would eat the
budget. K=30k at 5M brings it back to 1,333. **K must grow with N to keep
the scan flat** — that is the scaling law of this system, and it is what
the 1M sweep (item 6) is for.

---

## Open decisions

- **D1 — keep the cross-encoder? DECIDED 2026-10-21: KEEP.**
  E3 (50 real questions, known target pages): target recall@10 0.740 =
  0.740 (coarse = reranked), target MRR@10 0.641 → 0.712 (+0.071, sign
  test p = 0.18, not significant), 0 degraded queries. The pre-registered
  rule ("win on recall@10 OR MRR → keep") fired on MRR, so it stays — with
  the honest claim being "no harm on question-like queries, favourable
  direction", not "proven better". The earlier "it hurts" reading was a
  query-type artifact (document prefixes vs real questions). Artifacts:
  `eval/questions_toy_20261021.json`, `eval/e3_result_toy_20261021.json`;
  row M5h in EXPERIMENTS.md; LEARNINGS §13–14.
  Follow-up (not blocking): a bigger, human-judged eval would either
  confirm the +0.07 or dissolve it.
- **D2 — the corpus for the 1M build. DECIDED: same rule as the toy**
  (unranked first-N WET) so the scale comparison isolates scale. The
  density-ranked English slice is a **separate, separately reported run** —
  never the same row. The ASCII fraction is measured on both corpora and
  reported (toy: 73.7% mostly-ASCII) so any corpus drift is visible
  rather than assumed.
- **D3 — the 1M scale blockers. CLOSED** (cleanup wave 4): the filter
  build is two-pass streaming with roaring postings (**111MB @50k docs vs
  1356MB**); posting lists use one stable argsort (**9.85s → 0.11s** at
  n=1M/K=10k, byte-identical output); the centroid-assignment transient is
  pinned at 200MiB (was 1.9GB at K=10k); ground truth uses partition-based
  top-k (**777ms → 21ms** @5M, identical tie order); the sweep runs each
  query once instead of twice. Remaining: `stage_ingest`'s URL-dedup set
  (~660MB at 5M URLs, inherent to URL-second dedup).

---

## D1 experiment design (E3) — record of how the decision was made

Kept for the audit trail: the design, the pre-registered rule, and the
result. Protocol: [`eval/E3_PROTOCOL.md`](eval/E3_PROTOCOL.md).

**Hypothesis.** The cross-encoder is trained on question→passage (MS
MARCO). Every query we had measured it with was NOT a question —
200-char document prefixes (the sweeps) or URL slugs (M5f). If it was
being evaluated out of its training distribution, "it loses 11 points of
exact-id recall" may be measuring the query type, not the model.

**Design.** 50 (question, target document) pairs. The target is a page
sampled from the corpus; the question is written about that page's
content (grounded, natural, in-distribution). The question set is
committed (`eval/questions_toy_20261021.json`), so the experiment is
reproducible. Metric: target-document recall@10 and MRR@10, coarse vs
reranked over the same 50-candidate pool, paired per query. No human
labels needed — the correct answer is known by construction.

**Decision rule, fixed BEFORE running.** Win on recall@10 or MRR → keep.
Tie → drop to an opt-in flag. Lose → drop it. A win only for
question-like queries → a "deep" tier.

**Result.** recall@10 0.740 = 0.740; MRR@10 0.641 → 0.712 (p = 0.18);
0 degraded. → **Keep**, with the effect size and the p-value recorded.

---

## Measurement discipline

- Every gate/sweep run appends to `runs.jsonl` with the **full config**
  (`nprobe`, `top_coarse`, `max_pair_tokens`, `batch_size`, `n`).
- Notebook rows that lack knobs are provenance-poor and are labelled as
  such; they cannot support a before/after claim.
- Numbers quoted in `README.md` / `EXPERIMENTS.md` must be re-derivable
  from a committed artifact (`runs.jsonl`, `sweeps/*/`, `eval/*`). The
  docs audit found several that were not — those are the ones that got
  corrected.
