# Learnings

What this repo taught us that the numbers alone don't say. Each item: the
principle, the evidence, where it bit us. New entries go here, not in chat.

## 1. The bottleneck is never where the spans say — until you've profiled

`rerank_ms` read 1400ms and blamed the GPU. `torch.profiler` showed 10ms of
real GPU compute hiding behind ~1400ms of JSON shard scans: 50 cold
`ContentStore.get` calls at ~28ms each, inside a span named "rerank".
Fix the span, not the GPU. (M5b; `store/offsets.json` took total 4025ms → 46ms.)

## 2. A parked Minor on the hot path is a Major with a delay fuse

The Task-2 reviewer flagged "full shard scan on every LRU miss" as Minor;
it was parked as "acceptable for V1" and cost two days of GPU witch-hunt.
Severity of a finding must be judged by where it sits (hot path amplifies),
not by how it looks in the diff.

## 3. Measure the unit you claim to optimize

The rerank span covered store-fetch + tokenize + model + syncs as one
number, so no experiment could move it intelligibly. The observability
split (fetch vs model) turned one mystery number into two actionable ones.
If a stage can't be decomposed, you can't optimize it — instrument first.

## 4. Exact-id recall is the harshest lens — report lift, not just absolutes

recall@10 = 0.29 vs brute force sounds bad until you see what the same
measurement says without the reranker: 0.40. The reranker *lowers* exact-id
overlap by 11 points, because it optimizes its own relevance notion and
displaces the embedding-space neighbours that the id-overlap metric
rewards. Two consequences we had to learn the hard way:

- **Never measure a baseline with a sabotaged instrument.** Our "coarse-only
  0.0270" was measured with a `MockReranker` that *reverses* the candidate
  list — so it scored the worst-10 of top-50, not the top-10. The honest
  direct measurement is 0.4015. The "10.7x rerank lift" was an artifact of
  that inverted baseline, and it lived in EXPERIMENTS.md and the README
  until an audit re-derived it.
- **A metric is not the objective.** Exact-id overlap measures the ANN; a
  cross-encoder exists to change the ranking. Judging the reranker on the
  ANN's metric is like judging a spell-checker on recall.

## 5. Know your harness failure modes — we measured one pred five times

The first nprobe "curve" was flat 0.2885 throughout because the pred cell
hardcoded nprobe=8: five gate runs, one shared prediction file. A sweep is
only as honest as its varying path — every knob in the ledger row must
trace to code that actually varies. (M5c; the pred cell now follows the
sliders, the sweep rows carry the full config, and the ledger records
recall with the run.)

## 6. Funnels are non-monotonic — and a timeout can masquerade as a curve

top_coarse 50 → 100 → 200 gave recall 0.293 → 0.250 → 0.317. The obvious
reading ("the middle width is a trap; the widest is best") was wrong: the
ledger showed `rerank_model_ms` pinned at exactly the 100ms timeout, so at
tc=200 half the queries had **degraded to coarse order** and the number was
a mixture of two different systems. Before believing a curve, check that
each point was produced by the system you think it was. (M5d, revised.)

## 7. Find knees, don't chase maxima

nprobe 1→8: +87% recall for ~3ms. 8→16: +6% for ~nothing. The knee at 8 is
the decision; everything past it is budget spent for pride. Batch size is
the same shape (b16 57ms / b32 42ms / b64 43ms). Same shape will reappear
at every scale — re-find it there, don't assume it transfers.

## 8. One knob per row, or the row means nothing

If recall moves after three simultaneous changes, you've learned nothing.
The sweep rows enforce it mechanically (one config delta per row, full config
in the ledger). The 2026-10-07 notebook rows (runs.jsonl 0–15) predate that
and carry no knobs — they cannot be attributed to a setting, and the
notebook now records every knob it used.

## 9. At 1M scale, models are fixed cost — time is the constraint, not memory

1M needs <8GB VRAM of 96 and <15GB RAM of 160; even 5M fits. What scales is
bulk-embed wall time (30–60 min of GPU) and download bytes — i.e. attention
and session planning, not hardware. Size the operation, not the machine.

## 10. Build-time work beats query-time work, always

Offset index at build time (one scan, saved to disk) vs lazy per-query scans;
centroids trained offline vs served; manifest written last. Every cost pushed
left of serving is paid once; every cost left on the hot path is paid per
query forever, then misattributed (see §1).

## 11. Do the science where the science is welcome

Sustained unattended GPU compute on a hosted notebook reads as abuse no
matter the intent. Literate notebooks (reasoning + imports + plots), attended
stages with progress, committed ledgers — these aren't ceremony, they're what
keep the lab open. Second strike is a ban; act like it.

## 12. A negative result is a result — publish it

The cross-encoder was the most expensive part of the pipeline (33ms of a
48ms budget) and, after honest measurement, no metric showed it earning
its keep: −11 points of exact-id recall, a tie on source-page metrics, and
+0.03 (noise, n=10) on judged topical relevance. The tempting move was to
keep the claim and quietly re-define the metric. The right move was to
publish the negative result, and to plan the experiment that could
overturn it (a bigger, human-judged eval). A learning repo that only
contains positive results is a marketing artifact.
