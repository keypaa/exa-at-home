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

recall@10 = 0.29 vs brute force sounds bad until you see coarse-only was
0.027: the funnel + reranker deliver a 10x lift. Absolute overlap punishes
any rank shuffling among near-duplicates; the lift is the honest story of
whether the architecture works. Always pair recall with its baseline.

## 5. Know your harness failure modes — we measured one pred five times

The first nprobe "curve" was flat 0.2885 throughout because the pred cell
hardcoded nprobe=8: five gate runs, one shared prediction file. A sweep is
only as honest as its varying path — every knob in the ledger row must
trace to code that actually varies. (M5c; pred cell now follows the sliders;
sweep rows carry full config by construction.)

## 6. Funnels are non-monotonic — the metric is not the objective

top_coarse 50 → 100 → 200 gave recall 0.293 → 0.250 → 0.317, reproduced to
the digit. The cross-encoder optimizes its own relevance, not dot-product
order: candidates 51–100 displace exact hits, 101–200 bring better ones.
Wider is not better; operating points beat maxima. (M5d; 50 stays.)

## 7. Find knees, don't chase maxima

nprobe 1→8: +87% recall for ~3ms. 8→16: +6% for ~nothing. The knee at 8 is
the decision; everything past it is budget spent for pride. Same shape will
reappear at every scale — re-find it there, don't assume it transfers.

## 8. One knob per row, or the row means nothing

If recall moves after three simultaneous changes, you've learned nothing.
The ledger enforces it mechanically (one config delta per row). Sweep discipline
is not tidiness — it's what makes each EXPERIMENTS.md row a claim instead
of an anecdote.

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
