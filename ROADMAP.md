# Roadmap

The one-page map: where this project came from, what it is doing now, what
comes next, and what it is ultimately for.

---

## 1. Ultimate goal

**A working, honest, measured reproduction of Exa's serving architecture at
1M–5M documents — in a repo good enough that a stranger could trust it.**

Concretely, that means four things, and all four have to be true:

1. **It works.** The full funnel (embed → binary IVF → rerank → snippet) over
   **5M** real Common Crawl pages (1M is the milestone, 5M the headline,
   10M a stretch), served over MCP, at a p50 inside the 100ms budget.
2. **Every decision is measured.** No design choice in the code exists
   because it felt right. Each has a row in `EXPERIMENTS.md` and a
   measurement in `runs.jsonl` that someone else can re-run.
3. **It is professional.** Modular, linted, CI-gated, tested, documented —
   the thing you would not be embarrassed to show an engineer, and the thing
   you would be happy to keep working in.
4. **The knowledge is written down.** `LEARNINGS.md` holds the principles
   that outlive this codebase, not just the numbers of this project.

This is a **learning repo** first and a product second. A working mini-Exa is
the vehicle; the transferable knowledge is the payload.

---

## 2. What we did (2026-09-13 → 2026-10-21)

| Phase | What | Result |
|---|---|---|
| **M1–M4** | The ANN core in Rust/PyO3: float brute force → sign-bit binary quantization → IVF routing → 64×16 nibble LUT scoring | `lut_scan_2k` 60µs vs 1.978ms naive (33×); recall harness + ground truth |
| **M5** | The serving stack: Canon-lite DAG orchestrator, FastMCP server (`home_search`/`home_contents`), content store with a byte-offset index, cross-encoder rerank with degraded fallback, corpus pipeline, literate notebook, cloud runbook | Toy on 81k real Common Crawl pages: **p50 47.7ms vs a 100ms budget (PASS)**, recall@10 0.2885, 0 degraded queries |
| **Instruments** | `sweep.py` (4-D knob grid, timestamped artifacts, full-config ledger), `e2e_latency.py` (p50/p99 + waterfall integrity check), `score_relevance.py` (judged precision@10), `runs.jsonl` as the single source of truth | Every run re-derivable; the ledger is what the docs cite |
| **Three audits** | Read-only reviews: correctness, docs-truth, architecture | 25+ real bugs found and fixed in three waves; scale blockers closed (filter build 21× less RAM, posting lists 86× faster, ground truth 37× faster); every doc number re-derived from an artifact |
| **D1** | The reranker decision, designed before it was measured (E3: 50 real questions, known targets), rule pre-registered, then executed | **Kept.** recall@10 0.740 = 0.740, MRR 0.641 → 0.712, p = 0.18 (favourable, not significant). The earlier "it hurts" reading was a query-type artifact |

Knowledge captured: `EXPERIMENTS.md` (M1–M6, M5b–M5h), `LEARNINGS.md`
(14 principles), `WORKPLAN.md` (protocol + open decisions).

---

## 3. Where we are — the current goal

**Make the system work at 1M documents, and measure what scale costs and
what it buys.**

The toy (81k) proved the architecture; it did not prove it scales. 1M is
where the corpus stops being a curiosity and starts being a system: the
centroid count goes 1k → 10k, the posting lists get 12× longer, the ANN
scan touches 10× more rows, the content store grows past 1GB, and the
filter index has to hold hundreds of thousands of bitmaps. Every one of
those is a place the design can break, and every break is a lesson.

Two sub-goals, in order:

1. **Make the embed path fast and safe.** Two problems, one place:
   - *Slow*: we run fp32 weights (1.3GB) with sentence-transformers'
     **default batch of 32**, and we never told it the sequence length — so
     442 docs/s uses ~3GB of a 96GB GPU. fp16/bf16 + a real batch
     (1024+) + length bucketing should buy 4–8×, which is what moves 5M
     from "3 hours" to "half an hour".
   - *Fragile*: the stage is all-or-nothing (memmap, then `os.replace` at
     the end). A 1M embed is ~37 minutes of continuous GPU; if it dies, the
     work is gone. Chunk it and make it resumable.
2. **Build it with a written plan** — one session, one paste, pre-registered
   acceptance criteria, the way E3 was run.

**Corpus decision (D2), made:** the 1M slice uses the **same selection rule
as the toy** (unranked first-N WET) so the scale comparison isolates
scale. The density-ranked English slice is a **separate, separately
reported run** — never the same row. The ASCII fraction of both corpora is
measured and reported (toy: 73.7% mostly-ASCII) so any corpus drift is
visible rather than assumed.

---

## 4. What we need to do next

| # | Item | Why it matters | Cost |
|---|---|---|---|
| 1 | Resumable chunked embed + chunked centroid training | Makes a 37-min GPU run survivable on a box that can vanish | small |
| 2 | The 1M session plan (written, like E3's) | One paste, no improvisation; pre-registered acceptance criteria | ~1h of my time, then ~1.5h attended box time |
| 3 | 1M build + gate + sweep | The scale experiment itself | box time |
| 4 | **The four gates** (ANN fidelity, answer coverage, latency, relevance) | "Valid recall" means a thresholded, repeatable check. WORKPLAN.md now defines all four with toy baselines and pre-registered drop budgets; they still need implementing as one command | small |
| 5 | A relevance metric that scales | Our id-overlap metric is not a quality metric (LEARNINGS §4/§13). The human-judged eval is n=50 and doesn't scale — this is the next *instrument* we need | medium |
| 6 | The serving tier (`lite` / `auto`) | The Exa-3.0 mirror: coarse-only vs coarse+rerank, one switch, one measured frontier. D1 just made this concrete: both paths are now validated | medium |
| 7 | The MCP server as a real product | It starts now (`python -m exa_home.mcp_server`) but has never been benchmarked against the real index | small-medium |

**Then, only if the above are solid:** the optional/stretch items —
answer/similar DAG nodes, a Matryoshka fine-tune, and the "better
representation instead of a reranker" lever Exa 3.0 pulled (a larger
embedder, so the binary codes lose less).

---

## 5. What we are deliberately NOT doing

Web-scale (billions), GPU ANN, live crawling, freshness pipelines,
auth/billing/multi-tenancy, a UI, and training our own foundation
embedding model. The scale ladder stops at what one 32 GiB / RTX Pro 6000
box can hold: **1M → 5M → 10M**, and 20M is out on RAM and disk, not on
the GPU. All of these are explicitly out of scope (spec §3) —
saying so here is what keeps the project from expanding into a
twelve-month programme we never finish.

---

## 6. How we work

**Decide, then measure, one knob at a time.**

1. **Design** — write the experiment down before running it, including the
   decision rule and what each outcome would mean.
2. **Decide** — apply the pre-registered rule. Not after the fact.
3. **Measure** — one change, one row, full config in the ledger.
4. **Record** — `EXPERIMENTS.md` (numbers + mechanism), `LEARNINGS.md`
   (principles), `WORKPLAN.md` (state + open decisions).

The rules that keep us honest, learned the hard way:
- Every knob in a ledger row, or the row means nothing (§8).
- Measure a baseline with an instrument that isn't sabotaged (§4).
- When a component measures badly, check whether the measurement is in the
  component's domain before cutting it (§13).
- A negative result is a result — publish it (§12).
- "Not significant" is an answer; write down the sample size that would
  settle it (§14).
