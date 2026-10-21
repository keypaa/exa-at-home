# E3 — question-grounded retrieval eval (the D1 experiment)

**Question this answers:** does the cross-encoder earn its 33ms?

**Why a new eval.** Every measurement we have of the reranker used a query
that is *not a question*: 200-char document prefixes (the sweeps) or URL
slugs (M5f). MiniLM-L6 is trained on question→passage (MS MARCO). If we
have been evaluating it out of its training distribution, "it loses 11
points of exact-id recall" may be measuring our query type, not the model.

**Design.** 50 (question, target document) pairs. The target is a document
sampled from the corpus; the question is written about that document's
content. Known correct answer → no human labels needed. Paired: every
question goes to both systems over the same 50-candidate pool.

**Decision rule (fixed in advance — see WORKPLAN.md §D1).**
- reranker wins on target recall@10 / MRR → keep it
- tie → drop it from the default path, keep as an opt-in flag
- reranker loses → drop it; the project is a pure binary-IVF engine

---

## Step 1 — sample the target documents (run on the box)

```python
import json, os, random, sys
import numpy as np

for cand in [os.getcwd(), "/marimo/exa-at-home"]:
    if os.path.isdir(os.path.join(cand, "exa_home")):
        ROOT = cand; break
else:
    raise SystemExit(f"repo not found; cwd={os.getcwd()}")
sys.path.insert(0, ROOT)

# English-ish, substantial pages so the questions are answerable.
def ascii_heavy(s):
    return sum(ch.isascii() for ch in s) / max(1, len(s)) > 0.9

pool = []
with open(f"{ROOT}/docs.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        body = " ".join((d.get("text") or "").split())
        if len(body) < 800 or not ascii_heavy(body[:400]):
            continue
        pool.append((d["id"], d.get("url", ""), body[:320]))
print(f"pool: {len(pool)}")
rng = random.Random(20261021)
picked = rng.sample(pool, 50)
json.dump([{"n": i, "id": d, "url": u, "text": t} for i, (d, u, t) in enumerate(picked)],
          open(f"{ROOT}/eval/sampled_targets.json", "w"), indent=1)
for i, (d, u, t) in enumerate(picked):
    print(f"{i:>2} | {d} | {u[:45]} | {t[:150]}")
```

Copy that output (or download `eval/sampled_targets.json`) — the questions
are written from it, grounded in each document.

## Step 2 — write the questions

One question per sampled document, in natural question form, answerable
from that document's content. Store as `eval/questions_toy_<date>.json`:

```json
[
  {"n": 0, "target_id": "sha1:...", "question": "How do I ...?"},
  ...
]
```

Keep the `target_id` exactly as printed in step 1.

## Step 3 — run the paired comparison (on the box)

```python
import json, os, sys
import numpy as np

for cand in [os.getcwd(), "/marimo/exa-at-home"]:
    if os.path.isdir(os.path.join(cand, "exa_home")):
        ROOT = cand; break
sys.path.insert(0, ROOT)

from exa_home.embed import Embedder
from exa_home.rerank import Reranker
from ann_core import IvfIndex

Q = json.load(open(f"{ROOT}/eval/questions_toy_20261021.json"))
ann = IvfIndex.load(f"{ROOT}/index")
emb, rr = Embedder(), Reranker(max_pair_tokens=64, batch_size=32, max_query_tokens=32)
qv = np.atleast_2d(emb.encode_queries([q["question"] for q in Q]))

coarse_hit = reranked_hit = 0
coarse_mrr = reranked_mrr = 0.0
wins = losses = ties = 0
for i, q in enumerate(Q):
    vec = qv[i]
    c50, _ = ann.search(vec, nprobe=8, top_k=50)
    texts = []
    for cid in c50:
        with open(f"{ROOT}/docs.jsonl", encoding="utf-8") as f:   # slow but simple
            for line in f:
                d = json.loads(line)
                if d["id"] == cid:
                    texts.append(" ".join((d.get("text") or "").split())[:2000])
                    break
    ranked, deg = rr.rerank(q["question"], [{"id": c, "text": t} for c, t in zip(c50, texts)])
    rr_ids = [r["id"] for r in ranked]

    tgt = q["target_id"]
    c_pos = c50.index(tgt) + 1 if tgt in c50 else 0
    r_pos = rr_ids.index(tgt) + 1 if tgt in rr_ids else 0
    coarse_hit += c_pos <= 10 and c_pos > 0
    reranked_hit += r_pos <= 10 and r_pos > 0
    coarse_mrr += 1 / c_pos if c_pos else 0
    reranked_mrr += 1 / r_pos if r_pos else 0
    if (r_pos <= 10) > (c_pos <= 10):
        wins += 1
    elif (c_pos <= 10) > (r_pos <= 10):
        losses += 1
    else:
        ties += 1
    print(f"Q{i:<3} target_found_coarse={c_pos:<3} reranked={r_pos:<3} deg={deg}")

n = len(Q)
print(f"\nqueries: {n}")
print(f"target recall@10  coarse={coarse_hit/n:.3f}  reranked={reranked_hit/n:.3f}")
print(f"target MRR        coarse={coarse_mrr/n:.3f}  reranked={reranked_mrr/n:.3f}")
print(f"paired: reranker wins {wins} / loses {losses} / ties {ties}")
```

(The docs lookup in step 3 is deliberately simple — replace it with the
content store's `get()` if it is slow.)
