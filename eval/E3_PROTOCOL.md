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

The target document id is read from `eval/sampled_targets.json` **by index**
(sampled on the box) — never transcribed by hand, so a mistyped hash cannot
silently mark every query a miss.

```python
import json, os, sys, urllib.request
import numpy as np

for cand in [os.getcwd(), "/marimo/exa-at-home"]:
    if os.path.isdir(os.path.join(cand, "exa_home")):
        ROOT = cand; break
else:
    raise SystemExit(f"repo not found; cwd={os.getcwd()}")
sys.path.insert(0, ROOT)

QPATH = f"{ROOT}/eval/questions_toy_20261021.json"
if not os.path.exists(QPATH):
    url = ("https://raw.githubusercontent.com/keypaa/exa-at-home/"
           "main/eval/questions_toy_20261021.json")
    print("downloading questions from", url)
    urllib.request.urlretrieve(url, QPATH)
QQ = json.load(open(QPATH))["questions"]
TARGETS = json.load(open(f"{ROOT}/eval/sampled_targets.json"))
Q = [{"question": q["question"], "target_id": TARGETS[q["n"]]["id"]} for q in QQ]
print(f"questions: {len(Q)}  (targets by index from sampled_targets.json)")

from exa_home.embed import Embedder
from exa_home.rerank import Reranker
from exa_home.store import ContentStore
from ann_core import IvfIndex

ann = IvfIndex.load(f"{ROOT}/index")
store = ContentStore(f"{ROOT}/index/store")
emb, rr = Embedder(), Reranker(max_pair_tokens=64, batch_size=32,
                               max_query_tokens=32)
qv = np.atleast_2d(emb.encode_queries([q["question"] for q in Q]))
K = json.load(open(f"{ROOT}/index/manifest.json"))["n_centroids"]

rows = []
for i, q in enumerate(Q):
    vec, tgt = qv[i], q["target_id"]
    c50, _ = ann.search(vec, nprobe=8, top_k=50)
    texts = [(store.get(c) or {}).get("text", "")[:2000] for c in c50]
    ranked, deg = rr.rerank(q["question"],
                            [{"id": c, "text": t} for c, t in zip(c50, texts)])
    r50 = [r["id"] for r in ranked]
    c_pos = c50.index(tgt) + 1 if tgt in c50 else 0
    r_pos = r50.index(tgt) + 1 if tgt in r50 else 0
    rows.append({"n": i, "target": tgt, "coarse_pos": c_pos,
                 "reranked_pos": r_pos, "degraded": bool(deg),
                 "question": q["question"]})
    print(f"Q{i:<3} coarse_pos={c_pos:<3} reranked_pos={r_pos:<3} deg={deg}")

n = len(rows)
hit = lambda p: 1 if 1 <= p <= 10 else 0
mrr = lambda p: (1.0 / p) if 1 <= p <= 10 else 0.0
c_hit = sum(hit(r["coarse_pos"]) for r in rows)
r_hit = sum(hit(r["reranked_pos"]) for r in rows)
c_mrr = sum(mrr(r["coarse_pos"]) for r in rows) / n
r_mrr = sum(mrr(r["reranked_pos"]) for r in rows) / n
wins = sum(1 for r in rows if hit(r["reranked_pos"]) and not hit(r["coarse_pos"]))
loss = sum(1 for r in rows if hit(r["coarse_pos"]) and not hit(r["reranked_pos"]))

print(f"\n=== E3 RESULT (n={n}) ===")
print(f"target recall@10   coarse={c_hit/n:.3f}   reranked={r_hit/n:.3f}")
print(f"target MRR@10       coarse={c_mrr:.3f}   reranked={r_mrr:.3f}")
print(f"paired: reranker wins {wins} / loses {loss} / ties {n-wins-loss}")

try:
    vecs = np.load(f"{ROOT}/vecs.npy", mmap_mode="r")
    ids = json.load(open(f"{ROOT}/ids.json"))
    pos = {d: i for i, d in enumerate(ids)}
    ceil = 0
    for i in range(n):
        scores = np.asarray(vecs) @ qv[i]
        exact = set(np.argsort(-scores, kind="stable")[:10].tolist())
        got, _ = ann.search(qv[i], nprobe=K, top_k=10)
        ceil += len(exact & {pos[g] for g in got if g in pos}) / 10
    print(f"binary ceiling (nprobe=K={K}, full scan vs exact float): {ceil/n:.3f}")
    print(f"  routing loss vs quantization loss: {c_hit/n:.3f} @ nprobe=8 "
          f"vs {ceil/n:.3f} at the ceiling")
except Exception as e:
    print(f"(ceiling measurement skipped: {e})")

json.dump({"rows": rows, "coarse_recall10": c_hit/n, "reranked_recall10": r_hit/n,
           "coarse_mrr10": c_mrr, "reranked_mrr10": r_mrr,
           "wins": wins, "losses": loss, "ties": n - wins - loss},
          open(f"{ROOT}/eval/e3_result.json", "w"), indent=1)
print("\nsaved eval/e3_result.json — download it and give it to me")
```
