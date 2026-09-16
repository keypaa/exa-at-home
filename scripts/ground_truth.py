# scripts/ground_truth.py
"""Brute-force exact top-10 for sample queries. Usage:
python scripts/ground_truth.py --vecs vecs.npy --ids ids.json --nq 1000 --out gt.jsonl
python scripts/ground_truth.py --vecs vecs.npy --ids ids.json --queries q.jsonl --out gt.jsonl

--nq picks random corpus vectors (self-retrieval sanity). --queries embeds
real query TEXTS with Embedder (GPU box) so gt query k aligns with the
serving stack's pred query k — REQUIRED for a meaningful recall number
(mismatched query sets score ~0.0 by construction; found on-box 2026-09-16).
The two modes are mutually exclusive.
"""
import argparse, json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs", required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--nq", type=int, default=1000); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--queries", default=None,
                    help="q.jsonl of {query} texts: ground truth per TEXT query "
                         "(aligns with pred.jsonl query order)")
    a = ap.parse_args()
    vecs = np.load(a.vecs)
    ids = json.load(open(a.ids))
    if a.queries:
        from exa_home.embed import Embedder
        qtexts = [json.loads(l)["query"] for l in open(a.queries)
                  if l.strip()]
        qvecs = Embedder().encode_queries(qtexts)
        qs = [(k, qv) for k, qv in enumerate(qvecs)]
    else:
        rng = np.random.default_rng(a.seed)
        qi = rng.choice(len(vecs), size=min(a.nq, len(vecs)), replace=False)
        qs = [(k, vecs[i]) for k, i in enumerate(qi)]
    with open(a.out, "w") as f:
        for k, qv in qs:
            top = np.argsort(-(vecs @ qv), kind="stable")[:10]
            f.write(json.dumps({"query_id": k, "top10": [ids[j] for j in top]}) + "\n")

if __name__ == "__main__":
    main()
