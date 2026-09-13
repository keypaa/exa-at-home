# scripts/ground_truth.py
"""Brute-force exact top-10 for sample queries. Usage:
python scripts/ground_truth.py --vecs vecs.npy --ids ids.json --nq 1000 --out gt.jsonl
"""
import argparse, json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs", required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--nq", type=int, default=1000); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    vecs = np.load(a.vecs)
    ids = json.load(open(a.ids))
    rng = np.random.default_rng(a.seed)
    qi = rng.choice(len(vecs), size=min(a.nq, len(vecs)), replace=False)
    with open(a.out, "w") as f:
        for k, i in enumerate(qi):
            top = np.argsort(-(vecs @ vecs[i]))[:10]
            f.write(json.dumps({"query_id": k, "top10": [ids[j] for j in top]}) + "\n")

if __name__ == "__main__":
    main()
