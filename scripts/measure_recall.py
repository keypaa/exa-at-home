# scripts/measure_recall.py
"""recall@10 = |pred ∩ exact| / 10 averaged over gt.jsonl queries.

Usage:
    python scripts/measure_recall.py --gt gt.jsonl --pred pred.jsonl

gt.jsonl lines: {"query_id": k, "top10": [ids...]} (see scripts/ground_truth.py;
the key is `top10` — canonical per Task 4 review).
pred.jsonl lines: {"query_id": k, "top10": [ids...]}.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    gt = [json.loads(line) for line in open(a.gt)]
    pr = {json.loads(line)["query_id"]: json.loads(line)["top10"] for line in open(a.pred)}
    recs = [len(set(g["top10"]) & set(pr[g["query_id"]])) / 10 for g in gt]
    print(f"recall@10 = {sum(recs)/len(recs):.4f} over {len(recs)} queries")


if __name__ == "__main__":
    main()
