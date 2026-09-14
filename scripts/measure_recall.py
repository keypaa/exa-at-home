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
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--min-recall", type=float, default=None,
                    help="exit 1 when recall@10 < threshold (default: print only, exit 0)")
    a = ap.parse_args(argv)
    gt = [json.loads(line) for line in open(a.gt)]
    pr = {json.loads(line)["query_id"]: json.loads(line)["top10"] for line in open(a.pred)}
    recs = []
    skipped = 0
    for g in gt:
        if g["query_id"] not in pr:
            print(f"measure_recall: warning: no prediction for query_id "
                  f"{g['query_id']} — skipping", file=sys.stderr)
            skipped += 1
            continue
        recs.append(len(set(g["top10"]) & set(pr[g["query_id"]])) / 10)
    if not recs:
        print("measure_recall: no comparable queries", file=sys.stderr)
        return 1 if a.min_recall is not None else 0
    recall = sum(recs) / len(recs)
    suffix = f" ({skipped} skipped: missing predictions)" if skipped else ""
    print(f"recall@10 = {recall:.4f} over {len(recs)} queries{suffix}")
    if a.min_recall is not None:
        print(f"measure_recall: recall@10 {recall:.4f} vs "
              f"threshold {a.min_recall:.4f} -> "
              f"{'PASS' if recall >= a.min_recall else 'FAIL'}")
        return 0 if recall >= a.min_recall else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
