# scripts/measure_recall.py
"""recall@10 = |pred ∩ exact| / 10 averaged over gt.jsonl queries.

Usage:
    python scripts/measure_recall.py --gt gt.jsonl --pred pred.jsonl

gt.jsonl lines: {"query_id": k, "top10": [ids...]} (see scripts/ground_truth.py;
the key is `top10` — canonical per Task 4 review).
pred.jsonl lines: {"query_id": k, "top10": [ids...]}.

THE HOME of the recall@10 formula: `recall_at_10(gt_rows, pred_rows)` is
imported by scripts/sweep.py (the knob grid) and scripts/score_relevance.py
(the judged relevance set) so the metric exists exactly once, with ONE
error policy: a gt row without a prediction is SKIPPED and COUNTED, never
indexed blind (`pr[g["query_id"]]` used to raise a bare KeyError and abort
the sweep, losing every remaining grid point).
"""

import argparse
import sys


def recall_at_10(gt_rows, pred_rows, on_skip=None) -> tuple[float | None, int]:
    """recall@10 over the gt rows that HAVE a prediction.

    gt_rows: iterable of {"query_id": k, "top10": [ids...]}
    pred_rows: iterable of {"query_id": k, "top10": [ids...]}

    `on_skip` (optional) is called with the query_id of every gt row that
    has no prediction (the caller prints the warning).

    Returns (recall, n_skipped); recall is None when nothing is comparable.
    """
    pr = {r["query_id"]: r["top10"] for r in pred_rows}
    recs: list[float] = []
    skipped = 0
    for g in gt_rows:
        top10 = pr.get(g["query_id"])
        if top10 is None:
            skipped += 1
            if on_skip is not None:
                on_skip(g["query_id"])
            continue
        recs.append(len(set(g["top10"]) & set(top10)) / 10)
    if not recs:
        return None, skipped
    return sum(recs) / len(recs), skipped


def hits_at_10(hits: int) -> float:
    """`hits` relevant entries out of the 10-entry window (hit rate).

    The shared top-10 arithmetic (scripts/score_relevance.py scores a
    judged relevance set: hits out of the 10 presented entries).
    """
    return hits / 10


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="exit 1 when recall@10 < threshold (default: print only, exit 0)",
    )
    a = ap.parse_args(argv)
    from scripts.jsonl import read_jsonl

    gt = list(read_jsonl(a.gt))
    pr = list(read_jsonl(a.pred))

    def _warn(query_id):
        print(
            f"measure_recall: warning: no prediction for query_id "
            f"{query_id} — skipping",
            file=sys.stderr,
        )

    recall, skipped = recall_at_10(gt, pr, on_skip=_warn)
    if recall is None:
        print("measure_recall: no comparable queries", file=sys.stderr)
        return 1 if a.min_recall is not None else 0
    suffix = f" ({skipped} skipped: missing predictions)" if skipped else ""
    n_compared = len(gt) - skipped
    print(f"recall@10 = {recall:.4f} over {n_compared} queries{suffix}")
    if a.min_recall is not None:
        print(
            f"measure_recall: recall@10 {recall:.4f} vs "
            f"threshold {a.min_recall:.4f} -> "
            f"{'PASS' if recall >= a.min_recall else 'FAIL'}"
        )
        return 0 if recall >= a.min_recall else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
