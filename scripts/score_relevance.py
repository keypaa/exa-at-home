#!/usr/bin/env python3
"""Score a relevance-judged eval set: precision@10 per serving configuration.

Reads the review file (the query set + the blinded A/B presentation) and a
labels file (one 20-bit string per query, in review-file order), then reports
precision@10 for the two systems the review encodes:

    A = coarse ANN top-10          (no rerank — the "Lite" path)
    B = MiniLM-L6 reranked top-10   (the "Auto" path)

Usage:
    python scripts/score_relevance.py \
        --review eval/review_toy_20261021.json \
        --labels eval/labels_llm_judge_20261021.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SYSTEMS = (("A", "coarse ANN (no rerank)"), ("B", "reranked (MiniLM-L6)"))


def score(review: list[dict], labels: dict) -> dict:
    per_query, per_system = [], {k: [] for k, _ in SYSTEMS}
    for s in review:
        qid = str(s["qid"])
        bits = labels.get(qid) or labels.get(int(qid, -1))
        if bits is None:
            raise ValueError(f"no labels for query {qid}")
        bits = bits.replace(" ", "")
        entries = s["entries"]
        if len(bits) != len(entries):
            raise ValueError(
                f"query {qid}: {len(bits)} labels vs {len(entries)} entries")
        row = {"qid": s["qid"], "query": s["query"]}
        for key, _name in SYSTEMS:
            hits = sum(
                int(bits[n]) for n, e in enumerate(entries) if e["label"] == key
            )
            per_system[key].append(hits / 10)
            row[key] = hits / 10
        per_query.append(row)
    return {"per_query": per_query, "per_system": per_system}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    a = ap.parse_args(argv)

    review = json.load(open(a.review, encoding="utf-8"))
    blob = json.load(open(a.labels, encoding="utf-8"))
    labels = blob["labels"] if isinstance(blob, dict) and "labels" in blob else blob
    res = score(review, labels)

    out = {
        "n_queries": len(review),
        "precision_at_10": {
            name: round(sum(res["per_system"][key]) / len(review), 4)
            for key, name in SYSTEMS
        },
        "per_query": res["per_query"],
    }
    d = [
        r["B"] - r["A"] for r in res["per_query"]
    ]
    out["mean_delta_reranked_minus_coarse"] = round(st.mean(d), 4)
    out["queries_where_reranked_better"] = sum(1 for x in d if x > 0)
    out["queries_where_coarse_better"] = sum(1 for x in d if x < 0)
    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"queries: {out['n_queries']}")
    print(f"{'system':<28}{'precision@10':>14}")
    for key, name in SYSTEMS:
        print(f"{name:<28}{out['precision_at_10'][name]:>14.3f}")
    print(f"\nmean delta (reranked - coarse): "
          f"{out['mean_delta_reranked_minus_coarse']:+.3f} "
          f"({out['queries_where_reranked_better']} better / "
          f"{out['queries_where_coarse_better']} worse)")
    print("\nper query:")
    for r in res["per_query"]:
        print(f"  Q{r['qid']} {r['query'][:46]:<46} "
              f"coarse={r['A']:.1f} reranked={r['B']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
