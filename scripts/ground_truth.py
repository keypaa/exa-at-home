# scripts/ground_truth.py
"""Brute-force exact top-10 for sample queries. Usage:
python scripts/ground_truth.py --vecs vecs.npy --ids ids.json --nq 1000 --out gt.jsonl
python scripts/ground_truth.py --vecs vecs.npy --ids ids.json --queries q.jsonl --out gt.jsonl

--nq picks random corpus vectors (self-retrieval sanity). --queries embeds
real query TEXTS with Embedder (GPU box) so gt query k aligns with the
serving stack's pred query k — REQUIRED for a meaningful recall number
(mismatched query sets score ~0.0 by construction; found on-box 2026-09-16).
The two modes are mutually exclusive.

`ground_truth(...)` is the importable function (the pipeline calls it
IN-PROCESS — no subprocess); `main(argv)` is the CLI wrapper around it.
"""

import argparse
import json

import numpy as np


def _reference_top_k(scores, k: int):
    """The reference implementation (a full stable argsort).

    Used ONLY for non-finite scores, where the comparison-based selection
    below has no defined order.
    """
    return np.argsort(-scores, kind="stable")[:k]


def top_k_indices(scores, k: int = 10):
    """Indices of the k largest scores, ties broken by ASCENDING row id.

    Exactly `np.argsort(-scores, kind="stable")[:k]` — but O(n) instead of
    O(n log n): a full argsort over 5M docs costs ~0.4s and a 40MB index
    allocation PER QUERY (the ground truth is 1000 queries).

    The tie order matters: the stable argsort breaks equal scores by
    ascending row id, so the k-th slot of a tie block goes to the
    SMALLEST row ids (see the test for a tie case).
    """
    n = scores.shape[0]
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        # Every row qualifies — but the ORDER is still (-score, row id).
        idx = np.arange(n, dtype=np.int64)
        return idx[np.lexsort((idx, -scores[idx]))]
    if not np.isfinite(scores).all():
        # NaN/inf: the sort order is not well defined by the comparisons
        # below, so fall back to the reference implementation.
        return _reference_top_k(scores, k)
    # The k-th largest score value (the tie threshold).
    threshold = np.partition(scores, n - k)[n - k]
    better = np.flatnonzero(scores > threshold)
    need = k - better.size
    if need > 0:
        # Ties at the threshold: the smallest row ids win.
        ties = np.flatnonzero(scores == threshold)[:need]
        idx = np.concatenate((better, ties))
    else:
        idx = better
    # Descending score, then ascending row id (np.lexsort: LAST key wins).
    order = np.lexsort((idx, -scores[idx]))
    return idx[order]


def ground_truth(
    vecs_path: str,
    ids_path: str,
    out: str,
    nq: int = 1000,
    seed: int = 0,
    queries: str | None = None,
    embedder=None,
) -> int:
    """Write the exact top-10 per query to `out` (JSONL). Returns #queries.

    `queries`: path to a q.jsonl of {"query": text} rows — the texts are
    embedded so gt query k aligns with pred query k (the serving stack's
    query order). Without it, `nq` random corpus vectors are used (the
    self-retrieval sanity path).

    `embedder`: injectable query embedder (default: the real
    `Embedder`). The injection point exists so the Tier-0 suite can build
    a ground truth without loading a 335M-parameter model — the query
    texts still have to be embedded, one way or another.
    """
    vecs = np.load(vecs_path)
    with open(ids_path) as f:
        ids = json.load(f)
    if queries:
        from scripts.jsonl import read_jsonl

        qtexts = [row["query"] for row in read_jsonl(queries)]
        if embedder is None:
            from exa_home.embed import Embedder

            embedder = Embedder()
        qvecs = embedder.encode_queries(qtexts)
        qs = list(enumerate(qvecs))
    else:
        rng = np.random.default_rng(seed)
        qi = rng.choice(len(vecs), size=min(nq, len(vecs)), replace=False)
        qs = [(k, vecs[i]) for k, i in enumerate(qi)]
    with open(out, "w") as f:
        for k, qv in qs:
            scores = vecs @ qv
            top = top_k_indices(scores, 10)
            f.write(json.dumps({"query_id": k, "top10": [ids[j] for j in top]}) + "\n")
    return len(qs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--queries",
        default=None,
        help="q.jsonl of {query} texts: ground truth per TEXT query "
        "(aligns with pred.jsonl query order)",
    )
    a = ap.parse_args(argv)
    ground_truth(a.vecs, a.ids, a.out, nq=a.nq, seed=a.seed, queries=a.queries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
