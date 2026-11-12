# tests/test_ann_recall.py — T3: ANN recall against an exact ground truth.
#
# The old suite only queried vectors that WERE in the index (a self-match:
# the nearest neighbour is trivially yourself), so recall could be 0 and the
# test still passed. These tests build an index over SEMANTICALLY CLUSTERED
# vectors, query FRESH vectors that are not in it, and compare against a
# numpy brute-force ground truth (the same sign-bit score the index uses).
import itertools
import json
import struct
from pathlib import Path

import numpy as np
from fixtures.synth import make_vectors


def _write_ivf_index(tmp_path, vecs, k, seed=0):
    """Minimal valid index/ dir per the locked layout (nearest-centroid
    assignment stands in for K-means; the routing under test does not care
    how the centroids were produced)."""
    from exa_home.index_format import (
        CENTROIDS_FILE,
        CODES_FILE,
        DOC_IDS_FILE,
        LISTS_FILE,
        write_manifest,
    )

    n, dim = vecs.shape
    rng = np.random.default_rng(seed)
    centroids = vecs[rng.choice(n, size=k, replace=False)]
    assign = np.argmax(vecs @ centroids.T, axis=1)
    d = Path(tmp_path)
    centroids.astype("<f4").tofile(d / CENTROIDS_FILE)
    pack = np.packbits((vecs > 0).astype(np.uint8), axis=1, bitorder="little")
    assert pack.shape == (n, 32)
    pack.tofile(d / CODES_FILE)
    with open(d / LISTS_FILE, "wb") as f:
        f.write(struct.pack("<I", k))
        for c in range(k):
            rows = np.where(assign == c)[0].astype("<u4")
            f.write(struct.pack("<I", len(rows)))
            f.write(rows.tobytes())
    ids = [f"d{i}" for i in range(n)]
    (d / DOC_IDS_FILE).write_text(json.dumps(ids))
    write_manifest(str(d), {"dim": dim, "n_docs": n, "n_centroids": k})
    return ids


def _clustered(n: int, k: int, seed: int, noise: float = 0.6):
    """n unit vectors in k gaussian clusters (+ the cluster centres).

    Isotropic random vectors have no neighbourhood structure and IVF
    pruning over them is arbitrary — recall bars only mean something on
    data that actually clusters (the real embedding space does).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, 256))
    assign = rng.integers(0, k, size=n)
    vecs = centers[assign] + noise * rng.normal(size=(n, 256))
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    return vecs.astype(np.float32), centers


def _queries(centers: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picked = centers[rng.integers(0, len(centers), size=count)]
    q = picked + 0.6 * rng.normal(size=picked.shape)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    return np.ascontiguousarray(q.astype(np.float32))


def _exact_topk(vecs, q, top_k):
    """Brute-force ground truth: ranking by sign-bit dot product, i.e. the
    score the binary codes (and the Rust LUT) compute."""
    scores = np.where(vecs > 0, q, -q).sum(axis=1)
    return [f"d{i}" for i in np.argsort(-scores, kind="stable")[:top_k]]


def _recall(got, want):
    return len(set(got) & set(want)) / len(want)


def _build(tmp_path, n=800, k=16, seed=21):
    """Build an index over clustered vectors; return (index, vecs, centers)."""
    from ann_core import IvfIndex

    vecs, centers = _clustered(n, k, seed=seed)
    _write_ivf_index(tmp_path, vecs, k=k)
    return IvfIndex.load(str(tmp_path)), vecs, centers


def test_ivf_exhaustive_probe_matches_ground_truth_on_unseen_queries(tmp_path):
    """nprobe=K scans every list, so the ANN must BE the brute force —
    including on queries that are NOT in the index (the old test queried
    its own vector and could never fail)."""
    idx, vecs, centers = _build(tmp_path)
    k = 16
    for q in _queries(centers, 25, seed=987):
        want = _exact_topk(vecs, q, 10)
        got, scores = idx.search(np.ascontiguousarray(q), nprobe=k, n_coarse=10)
        assert got == want, f"exhaustive scan disagreed with brute force: {got}"
        assert scores == sorted(scores, reverse=True)


def test_ivf_partial_probe_recall_meets_measured_bar(tmp_path):
    """With a realistic nprobe the IVF pruning is APPROXIMATE. Asserting
    the measured bar (not tautology, not a self-match) is the point."""
    idx, vecs, centers = _build(tmp_path)
    queries = _queries(centers, 40, seed=4242)
    recs = [
        _recall(idx.search(q, nprobe=4, n_coarse=10)[0], _exact_topk(vecs, q, 10))
        for q in queries
    ]
    mean = float(np.mean(recs))
    print(f"recall@10 (nprobe=4, K=16): mean={mean:.3f} min={min(recs):.3f}")
    assert mean >= 0.99, f"recall@10 dropped to {mean:.3f}"
    assert min(recs) >= 0.5


def test_ivf_recall_query_row_is_not_self_matched(tmp_path):
    """Guard the guard: a query taken from the corpus ranks ITSELF first
    under brute force; an unseen query's top hit agrees with brute force
    (it is not the query itself — an unseen vector has no row)."""
    idx, vecs, centers = _build(tmp_path, n=200, k=4, seed=8)
    seen, _ = idx.search(np.ascontiguousarray(vecs[7]), nprobe=4, n_coarse=10)
    unseen_q = _queries(centers, 1, seed=77)[0]
    unseen, _ = idx.search(np.ascontiguousarray(unseen_q), nprobe=4, n_coarse=10)
    assert _exact_topk(vecs, vecs[7], 10)[0] == "d7"
    assert seen[0] == "d7"
    assert unseen == _exact_topk(vecs, unseen_q, 10)


def test_ivf_recall_does_not_degrade_as_nprobe_grows(tmp_path):
    """The bound that makes approximate search usable: recall must not
    DROP as nprobe grows (aggregate over unseen queries)."""
    idx, vecs, centers = _build(tmp_path)
    queries = _queries(centers, 30, seed=555)
    means = [
        float(
            np.mean(
                [
                    _recall(
                        idx.search(q, nprobe=p, n_coarse=10)[0],
                        _exact_topk(vecs, q, 10),
                    )
                    for q in queries
                ]
            )
        )
        for p in (1, 2, 4, 8, 16)
    ]
    print("recall@10 by nprobe:", [round(m, 3) for m in means])
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(means)), means
    assert means[-1] == 1.0  # exhaustive is exact


def test_exact_topk_helper_ranks_self_query_first():
    """The ground-truth helper is itself correct (self is top for the
    sign-bit score) — otherwise every recall bar above is empty."""
    vecs = make_vectors(50, 256, seed=1)
    assert _exact_topk(vecs, vecs[7], 5)[0] == "d7"
