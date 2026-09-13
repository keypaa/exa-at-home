# tests/test_ann.py
import numpy as np
from fixtures.synth import make_vectors

def test_m1_brute_force_finds_nearest():
    from ann_core import AnnIndex
    vecs = make_vectors(200, 256, seed=7)
    ids = [f"d{i}" for i in range(200)]
    idx = AnnIndex(vecs, ids)
    q = vecs[42].copy()
    got_ids, scores = idx.search(q, top_k=5)
    assert got_ids[0] == "d42"
    assert scores[0] > scores[-1]
    # agreement with numpy exact
    exact = np.argsort(-(vecs @ q), kind="stable")[:5]
    assert got_ids == [f"d{i}" for i in exact]
