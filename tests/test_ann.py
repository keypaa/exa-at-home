# tests/test_ann.py
import numpy as np
import pytest
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


def test_m2_binary_search_agrees_on_easy_query():
    from ann_core import AnnIndex
    vecs = make_vectors(200, 256, seed=7)
    ids = [f"d{i}" for i in range(200)]
    codes = (vecs > 0).astype(np.uint8)
    pack = np.packbits(codes, axis=1, bitorder="little")  # 200×32
    idx = AnnIndex.from_binary(pack, ids)
    got_ids, _ = idx.search_binary(vecs[42], top_k=10)
    assert "d42" in got_ids  # easy self-query must survive quantization


def test_m2_binary_search_matches_numpy_reference():
    """search_binary must agree with a numpy reference of binary_dot_packed."""
    from ann_core import AnnIndex
    vecs = make_vectors(50, 256, seed=11)
    ids = [f"d{i}" for i in range(50)]
    pack = np.packbits((vecs > 0).astype(np.uint8), axis=1, bitorder="little")
    idx = AnnIndex.from_binary(pack, ids)
    q = vecs[3]
    want = np.array([
        float(np.sum(np.where(
            np.unpackbits(row, bitorder="little")[:256].astype(bool), q, -q)))
        for row in pack
    ])
    order = np.argsort(-want, kind="stable")[:5]
    got_ids, got_scores = idx.search_binary(q, top_k=5)
    assert got_ids == [f"d{i}" for i in order]
    assert np.allclose(got_scores, want[order], atol=1e-3)


def test_m2_from_binary_rejects_bad_shape():
    from ann_core import AnnIndex
    with pytest.raises(Exception):
        AnnIndex.from_binary(np.zeros((10, 31), dtype=np.uint8),
                             [f"d{i}" for i in range(10)])
