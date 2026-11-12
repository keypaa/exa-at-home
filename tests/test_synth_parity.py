# tests/test_synth_parity.py — fakes must match the PRODUCTION contracts.
#
# The Tier-0 suite used to be green while testing contracts no real
# backend implements: FakeEmbedder returned 1-D text-length vectors and
# FakeANN returned fixed descending scores (finding T3/P2-14). These tests
# compare the fakes against numpy references of the real scoring/filters.
import numpy as np
import pytest
from fixtures.backends import (
    FakeANN,
    FakeEmbedder,
    FakeReranker,
    _stable_vector,
    make_docs,
)

from exa_home.config import DIM


def _binary_dot(codes_sign: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The numpy reference for Rust `quant::binary_dot_packed`."""
    return np.where(codes_sign, q, -q).sum(axis=1)


def test_fake_embedder_matches_real_shape_and_normalization():
    e = FakeEmbedder()
    out = e.encode_queries(["hello", "world"])
    assert out.shape == (2, DIM) and out.dtype == np.float32
    assert e.calls == ["hello", "world"]
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    # Deterministic (same input -> same vector) like any seeded embedder.
    assert np.array_equal(e.encode_queries(["hello"])[0], out[0])


def test_fake_ann_scores_exactly_against_numpy_reference():
    vecs = np.vstack([_stable_vector(f"doc{i}") for i in range(9)])
    ids = [f"d{i}" for i in range(9)]
    ann = FakeANN(ids, vecs=vecs)
    q = _stable_vector("query")
    got_ids, got_scores = ann.search(q, n_coarse=4)
    want = _binary_dot(vecs > 0, q)
    ref_order = np.argsort(-want, kind="stable")[:4]
    assert got_ids == [ids[i] for i in ref_order]
    assert np.allclose(got_scores, want[ref_order], atol=1e-4)
    # Best-first, like the Rust sort (cmp_desc) truncation to n_coarse.
    assert got_scores == sorted(got_scores, reverse=True)


def test_fake_ann_orders_by_similarity_not_input_order():
    """The old fake scored 1.0/0.9/0.8 by POSITION: ranking by similarity
    was never exercised at all."""
    vecs = np.vstack([_stable_vector(f"doc{i}") for i in range(9)])
    ann = FakeANN([f"d{i}" for i in range(9)], vecs=vecs)
    # Pick a query whose reference ranking does NOT start at row 0.
    for i in range(9):
        q = vecs[i]
        ref = _binary_dot(vecs > 0, q)
        top = int(np.argmax(ref))
        if top != 0:
            break
    ids, scores = ann.search(q, n_coarse=9)
    assert ids == [f"d{r}" for r in np.argsort(-ref, kind="stable")[:9]]
    assert scores == sorted(scores, reverse=True)
    assert ids[0] != "d0"  # not input order: similarity put another row first


def test_fake_ann_filters_match_rust_semantics():
    docs = make_docs(3)
    # Make the docs distinguishable: only d1 has term "alpha", only d2 is
    # dated inside September, only d0 is on a foreign domain.
    docs["d0"]["url"] = "https://other.example.org/0"
    docs["d0"]["text"] = "neural retrieval passage with beta terms"
    docs["d0"]["date"] = "2026-09-01"
    docs["d1"]["text"] = "alpha and beta together in one passage"
    docs["d1"]["date"] = "2026-05-04"
    docs["d2"]["text"] = "alpha only here and nothing else"
    docs["d2"]["date"] = "2026-09-01"
    ann = FakeANN(list(docs), docs=docs)
    q = _stable_vector("q")

    # domains: OR over hostnames (www. stripped, lowercased).
    ids, _ = ann.search(q, domains=["example.com"])
    assert set(ids) == {"d1", "d2"}

    # terms: AND over lowercase tokens.
    ids, _ = ann.search(q, terms=["alpha", "beta"])
    assert ids == ["d1"]
    ids, _ = ann.search(q, terms=["alpha"])
    assert set(ids) == {"d1", "d2"}

    # month_range: inclusive; docs without a month bucket are excluded.
    ids, _ = ann.search(q, month_range=("2026-01", "2026-06"))
    assert set(ids) == {"d1"}

    # allow: row-id whitelist (d0 is row 0).
    ids, _ = ann.search(q, allow=[0])
    assert ids == ["d0"]


def test_fake_ann_empty_results_after_filter():
    ann = FakeANN(list(make_docs(3)), docs=make_docs(3))
    ids, scores = ann.search(_stable_vector("q"), terms=["notpresentanywhere"])
    assert ids == [] and scores == []


def test_fake_reranker_degraded_flag_matches_real_contract():
    r = FakeReranker()
    out, degraded = r.rerank("q", [{"id": "a", "coarse_score": 0.5}])
    assert degraded is False and out[0]["score"] is not None

    r = FakeReranker(degraded=True)
    cands = [{"id": "a", "coarse_score": 0.5}, {"id": "b", "coarse_score": 0.9}]
    out, degraded = r.rerank("q", cands)
    assert degraded is True
    # Degraded keeps coarse order and adds NO score (orch falls back).
    assert [c["id"] for c in out] == ["a", "b"]
    assert all("score" not in c for c in out)


def test_fakes_wire_into_the_real_dag_without_shape_errors():
    from fixtures.backends import FakeStore

    from exa_home.orch import build_search_dag, run_search

    dag = build_search_dag(
        FakeEmbedder(), FakeANN(), FakeReranker(), FakeStore(make_docs())
    )
    res = run_search(dag, "topic 3", {}, top_k=3)
    scores = [r["score"] for r in res["results"]]
    assert scores == sorted(scores, reverse=True)
    assert {r["id"] for r in res["results"]} <= set(make_docs(3))


@pytest.mark.parametrize("degraded", [False, True])
def test_fake_store_get_many_matches_get(degraded):
    from fixtures.backends import FakeStore

    s = FakeStore(make_docs(3))
    assert s.get_many(["d0", "d2", "nope"]) == {
        "d0": s.get("d0"),
        "d2": s.get("d2"),
        "nope": None,
    }
