# tests/test_ann.py
import json
import struct
import subprocess
import sys
from pathlib import Path

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


# --- Task 6 (M3): IVF routing over a versioned index/ dir ---

def _write_ivf_index(tmp_path, vecs, k, seed=0):
    """Minimal valid index/ dir per the locked layout.

    Nearest-centroid assignment stands in for K-means (no sklearn in
    Tier 0); the loader and routing logic under test don't care how the
    centroids were produced.
    """
    from exa_home.index_format import (
        CENTROIDS_FILE, CODES_FILE, DOC_IDS_FILE, LISTS_FILE,
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


def test_ivf_exact_routing_matches_bruteforce(tmp_path):
    """nprobe=K must equal M2 brute force within fp-reassociation (M4 LUT)."""
    from ann_core import AnnIndex, IvfIndex
    vecs = make_vectors(300, 256, seed=3)
    ids = _write_ivf_index(tmp_path, vecs, k=8)
    idx = IvfIndex.load(str(tmp_path))
    q = vecs[0]
    got_ids, got_scores = idx.search(q, nprobe=8, top_k=10)
    pack = np.packbits((vecs > 0).astype(np.uint8), axis=1, bitorder="little")
    ref = AnnIndex.from_binary(pack, ids)
    want_ids, want_scores = ref.search_binary(q, top_k=10)
    assert got_ids == want_ids
    # M4: IVF scan is LUT-scored, so fp reassociation vs the naive M2 loop
    # costs ~1e-6; 1e-3 is the LUT==naive bar (cf. Rust lut_matches_naive).
    assert np.allclose(got_scores, want_scores, atol=1e-3)
    assert got_ids[0] == "d0"


def test_ivf_search_allow_list_filters(tmp_path):
    from ann_core import IvfIndex
    vecs = make_vectors(100, 256, seed=5)
    _write_ivf_index(tmp_path, vecs, k=4)
    idx = IvfIndex.load(str(tmp_path))
    allow = [7, 1, 5]  # deliberately unsorted: Rust sorts once per query
    got_ids, _ = idx.search(vecs[10], nprobe=4, top_k=10, allow=allow)
    assert sorted(got_ids) == ["d1", "d5", "d7"]


def test_ivf_load_rejects_corruption(tmp_path):
    from ann_core import IvfIndex
    from exa_home.index_format import CODES_FILE
    vecs = make_vectors(50, 256, seed=6)
    _write_ivf_index(tmp_path, vecs, k=4)
    p = Path(tmp_path) / CODES_FILE
    raw = bytearray(p.read_bytes())
    raw[0] ^= 0xFF
    p.write_bytes(bytes(raw))
    with pytest.raises(Exception):
        IvfIndex.load(str(tmp_path))


def test_train_centroids_importable():
    import inspect
    from scripts.train_centroids import main as train
    params = list(inspect.signature(train).parameters)
    assert params[:5] == ["vecs_path", "k", "sample", "out", "seed"]
    # verbose is an optional additive flag (default False), not a contract break
    if len(params) > 5:
        assert params[5] == "verbose"
        assert inspect.signature(train).parameters["verbose"].default is False


def test_train_centroids_cli_help():
    script = Path(__file__).resolve().parent.parent / "scripts" / "train_centroids.py"
    r = subprocess.run([sys.executable, str(script), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    for flag in ("--vecs", "--k", "--sample", "--out"):
        assert flag in r.stdout


def test_m4_scores_match_m2(tmp_path):
    """M4 no-regression: IVF search() (now LUT-scored) agrees with the M2
    naive `binary_dot_packed` numpy reference beyond fp-reassociation
    tolerance. Pins API stability (ids + float scores); the strong
    bit-exactness check is the Rust `lut_matches_naive` unit test."""
    from ann_core import IvfIndex
    vecs = make_vectors(300, 256, seed=3)
    ids = _write_ivf_index(tmp_path, vecs, k=8)
    idx = IvfIndex.load(str(tmp_path))
    q = vecs[0]
    got_ids, got_scores = idx.search(q, nprobe=8, top_k=10)
    pack = np.packbits((vecs > 0).astype(np.uint8), axis=1, bitorder="little")
    want = np.array([
        float(np.sum(np.where(
            np.unpackbits(row, bitorder="little")[:256].astype(bool), q, -q)))
        for row in pack
    ])
    order = np.argsort(-want, kind="stable")[:10]
    assert got_ids == [ids[i] for i in order]
    assert np.allclose(got_scores, want[order], atol=1e-3)
    assert all(isinstance(s, float) for s in got_scores)


@pytest.mark.cloud_only
def test_train_centroids_mini_run(tmp_path):
    """Cloud-only: needs sklearn (declared dep, unavailable offline)."""
    from scripts.train_centroids import main as train
    vecs = make_vectors(200, 256, seed=9)
    vpath = tmp_path / "vecs.npy"
    out = tmp_path / "centroids.npy"
    np.save(vpath, vecs)
    train(str(vpath), 4, 200, str(out))
    c = np.load(out)
    assert c.shape == (4, 256) and c.dtype == np.float32


def test_direct_coarse_recall_bounds_reversed_skew(tmp_path):
    """M5 honesty pin (Tier-0 synth analogue): MockReranker reverses the
    coarse top-10, so the cloud 0.0270 coarse-only number understates the
    true direct-ANN top-10. Here: direct recall must exceed reversed
    recall on the same index — the skew direction, pinned as regression."""
    from ann_core import IvfIndex
    vecs = make_vectors(2000, 256, seed=3)
    ids = _write_ivf_index(tmp_path, vecs, k=64)
    idx = IvfIndex.load(str(tmp_path))
    pack = np.packbits((vecs > 0).astype(np.uint8), axis=1,
                       bitorder="little")
    bits = np.unpackbits(pack, axis=1,
                         bitorder="little")[:, :256].astype(bool)
    rng = np.random.default_rng(0)
    direct, reversed_ = [], []
    for i in rng.choice(len(vecs), size=50, replace=False):
        q = vecs[i]
        want = set(np.argsort(-np.where(bits, q, -q).sum(axis=1),
                              kind="stable")[:10])
        got, _ = idx.search(q, nprobe=8, top_k=50)
        rows = [int(g[1:]) for g in got]  # "d123" -> 123 (ids are row-order)
        direct.append(len(set(rows[:10]) & want) / 10)
        # MockReranker reverses the full 50-candidate list: its top-10 is
        # the coarse WORST 10 — the cloud 0.0270 skew, reproduced.
        reversed_.append(len(set(rows[::-1][:10]) & want) / 10)
    # Direct top-10 must beat worst-10 on average (else the corpus or the
    # index is degenerate); direct must clear a sanity floor.
    assert sum(direct) / len(direct) > sum(reversed_) / len(reversed_)
    assert sum(direct) / len(direct) > 0.15  # sanity: funnel works on synth


# --- Task 8 (M6-partial): inverted filter index + in-scan intersect ---

FILTER_DOCS = [
    {"id": "d0", "url": "https://example.com/a", "title": "",
     "text": "alpha beta gamma of shared words here", "date": "2026-01-15"},
    {"id": "d1", "url": "https://other.org/b", "title": "",
     "text": "alpha delta epsilon of shared words here", "date": "2026-02-20"},
    {"id": "d2", "url": "https://example.com/c", "title": "",
     "text": "zeta eta theta of shared words here", "date": "2026-01-25"},
]


def _write_ivf_index_with_filter(tmp_path, vecs, k, docs, seed=0):
    """Full index/ dir incl. filter/: flat IVF files + filter/ subdir, with the
    manifest rewritten LAST so the filter files are covered by verify."""
    from exa_home.index_format import write_manifest
    from scripts.build_filter import build
    ids = _write_ivf_index(tmp_path, vecs, k, seed=seed)
    build(docs, str(Path(tmp_path) / "filter"))
    n, dim = vecs.shape
    write_manifest(str(tmp_path), {"dim": dim, "n_docs": n, "n_centroids": k})
    return ids


def test_filter_build_writes_locked_layout(tmp_path):
    from scripts.build_filter import build
    from pyroaring import BitMap
    out = tmp_path / "filter"
    build(FILTER_DOCS, str(out))
    for name in ("domains.bin", "domains.json", "dates.bin", "dates.json",
                 "terms.bin", "terms.json"):
        assert (out / name).is_file(), name
    # domains.json offsets slice the right bitmaps out of domains.bin.
    raw = (out / "domains.bin").read_bytes()
    doms = json.loads((out / "domains.json").read_text())
    assert set(doms) == {"example.com", "other.org"}
    off, ln = doms["example.com"]
    assert set(BitMap.deserialize(raw[off:off + ln])) == {0, 2}
    off, ln = doms["other.org"]
    assert set(BitMap.deserialize(raw[off:off + ln])) == {1}
    # dates bucket per YYYY-MM.
    dates = json.loads((out / "dates.json").read_text())
    assert set(dates) == {"2026-01", "2026-02"}
    # terms: lowercase alnum tokens len>=3; short tokens dropped.
    terms = json.loads((out / "terms.json").read_text())
    assert "alpha" in terms and "zeta" in terms
    assert "of" not in terms
    assert all(t == t.lower() and len(t) >= 3 for t in terms)


def test_filter_build_respects_max_terms(tmp_path):
    from scripts.build_filter import build
    build(FILTER_DOCS, str(tmp_path / "f"), max_terms=3)
    terms = json.loads((tmp_path / "f" / "terms.json").read_text())
    # "shared"/"words"/"here" have df=3 each, the max; everything else <= 2.
    assert set(terms) == {"shared", "words", "here"}


def test_filter_domains_restricts_results(tmp_path):
    from ann_core import IvfIndex
    vecs = make_vectors(3, 256, seed=21)
    _write_ivf_index_with_filter(tmp_path, vecs, 2, FILTER_DOCS)
    idx = IvfIndex.load(str(tmp_path))
    q = vecs[0]
    all_ids, _ = idx.search(q, nprobe=2, top_k=10)
    assert set(all_ids) == {"d0", "d1", "d2"}
    dom_ids, _ = idx.search(q, nprobe=2, top_k=10, domains=["example.com"])
    assert set(dom_ids) <= set(all_ids)
    assert set(dom_ids) == {"d0", "d2"}
    # Unknown domain matches nothing (empty, not an error).
    assert idx.search(q, nprobe=2, top_k=10, domains=["nope.example"])[0] == []


def test_filter_month_range_and_terms(tmp_path):
    from ann_core import IvfIndex
    vecs = make_vectors(3, 256, seed=21)
    _write_ivf_index_with_filter(tmp_path, vecs, 2, FILTER_DOCS)
    idx = IvfIndex.load(str(tmp_path))
    q = vecs[0]
    jan, _ = idx.search(q, nprobe=2, top_k=10, month_range=("2026-01", "2026-01"))
    assert set(jan) == {"d0", "d2"}
    feb, _ = idx.search(q, nprobe=2, top_k=10, month_range=("2026-02", "2026-02"))
    assert set(feb) == {"d1"}
    wide, _ = idx.search(q, nprobe=2, top_k=10, month_range=("2026-01", "2026-02"))
    assert set(wide) == {"d0", "d1", "d2"}
    # Terms conjoin (AND); lookup is case-insensitive like the index keys.
    one, _ = idx.search(q, nprobe=2, top_k=10, terms=["ALPHA"])
    assert set(one) == {"d0", "d1"}
    two, _ = idx.search(q, nprobe=2, top_k=10, terms=["alpha", "beta"])
    assert set(two) == {"d0"}
    # Dimensions conjoin: domain AND term.
    both, _ = idx.search(q, nprobe=2, top_k=10,
                         domains=["example.com"], terms=["alpha"])
    assert set(both) == {"d0"}
    # Explicit allow-list composes with filter bitsets (intersection).
    combo, _ = idx.search(q, nprobe=2, top_k=10, allow=[0, 1],
                          domains=["example.com"])
    assert set(combo) == {"d0"}


def test_filter_kwargs_without_filter_dir_raise(tmp_path):
    from ann_core import IvfIndex
    vecs = make_vectors(3, 256, seed=21)
    _write_ivf_index(tmp_path, vecs, 2)  # no filter/ subdir
    idx = IvfIndex.load(str(tmp_path))
    with pytest.raises(Exception, match="filter"):
        idx.search(vecs[0], nprobe=2, top_k=10, domains=["example.com"])
