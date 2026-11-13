# tests/test_build_index.py (Task 13)
"""Tier-0 for scripts/build_index.py + offline coverage for the net scripts.

select_slice.py / download_wet.py are network-bound (DuckDB-over-HTTPS,
CloudFront fetches) and cannot run in the offline sandbox — see docs/RUNBOOK.md
"why no local tests". What IS asserted offline here: the locked SQL string
content, all three argparse CLIs (--help), download resume/skip logic via
mocked transport, and the full build_index.py assemble path.

The build_index round-trip asserts the IvfIndex.load contract WITHOUT
importing ann_core (same pattern as Task 12): write via build_index.main,
verify via exa_home.index_format + independent numpy/struct/json readback.
"""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fixtures.synth import make_docs, make_vectors

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _write_inputs(tmp_path, n=200, k=8, seed=13):
    vecs = make_vectors(n, 256, seed=seed)
    docs = make_docs(n, seed=seed)
    ids = [d["id"] for d in docs]
    rng = np.random.default_rng(seed)
    centroids = vecs[rng.choice(n, size=k, replace=False)]
    vpath, ipath, cpath, dpath = (
        tmp_path / "vecs.npy",
        tmp_path / "ids.json",
        tmp_path / "centroids.npy",
        tmp_path / "docs.jsonl",
    )
    np.save(vpath, vecs)
    ipath.write_text(json.dumps(ids))
    np.save(cpath, centroids)
    dpath.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    return vecs, ids, docs, centroids, vpath, ipath, cpath, dpath


def test_build_index_roundtrip_200_docs(tmp_path):
    """200 synth docs: assemble -> manifest verifies -> bytes parse back
    exactly per the locked layout (the IvfIndex.load read contract)."""
    from exa_home.index_format import (
        CENTROIDS_FILE,
        CODES_FILE,
        DOC_IDS_FILE,
        LISTS_FILE,
        verify_manifest,
    )
    from scripts.build_index import main as build

    n, k, dim = 200, 8, 256
    vecs, ids, docs, centroids, vpath, ipath, cpath, dpath = _write_inputs(
        tmp_path, n, k
    )
    out = tmp_path / "index"

    stats = build(str(vpath), str(ipath), str(cpath), str(dpath), str(out))
    assert stats == {"n_docs": n, "n_centroids": k, "n_stored": n}

    m = verify_manifest(str(out))
    assert m["n_docs"] == n and m["n_centroids"] == k and m["dim"] == dim

    # Independent readback, one assertion per locked file.
    got_c = np.fromfile(out / CENTROIDS_FILE, dtype="<f4").reshape(k, dim)
    assert np.array_equal(got_c, centroids.astype(np.float32))
    got_codes = np.fromfile(out / CODES_FILE, dtype=np.uint8).reshape(n, 32)
    want_codes = np.packbits((vecs > 0).astype(np.uint8), axis=1, bitorder="little")
    assert np.array_equal(got_codes, want_codes)
    raw = (out / LISTS_FILE).read_bytes()
    (k2,) = struct.unpack_from("<I", raw, 0)
    assert k2 == k
    off, total = 4, 0
    for _ in range(k):
        (ln,) = struct.unpack_from("<I", raw, off)
        off += 4 + 4 * ln
        total += ln
    assert total == n and off == len(raw)
    # Lists partition rows by nearest centroid (argmax assignment, chunked).
    assign = np.argmax(vecs @ centroids.T, axis=1)
    off = 4
    for c in range(k):
        (ln,) = struct.unpack_from("<I", raw, off)
        off += 4
        rows = struct.unpack_from(f"<{ln}I", raw, off) if ln else ()
        off += 4 * ln
        assert sorted(rows) == sorted(np.where(assign == c)[0].tolist())
    assert json.loads((out / DOC_IDS_FILE).read_text()) == ids

    # The default <out>/store serves back doc text (home_contents path).
    from exa_home.store import OFFSETS_FILENAME, ContentStore

    # Offset index written at build time. The content store is CONTENT, not
    # index state: it is deliberately EXCLUDED from the manifest (hashing
    # every shard made IvfIndex.load O(corpus)).
    assert (out / "store" / OFFSETS_FILENAME).exists()
    assert not any(k.startswith("store/") for k in m["files"])
    cs = ContentStore(str(out / "store"))
    assert cs.get(ids[0])["text"] == docs[0]["text"]
    assert cs.get(ids[-1])["text"] == docs[-1]["text"]
    assert cs._index is not None and len(cs._index) == n


def test_build_index_rejects_shape_mismatch(tmp_path):
    from scripts.build_index import main as build

    _vecs, ids, _docs, _c, vpath, ipath, cpath, dpath = _write_inputs(tmp_path)
    ipath.write_text(json.dumps(ids[:-1]))  # 199 ids vs 200 vecs
    with pytest.raises(ValueError, match="ids"):
        build(str(vpath), str(ipath), str(cpath), str(dpath), str(tmp_path / "index"))
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((10, 128), dtype=np.float32))
    with pytest.raises(ValueError, match="256"):
        build(str(bad), str(ipath), str(cpath), str(dpath), str(tmp_path / "index2"))


def _write_corpus(tmp_path, name, n=20, seed=1):
    """A (vecs, ids, docs) corpus triple on disk."""
    docs = make_docs(n, seed=seed)
    vecs = make_vectors(n, 256, seed=seed)
    ids = [d["id"] for d in docs]
    vpath = tmp_path / f"{name}-vecs.npy"
    ipath = tmp_path / f"{name}-ids.json"
    dpath = tmp_path / f"{name}-docs.jsonl"
    np.save(vpath, vecs)
    ipath.write_text(json.dumps(ids))
    dpath.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    return str(vpath), str(ipath), str(dpath)


def _fake_train_centroids(monkeypatch):
    """MiniBatchKMeans is cloud-only; a deterministic stand-in is enough."""
    import scripts.train_centroids as tc

    def main(vecs_path, k, sample, out, seed=0, verbose=False):
        v = np.load(vecs_path)
        np.save(out, v[: max(1, min(k, len(v)))].astype(np.float32))

    monkeypatch.setattr(tc, "main", main)


def test_stage_index_refuses_to_reuse_an_index_built_from_another_corpus(
    tmp_path, monkeypatch
):
    """A stale index must never be reused for a different corpus.

    The old stage_index short-circuited on the mere EXISTENCE of
    <out>/manifest.json: running the FULL pipeline after the toy reused the
    4-doc toy index while stage_queries built q.jsonl/gt.jsonl from the FULL
    corpus — the gate then measured a mismatched index, invisibly.
    """
    import scripts.build_index as bi
    from scripts.run_pipeline import stage_index

    _fake_train_centroids(monkeypatch)
    builds = []
    orig_build = bi.main

    def counting_build(*args, **kwargs):
        builds.append(kwargs.get("out", args[4] if len(args) > 4 else None))
        return orig_build(*args, **kwargs)

    monkeypatch.setattr(bi, "main", counting_build)

    va, ia, da = _write_corpus(tmp_path, "a", n=20, seed=1)
    vb, ib, db = _write_corpus(tmp_path, "b", n=30, seed=2)
    out = str(tmp_path / "index")

    m1 = stage_index(va, ia, da, out, k=4, use_filter=False)
    assert m1["n_docs"] == 20
    assert len(builds) == 1

    # Same corpus -> reuse (no rebuild).
    m2 = stage_index(va, ia, da, out, k=4, use_filter=False)
    assert m2["n_docs"] == 20
    assert len(builds) == 1

    # Different corpus in the same dir -> rebuild, never a silent reuse.
    m3 = stage_index(vb, ib, db, out, k=4, use_filter=False)
    assert len(builds) == 2
    assert m3["n_docs"] == 30


def test_stage_index_rebuilds_when_the_provenance_is_unknown(tmp_path, monkeypatch):
    """An index with no recorded provenance cannot be proven reusable."""
    import scripts.build_index as bi
    from scripts.run_pipeline import stage_index

    _fake_train_centroids(monkeypatch)
    builds = []
    orig_build = bi.main

    def counting_build(*args, **kwargs):
        builds.append(kwargs.get("out", args[4] if len(args) > 4 else None))
        return orig_build(*args, **kwargs)

    monkeypatch.setattr(bi, "main", counting_build)

    va, ia, da = _write_corpus(tmp_path, "a", n=20, seed=1)
    out = str(tmp_path / "index")
    stage_index(va, ia, da, out, k=4, use_filter=False)
    assert len(builds) == 1

    os.remove(os.path.join(out, bi.SOURCE_FILE))  # legacy / hand-made index
    stage_index(va, ia, da, out, k=4, use_filter=False)
    assert len(builds) == 2  # rebuilt, not silently reused


def test_toy_and_full_pipelines_do_not_share_an_index_dir(monkeypatch):
    """The toy owns index/; the full build writes its own corpus-named dir."""
    import scripts.run_pipeline as pipe

    seen_index: list = []
    seen_queries: list = []

    monkeypatch.setattr(pipe, "stage_fetch", lambda *a, **k: [])
    monkeypatch.setattr(pipe, "stage_download", lambda *a, **k: "")
    monkeypatch.setattr(pipe, "stage_ingest", lambda *a, **k: {})
    monkeypatch.setattr(pipe, "stage_embed", lambda *a, **k: ((1, 256), "float32"))
    monkeypatch.setattr(
        pipe,
        "stage_index",
        lambda *a, **k: (seen_index.append(a[3]), {})[1],
    )
    monkeypatch.setattr(
        pipe,
        "stage_queries",
        lambda *a, **k: (
            seen_queries.append(k.get("queries_path", "q.jsonl")),
            {},
        )[1],
    )

    class _Args:
        force = False
        verbose = False
        workers = 1
        batch = 8
        wet_files = 20
        k = 100

    pipe.cmd_toy(_Args())
    pipe.cmd_full(_Args())

    assert seen_index == ["index/", "index_20wet/"]
    assert seen_index[0] != seen_index[1]
    # ... and the query artifacts follow the corpus, so the gate can never
    # pair the toy index with the full corpus's queries.
    assert seen_queries == ["q.jsonl", "q_20wet.jsonl"]


def _write_inputs_for(tmp_path, tag, n, k, seed):
    """Second corpus written to its OWN paths (different ids/text)."""
    vecs = make_vectors(n, 256, seed=seed)
    docs = [
        {
            "id": f"{tag}-{i}",
            "url": f"https://example.com/{tag}/{i}",
            "title": "",
            "text": f"{tag} corpus document {i} " * 12,
            "date": "2026-01-01",
        }
        for i in range(n)
    ]
    ids = [d["id"] for d in docs]
    rng = np.random.default_rng(seed)
    centroids = vecs[rng.choice(n, size=k, replace=False)]
    vpath, ipath, cpath, dpath = (
        tmp_path / f"vecs_{tag}.npy",
        tmp_path / f"ids_{tag}.json",
        tmp_path / f"centroids_{tag}.npy",
        tmp_path / f"docs_{tag}.jsonl",
    )
    np.save(vpath, vecs)
    ipath.write_text(json.dumps(ids))
    np.save(cpath, centroids)
    dpath.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    return vpath, ipath, cpath, dpath


def _store_shard_text(store_dir) -> str:
    return "\n".join(p.read_text() for p in sorted(store_dir.glob("shard-*.jsonl")))


def test_build_index_rebuild_prunes_previous_corpus_store(tmp_path):
    """Rebuilding into the same out dir must REBUILD the derived store.

    The old code appended to the existing shards, so each rebuild grew the
    dir by a full corpus (and served two corpora' ids from one store).
    Correctness is unchanged, disk is not: this pins the pruning.
    """
    import json as _json

    from exa_home.store import OFFSETS_FILENAME
    from scripts.build_index import main as build

    n, k = 12, 4
    out = tmp_path / "index"

    _vecs, _ids, _docs, _cent, vpath, ipath, cpath, dpath = _write_inputs(
        tmp_path, n, k
    )
    build(str(vpath), str(ipath), str(cpath), str(dpath), str(out))
    store_a = out / "store"
    offsets_a = _json.loads((store_a / OFFSETS_FILENAME).read_text())
    bytes_a = sum(p.stat().st_size for p in store_a.glob("shard-*.jsonl"))
    assert len(offsets_a) == n and bytes_a > 0

    # Rebuild a DIFFERENT corpus into the same out dir.
    v2, i2, c2, d2 = _write_inputs_for(tmp_path, "beta", n, k, seed=99)
    stats = build(str(v2), str(i2), str(c2), str(d2), str(out))
    assert stats["n_stored"] == n
    assert stats["n_docs"] == n

    # Reference: the same rebuild into a fresh dir. The pruned rebuild must
    # match it byte-for-byte (no accumulation of the old corpus).
    out_fresh = tmp_path / "index_fresh"
    build(str(v2), str(i2), str(c2), str(d2), str(out_fresh))

    offsets_b = _json.loads((store_a / OFFSETS_FILENAME).read_text())
    shard_text = _store_shard_text(store_a)

    # Only the second corpus is on disk...
    assert set(offsets_b) == {f"beta-{i}" for i in range(n)}
    assert '"id": "beta-0"' in shard_text
    assert '"id": "d0"' not in shard_text
    # ... and the store is exactly as big as a fresh build of the same
    # corpus: nothing from the previous corpus survived.
    bytes_rebuilt = sum(p.stat().st_size for p in store_a.glob("shard-*.jsonl"))
    bytes_fresh = sum(
        p.stat().st_size for p in (out_fresh / "store").glob("shard-*.jsonl")
    )
    assert bytes_rebuilt == bytes_fresh, (
        f"store grew beyond a fresh rebuild: {bytes_rebuilt} vs {bytes_fresh}"
    )

    from exa_home.index_format import verify_manifest

    verify_manifest(str(out))  # the rest of the index is still consistent


def test_build_index_no_store_prunes_stale_store(tmp_path):
    """--no-store must not leave the previous corpus' store behind: the
    index would no longer match the text it serves (off-manifest)."""

    from exa_home.store import OFFSETS_FILENAME
    from scripts.build_index import main as build

    n, k = 8, 4
    out = tmp_path / "index"
    _vecs, _ids, _docs, _cent, vpath, ipath, cpath, dpath = _write_inputs(
        tmp_path, n, k
    )
    build(str(vpath), str(ipath), str(cpath), str(dpath), str(out))
    assert (out / "store" / OFFSETS_FILENAME).is_file()

    stats = build(
        str(vpath),
        str(ipath),
        str(cpath),
        str(dpath),
        str(out),
        no_store=True,
    )
    assert stats["n_stored"] == 0
    assert not (out / "store" / OFFSETS_FILENAME).exists()
    assert not list((out / "store").glob("shard-*.jsonl"))


def test_build_index_keep_store_flag_appends(tmp_path):
    """--no-reset-store keeps the opt-in append semantics (escape hatch)."""
    import json as _json

    from exa_home.store import OFFSETS_FILENAME
    from scripts.build_index import main as build

    n, k = 8, 4
    out = tmp_path / "index"
    _vecs, _ids, _docs, _cent, vpath, ipath, cpath, dpath = _write_inputs(
        tmp_path, n, k
    )
    build(str(vpath), str(ipath), str(cpath), str(dpath), str(out))
    before = sum(p.stat().st_size for p in (out / "store").glob("shard-*.jsonl"))

    build(
        str(vpath),
        str(ipath),
        str(cpath),
        str(dpath),
        str(out),
        reset_store=False,
    )
    after = sum(p.stat().st_size for p in (out / "store").glob("shard-*.jsonl"))
    assert after > before, "append semantics kept the old data (no pruning)"
    offsets = _json.loads((out / "store" / OFFSETS_FILENAME).read_text())
    assert len(offsets) == n  # ids are content digests: no duplicate entries


def test_build_index_filter_build_streams_documents(tmp_path, monkeypatch):
    """The filter builder gets a LAZY view of the corpus, not a list.

    The old code materialised every doc dict (plus a token set per doc
    inside the filter builder) before handing them over — multiple GB at
    the 5M-doc target.
    """
    import scripts.build_filter as bf
    from scripts.build_index import main as build

    _vecs, _ids, docs, _centroids, vpath, ipath, cpath, dpath = _write_inputs(
        tmp_path, n=60, k=4
    )
    seen: dict = {}
    orig = bf.build

    def spy(docs_arg, out_dir, max_terms=50_000):
        seen["is_list"] = isinstance(docs_arg, list)
        seen["len"] = len(docs_arg)
        seen["docs"] = list(docs_arg)
        return orig(docs_arg, out_dir, max_terms)

    monkeypatch.setattr(bf, "build", spy)

    out = tmp_path / "index"
    build(str(vpath), str(ipath), str(cpath), str(dpath), str(out), filter=True)

    assert seen["is_list"] is False
    assert seen["len"] == 60
    assert len(seen["docs"]) == 60
    assert seen["docs"][0]["url"] == docs[0]["url"]
    # ... and the filter artifacts are the real ones.
    fdir = out / "filter"
    assert (fdir / "terms.bin").exists()
    assert (fdir / "domains.json").exists()


def test_streamed_filter_matches_the_list_build(tmp_path):
    """Streaming the docs into the filter builder changes nothing on disk."""
    import scripts.build_filter as bf
    from scripts.build_index import main as build

    _vecs, _ids, docs, _centroids, vpath, ipath, cpath, dpath = _write_inputs(
        tmp_path, n=60, k=4
    )
    out = tmp_path / "index"
    build(str(vpath), str(ipath), str(cpath), str(dpath), str(out), filter=True)

    ref = tmp_path / "ref-filter"
    bf.build(list(docs), str(ref), 50_000)

    for name in (
        "terms.bin",
        "terms.json",
        "domains.bin",
        "domains.json",
        "dates.bin",
        "dates.json",
        "filter_stats.json",
    ):
        assert (out / "filter" / name).read_bytes() == (ref / name).read_bytes(), name


def test_stage_embed_streams_the_corpus(tmp_path, monkeypatch):
    """stage_embed embeds chunk by chunk and never holds the whole corpus.

    The old code did `docs = [json.loads(line) for line in f]` — every doc
    dict of the corpus in RAM at once (multiple GB at 5M docs).
    """
    import tracemalloc

    import exa_home.embed as embed_mod
    from scripts.run_pipeline import stage_embed

    n = 400
    docs = make_docs(n, seed=5)
    big = "x" * 4000
    dpath = tmp_path / "docs.jsonl"
    dpath.write_text(
        "\n".join(json.dumps({**d, "text": big + d["text"]}) for d in docs) + "\n"
    )
    corpus_bytes = dpath.stat().st_size

    calls: list[int] = []
    gpu_batches: list[int | None] = []

    class FakeEmbedder:
        def encode_docs(self, texts, batch_size=None):
            calls.append(len(texts))
            gpu_batches.append(batch_size)
            import numpy as np

            return np.zeros((len(texts), 256), dtype=np.float32)

    monkeypatch.setattr(embed_mod, "Embedder", FakeEmbedder)

    vpath = tmp_path / "vecs.npy"
    ipath = tmp_path / "ids.json"
    tracemalloc.start()
    try:
        shape, dtype = stage_embed(
            str(dpath), str(vpath), str(ipath), batch=32, chunk=32, verbose=False
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert shape == (n, 256)
    assert dtype == "float32"
    # Chunked: the encoder never sees more than `chunk` docs at once
    # (the RAM streaming boundary) ...
    assert calls and max(calls) <= 32
    assert sum(calls) == n
    # ... and the GPU batch is passed through separately (it is what the
    # model actually sees per forward pass).
    assert gpu_batches and all(b == 32 for b in gpu_batches)
    # The corpus is never materialised: the peak stays far below the
    # corpus size (the old code held every doc dict -> several x the
    # corpus in RAM).
    assert peak < corpus_bytes, f"peak={peak} corpus={corpus_bytes}"
    # The artifacts are the same as before.
    import numpy as np

    assert np.load(vpath).shape == (n, 256)
    assert json.loads(ipath.read_text()) == [d["id"] for d in docs]


def test_stage_queries_stops_after_nq(tmp_path, monkeypatch):
    """stage_queries needs only the first nq docs — it must not read them all."""
    import scripts.run_pipeline as pipe

    class _FakeEmbedder:
        """No model: the ground truth must be buildable in Tier-0."""

        def encode_queries(self, texts):
            return np.zeros((len(texts), 256), dtype=np.float32)

    n = 500
    docs = make_docs(n, seed=6)
    dpath = tmp_path / "docs.jsonl"
    dpath.write_text("\n".join(json.dumps(d) for d in docs) + "\n")

    parsed = []
    real_loads = json.loads

    def counting_loads(s, *a, **kw):
        parsed.append(s)
        return real_loads(s, *a, **kw)

    monkeypatch.setattr(pipe.json, "loads", counting_loads)

    # stage_queries now calls the ground-truth builder IN-PROCESS (no
    # subprocess), so the vectors it needs must exist for real.
    vpath = tmp_path / "vecs.npy"
    ipath = tmp_path / "ids.json"
    np.save(vpath, make_vectors(n, seed=6))
    ipath.write_text(json.dumps([d["id"] for d in docs]))

    qpath = tmp_path / "q.jsonl"
    gtpath = tmp_path / "gt.jsonl"
    pipe.stage_queries(
        str(dpath),
        str(vpath),
        str(ipath),
        5,
        queries_path=str(qpath),
        gt_path=str(gtpath),
        embedder=_FakeEmbedder(),
    )

    assert len(qpath.read_text().splitlines()) == 5
    # Only the first 5 DOCS were parsed — the other 495 were never read.
    # (The ground truth now runs in-process, so the query file and the
    # ids file are parsed too; count the document parses specifically.)
    doc_parses = [s for s in parsed if '"text"' in s]
    assert len(doc_parses) == 5
    assert all(f'"doc-{i}"' in s for i, s in enumerate(doc_parses))


def test_select_slice_sql_locked_content():
    """The researched SQL, verbatim: all WHERE gates, GROUP-BY density
    rank. No HTTPS glob: CloudFront 404s on directory globs, so the
    query reads an explicit per-file URL list (duck.py 'cloudfront'
    algo) built from crawl-data/<crawl>/cc-index-table.paths.gz."""
    from scripts.select_slice import CRAWL, PARTS_URL, render_sql, warc_to_wet

    assert CRAWL == "CC-MAIN-2026-34"
    assert PARTS_URL.format(crawl=CRAWL) == (
        "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-34/"
        "cc-index-table.paths.gz"
    )
    files = [
        (
            "https://data.commoncrawl.org/cc-index/table/cc-main/warc/"
            "crawl=CC-MAIN-2026-34/subset=warc/part-00000-x.parquet"
        )
    ]
    q = render_sql(files)
    assert "*.parquet" not in q  # explicit list, never a glob
    assert "part-00000-x.parquet" in q
    assert "subset = 'warc'" in q
    assert "fetch_status = 200" in q
    assert ("content_mime_detected IN ('text/html', 'application/xhtml+xml')") in q
    for like in (
        "content_languages = 'eng'",
        "LIKE 'eng,%'",
        "LIKE '%,eng'",
        "LIKE '%,eng,%'",
    ):
        assert like in q, like
    assert "GROUP BY warc_filename" in q
    assert "ORDER BY n_eng DESC" in q
    assert render_sql(files, limit_wet=130).strip().endswith("LIMIT 130")
    # /warc/ -> /wet/ sibling mapping; HTTPS prefix added by to_urls.
    assert (
        warc_to_wet("crawl-data/CC-MAIN-2026-34/segments/123/warc/f.warc.gz")
        == "crawl-data/CC-MAIN-2026-34/segments/123/wet/f.warc.gz"
    )
    from scripts.select_slice import to_urls

    assert to_urls([("crawl-data/C/segments/1/warc/f.warc.gz", 42)]) == [
        "https://data.commoncrawl.org/crawl-data/C/segments/1/wet/f.warc.gz"
    ]
    with pytest.raises(ValueError):
        warc_to_wet("crawl-data/CC-MAIN-2026-34/wet.paths.gz")


def test_select_slice_parts_filter_and_urls():
    """cc-index-table.paths.gz -> subset=warc parts only -> HTTPS URLs."""
    from scripts.select_slice import parts_to_urls

    paths = [
        "cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-34/",
        (
            "cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-34/"
            "subset=warc/part-00000-x.parquet"
        ),
        (
            "cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-34/"
            "subset=robotstxt/part-00000-y.parquet"
        ),
    ]
    urls = parts_to_urls(paths)
    assert urls == [
        (
            "https://data.commoncrawl.org/cc-index/table/cc-main/"
            "warc/crawl=CC-MAIN-2026-34/subset=warc/"
            "part-00000-x.parquet"
        )
    ]


def test_select_slice_max_parts_strided_sample():
    """--max-parts samples the part files with a stride (unbiased) instead
    of truncating: scanning half the corpus costs half the HTTP time."""
    from scripts.select_slice import sample_parts

    parts = [f"part-{i:05d}-x.parquet" for i in range(300)]
    assert len(sample_parts(parts, None)) == 300
    half = sample_parts(parts, 150)
    assert len(half) == 150
    # stride, not prefix: evenly spaced, first element included
    assert half[0] == parts[0]
    gaps = {
        parts.index(half[i + 1]) - parts.index(half[i]) for i in range(len(half) - 1)
    }
    assert gaps == {2}
    assert sample_parts(parts, 10) == parts[::30]
    assert sample_parts(parts, 300) == parts
    assert sample_parts(parts, 0) == parts  # 0 = no limit


def test_download_skips_complete_files(tmp_path, monkeypatch):
    """Skip-existing path needs no network once the size check is mocked."""
    import gzip

    import scripts.download_wet as dw

    payload = gzip.compress(b"data" * 100, mtime=0)
    f = tmp_path / "x.wet.gz"
    f.write_bytes(payload)
    monkeypatch.setattr(dw, "_remote_size", lambda url: len(payload))
    path, size = dw.fetch_one("https://example.com/x.wet.gz", str(tmp_path))
    assert (path, size) == (str(f), len(payload))
    assert f.read_bytes() == payload  # untouched (and gzip-verified)


def test_download_resumes_partial_file(tmp_path, monkeypatch):
    """Range-resume append logic with a stubbed transport (offline)."""
    import gzip
    import random
    import urllib.request

    import scripts.download_wet as dw

    # Incompressible payload so the gzip is bigger than the resume offset.
    rng = random.Random(0)
    full = gzip.compress(bytes(rng.getrandbits(8) for _ in range(4096)), mtime=0)
    assert len(full) > 800
    f = tmp_path / "y.wet.gz"
    f.write_bytes(full[:400])
    seen = {}

    class Resp:
        status = 206

        def __init__(self, payload):
            self._p = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            out, self._p = self._p[:n], self._p[n:]
            return out

    def fake_urlopen(req, timeout=None):
        seen["range"] = req.get_header("Range")
        assert seen["range"] == "bytes=400-"
        return Resp(full[400:])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    path, size = dw.fetch_one("https://example.com/y.wet.gz", str(tmp_path))
    assert size == len(full)
    assert Path(path).read_bytes() == full


@pytest.mark.parametrize(
    "script,flags",
    [
        ("select_slice.py", ("--crawl", "--lang", "--limit-wet", "--out")),
        ("download_wet.py", ("--urls", "--out", "--workers")),
        (
            "build_index.py",
            (
                "--vecs",
                "--ids",
                "--centroids",
                "--docs",
                "--out",
                "--store",
                "--no-store",
                "--filter",
                "--max-terms",
                "--crawl",
            ),
        ),
    ],
)
def test_script_cli_help(script, flags):
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    for flag in flags:
        assert flag in r.stdout, flag


@pytest.mark.cloud_only
def test_select_slice_live_toy_run(tmp_path):
    """Cloud box: real DuckDB-over-HTTPS query, 4 densest WET URLs."""
    from scripts.select_slice import main

    out = tmp_path / "wet_urls.txt"
    urls = main("CC-MAIN-2026-34", 4, str(out))
    assert len(urls) == 4
    assert all(u.startswith("https://data.commoncrawl.org/") for u in urls)
    assert all("/wet/" in u for u in urls)


def test_same_source_rejects_same_size_docs_with_different_bytes():
    """The index provenance must hash docs, not only size.

    A same-size docs edit used to pass `same_source`, so the index was
    reused for different text and stage_queries regenerated q/gt against
    the old index (the stale-artifact class C2/C3 exists to close).
    """
    from scripts.build_index import same_source

    base = {
        "docs_size": 100,
        "docs_sha256": "a" * 64,
        "vecs_size": 200,
        "vecs_sha256": "b" * 64,
        "ids_size": 50,
        "ids_sha256": "c" * 64,
    }
    assert same_source(base, dict(base))
    changed = dict(base, docs_sha256="d" * 64)
    assert not same_source(base, changed)


def test_build_filter_main_is_callable_without_argparse_globals(tmp_path):
    """C5: main() must take its inputs as parameters.

    It used to read module-global `a`, so importing and calling it (the
    pipeline path) raised NameError.
    """
    import scripts.build_filter as bf

    docs = tmp_path / "docs.jsonl"
    docs.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"d{i}",
                    "url": f"https://example.com/{i}",
                    "title": "t",
                    "text": "neural search and vector databases",
                    "date": "2026-01-01",
                }
            )
            for i in range(2)
        )
        + "\n"
    )
    out = tmp_path / "filter"
    bf.main(str(docs), str(out), 50_000)
    assert (out / "terms.bin").exists() and (out / "domains.json").exists()


def test_build_index_loads_vectors_with_mmap(tmp_path, monkeypatch):
    """C8: the 1M vector matrix must not be read fully into RAM."""
    import numpy as np

    from scripts.build_index import main as build

    _vecs, _ids, _docs, _cent, vpath, ipath, cpath, dpath = _write_inputs(
        tmp_path, n=8, k=4
    )
    real_load = np.load
    seen = []

    def recording_load(path, *args, **kwargs):
        seen.append((str(path), kwargs.get("mmap_mode")))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    build(str(vpath), str(ipath), str(cpath), str(dpath), str(tmp_path / "index"))
    assert (str(vpath), "r") in seen, seen


def test_no_store_with_a_custom_store_path_is_rejected(tmp_path):
    """Pruning a user-supplied --store under --no-store would be a surprise."""
    from scripts.build_index import main as build

    _vecs, _ids, _docs, _cent, vpath, ipath, cpath, dpath = _write_inputs(
        tmp_path, n=8, k=4
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        build(
            str(vpath),
            str(ipath),
            str(cpath),
            str(dpath),
            str(tmp_path / "index"),
            store=str(tmp_path / "custom-store"),
            no_store=True,
        )


def test_assign_cpu_matches_numpy():
    """The GPU path must be a drop-in for the CPU argmax (tie order aside)."""
    import numpy as np

    from scripts.build_index import assign

    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(50, 8)).astype(np.float32)
    cent = vecs[rng.choice(50, 4, replace=False)]
    got = assign(vecs, cent, device="cpu")
    want = np.argmax(vecs @ cent.T, axis=1)
    assert np.array_equal(got, want)


def test_assign_cuda_matches_numpy_when_available():
    import numpy as np
    import pytest

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from scripts.build_index import assign

    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(2048, 8)).astype(np.float32)
    cent = vecs[rng.choice(2048, 16, replace=False)]
    got = assign(vecs, cent, device="cuda")
    want = np.argmax(vecs @ cent.T, axis=1)
    assert np.array_equal(got, want)
