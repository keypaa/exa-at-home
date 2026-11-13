# tests/test_pipeline_provenance.py — C1/C2/C3 provenance gates.
#
# Each stage used to reuse an artifact merely because the file existed, so
# files from a different crawl were consumed silently. These tests pin the
# input-identity checks and the sidecars they read.
import gzip
import json

import numpy as np
import pytest
from fixtures.synth import make_docs, make_vectors, make_wet_gz

import scripts.run_pipeline as pipe


def _long_docs(start: int, n: int) -> list[dict]:
    return [
        {
            "id": f"d{start + i}",
            "url": f"https://example.com/{start + i}",
            "title": "",
            "text": (
                f"Document {start + i} has enough body text to pass the "
                "fifty-character ingest gate used by the pipeline."
            ),
            "date": "",
        }
        for i in range(n)
    ]


def test_stage_download_rejects_foreign_file_set(tmp_path, monkeypatch):
    """Two files from a DIFFERENT crawl must not satisfy a 2-URL request."""
    urls = tmp_path / "wet_urls.txt"
    urls.write_text(
        "https://data.commoncrawl.org/crawl/a.wet.gz\n"
        "https://data.commoncrawl.org/crawl/b.wet.gz\n"
    )
    out = tmp_path / "wet"
    out.mkdir()
    # Same COUNT as requested, wrong NAMES (the old `have >= len(urls)` skip).
    (out / "b.wet.gz").write_bytes(gzip.compress(b"stale-b"))
    (out / "c.wet.gz").write_bytes(gzip.compress(b"stale-c"))

    calls = []

    def fake_download(urls_path, out_dir, workers):
        calls.append((urls_path, out_dir, workers))
        for name in ("a.wet.gz", "b.wet.gz"):
            (tmp_path / "wet" / name).write_bytes(
                gzip.compress(b"fresh-" + name.encode())
            )

    monkeypatch.setattr("scripts.download_wet.main", fake_download)

    pipe.stage_download(str(urls), str(out), workers=2)

    assert calls, "foreign file set was reused instead of re-downloading"
    assert not (out / "c.wet.gz").exists(), "foreign WET file survived"
    assert gzip.decompress((out / "a.wet.gz").read_bytes()) == b"fresh-a.wet.gz"

    # Now the directory matches the URL set: a second call reuses it.
    calls.clear()
    pipe.stage_download(str(urls), str(out), workers=2)
    assert calls == []


def test_stage_ingest_reingests_when_wet_set_changes(tmp_path):
    wet = tmp_path / "wet"
    wet.mkdir()
    make_wet_gz(str(wet / "seg-0.warc.wet.gz"), _long_docs(0, 3))
    docs = tmp_path / "docs.jsonl"

    first = pipe.stage_ingest(str(wet), str(docs), verbose=False, progress_every=0)
    assert first["docs"] == 3
    assert (tmp_path / "docs.jsonl.provenance.json").is_file()

    # Same inputs: reuse without re-reading the WET files.
    second = pipe.stage_ingest(str(wet), str(docs), verbose=False, progress_every=0)
    assert second == {"docs": 3, "reused": True}

    # A new WET segment makes docs.jsonl stale: it must be rebuilt.
    make_wet_gz(str(wet / "seg-1.warc.wet.gz"), _long_docs(3, 2))
    third = pipe.stage_ingest(str(wet), str(docs), verbose=False, progress_every=0)
    assert third.get("reused") is not True
    assert third["docs"] == 5

    fourth = pipe.stage_ingest(str(wet), str(docs), verbose=False, progress_every=0)
    assert fourth == {"docs": 5, "reused": True}


def test_stage_ingest_reingests_when_docs_tampered(tmp_path):
    wet = tmp_path / "wet"
    wet.mkdir()
    make_wet_gz(str(wet / "seg.warc.wet.gz"), _long_docs(0, 2))
    docs = tmp_path / "docs.jsonl"
    pipe.stage_ingest(str(wet), str(docs), verbose=False, progress_every=0)

    with open(docs, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "injected", "url": "u", "text": "x"}) + "\n")

    res = pipe.stage_ingest(str(wet), str(docs), verbose=False, progress_every=0)
    assert res.get("reused") is not True
    assert res["docs"] == 2  # injected line was replaced by a real re-ingest


def _queries_fixture(tmp_path, docs, vecs, ids):
    dpath = tmp_path / "docs.jsonl"
    dpath.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    vpath = tmp_path / "vecs.npy"
    np.save(vpath, vecs)
    ipath = tmp_path / "ids.json"
    ipath.write_text(json.dumps(ids))
    return dpath, vpath, ipath


class _ZeroEmbedder:
    def encode_queries(self, texts):
        return np.zeros((len(texts), 256), dtype=np.float32)


def test_stage_queries_regenerates_on_corpus_change(tmp_path):
    n = 30
    docs = make_docs(n, seed=11)
    vecs = make_vectors(n, seed=11)
    dpath, vpath, ipath = _queries_fixture(
        tmp_path, docs, vecs, [d["id"] for d in docs]
    )
    qpath = tmp_path / "q.jsonl"
    gtpath = tmp_path / "gt.jsonl"

    kwargs = {
        "queries_path": str(qpath),
        "gt_path": str(gtpath),
        "embedder": _ZeroEmbedder(),
    }
    pipe.stage_queries(str(dpath), str(vpath), str(ipath), 5, **kwargs)
    assert len(qpath.read_text().splitlines()) == 5
    assert (tmp_path / "q.jsonl.provenance.json").is_file()

    again = pipe.stage_queries(str(dpath), str(vpath), str(ipath), 5, **kwargs)
    assert again == {"reused": True}

    # Rebuild the corpus (same paths, new content): stale q/gt must go.
    # make_docs() ignores its seed, so change the text explicitly.
    new_docs = make_docs(n, seed=11)
    for i, d in enumerate(new_docs):
        d["text"] = f"rebuilt corpus document {i} with completely new text"
    dpath.write_text("\n".join(json.dumps(d) for d in new_docs) + "\n")
    res = pipe.stage_queries(str(dpath), str(vpath), str(ipath), 5, **kwargs)
    assert res.get("reused") is not True
    first_query = json.loads(qpath.read_text().splitlines()[0])["query"]
    assert first_query == new_docs[0]["text"][:200]

    # A vector rebuild with the same docs also invalidates the ground truth.
    np.save(vpath, make_vectors(n, seed=123))
    res = pipe.stage_queries(str(dpath), str(vpath), str(ipath), 5, **kwargs)
    assert res.get("reused") is not True


def test_stage_queries_nq_change_regenerates(tmp_path):
    n = 20
    docs = make_docs(n, seed=5)
    vecs = make_vectors(n, seed=5)
    dpath, vpath, ipath = _queries_fixture(
        tmp_path, docs, vecs, [d["id"] for d in docs]
    )
    qpath = tmp_path / "q.jsonl"
    gtpath = tmp_path / "gt.jsonl"
    kwargs = {
        "queries_path": str(qpath),
        "gt_path": str(gtpath),
        "embedder": _ZeroEmbedder(),
    }
    pipe.stage_queries(str(dpath), str(vpath), str(ipath), 3, **kwargs)
    assert len(qpath.read_text().splitlines()) == 3
    res = pipe.stage_queries(str(dpath), str(vpath), str(ipath), 4, **kwargs)
    assert res.get("reused") is not True
    assert len(qpath.read_text().splitlines()) == 4


@pytest.mark.parametrize("missing", ["q", "gt"])
def test_stage_queries_regenerates_when_one_artifact_missing(tmp_path, missing):
    n = 10
    docs = make_docs(n, seed=7)
    vecs = make_vectors(n, seed=7)
    dpath, vpath, ipath = _queries_fixture(
        tmp_path, docs, vecs, [d["id"] for d in docs]
    )
    qpath = tmp_path / "q.jsonl"
    gtpath = tmp_path / "gt.jsonl"
    kwargs = {
        "queries_path": str(qpath),
        "gt_path": str(gtpath),
        "embedder": _ZeroEmbedder(),
    }
    pipe.stage_queries(str(dpath), str(vpath), str(ipath), 2, **kwargs)
    (qpath if missing == "q" else gtpath).unlink()
    res = pipe.stage_queries(str(dpath), str(vpath), str(ipath), 2, **kwargs)
    assert res.get("reused") is not True
    assert qpath.is_file() and gtpath.is_file()


def test_stage_download_rejects_corrupt_but_nonempty_file(tmp_path, monkeypatch):
    """A truncated gzip must be re-downloaded, not reused (C9 bypass)."""
    urls = tmp_path / "wet_urls.txt"
    urls.write_text("https://data.commoncrawl.org/crawl/a.wet.gz\n")
    out = tmp_path / "wet"
    out.mkdir()
    (out / "a.wet.gz").write_bytes(gzip.compress(b"x" * 100)[:20])  # truncated

    calls = []

    def fake_download(urls_path, out_dir, workers):
        calls.append(urls_path)
        (tmp_path / "wet" / "a.wet.gz").write_bytes(gzip.compress(b"fresh"))

    monkeypatch.setattr("scripts.download_wet.main", fake_download)
    pipe.stage_download(str(urls), str(out), workers=1)
    assert calls, "corrupt non-empty file was reused"
    assert gzip.decompress((out / "a.wet.gz").read_bytes()) == b"fresh"


def test_stage_download_force_prunes_foreign_files(tmp_path, monkeypatch):
    urls = tmp_path / "wet_urls.txt"
    urls.write_text("https://data.commoncrawl.org/crawl/a.wet.gz\n")
    out = tmp_path / "wet"
    out.mkdir()
    (out / "foreign.wet.gz").write_bytes(gzip.compress(b"other"))
    (out / "a.wet.gz").write_bytes(gzip.compress(b"old"))

    def fake_download(urls_path, out_dir, workers):
        (tmp_path / "wet" / "a.wet.gz").write_bytes(gzip.compress(b"new"))

    monkeypatch.setattr("scripts.download_wet.main", fake_download)
    pipe.stage_download(str(urls), str(out), workers=1, force=True)
    assert not (out / "foreign.wet.gz").exists(), "force skipped the prune"


def test_wet_fingerprint_detects_same_size_rewrite(tmp_path):
    """mtime+size is not provenance: a same-size rewrite must invalidate."""
    import os

    wet = tmp_path / "wet"
    wet.mkdir()
    path = wet / "seg.wet.gz"
    path.write_bytes(b"AAAA")
    first = pipe._wet_fingerprint(str(wet))
    st = path.stat()
    path.write_bytes(b"BBBB")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert pipe._wet_fingerprint(str(wet)) != first


def test_stage_ingest_refuses_empty_wet_and_keeps_docs(tmp_path):
    wet = tmp_path / "wet"
    wet.mkdir()
    make_wet_gz(str(wet / "seg.wet.gz"), _long_docs(0, 2))
    docs = tmp_path / "docs.jsonl"
    pipe.stage_ingest(str(wet), str(docs), verbose=False, progress_every=0)
    original = docs.read_bytes()
    assert original

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no WET files"):
        pipe.stage_ingest(str(empty), str(docs), verbose=False, progress_every=0)
    assert docs.read_bytes() == original, "empty WET dir destroyed the corpus"


def test_stage_index_trains_centroids_inside_the_index_dir(tmp_path, monkeypatch):
    """C15: centroids.npy must land under <out>, not the process cwd."""
    import numpy as np

    vecs = tmp_path / "vecs.npy"
    ids = tmp_path / "ids.json"
    docs = tmp_path / "docs.jsonl"
    np.save(vecs, np.zeros((2, 256), dtype=np.float32))
    ids.write_text(json.dumps(["a", "b"]))
    docs.write_text(
        "\n".join(
            json.dumps({"id": d, "text": "body " + "x" * 100}) for d in ("a", "b")
        )
        + "\n"
    )
    out = tmp_path / "index"
    seen = {}

    def fake_train(vecs_path, k, sample, out_path, verbose=False):
        seen["centroids"] = out_path
        np.save(out_path, np.zeros((k, 256), dtype=np.float32))

    def fake_build(vecs_path, ids_path, cent_path, docs_path, out_path, **kwargs):
        seen["build_centroids"] = cent_path

    monkeypatch.setattr("scripts.train_centroids.main", fake_train)
    monkeypatch.setattr("scripts.build_index.main", fake_build)
    monkeypatch.setattr(
        "exa_home.index_format.verify_manifest",
        lambda _out: {"n_docs": 2, "n_centroids": 2},
    )

    pipe.stage_index(str(vecs), str(ids), str(docs), str(out), k=2, sample=2)
    assert seen["centroids"] == str(out / "centroids.npy")
    assert seen["build_centroids"] == seen["centroids"]


def test_stage_fetch_retries_transient_http_errors(tmp_path, monkeypatch):
    """A 503 on the small wet.paths.gz index must not abort a 2.5 h run."""
    import io
    import urllib.error
    import urllib.request

    payload = gzip.compress(
        b"CC-MAIN-2026-34/wet/one.warc.wet.gz\nCC-MAIN-2026-34/wet/two.warc.wet.gz\n"
    )
    calls = {"n": 0}

    class Resp:
        def __init__(self, raw):
            self.raw = raw

        def read(self):
            return self.raw

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:
            url = getattr(req, "full_url", req)
            raise urllib.error.HTTPError(url, 503, "busy", {}, io.BytesIO(b""))
        return Resp(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(pipe.time, "sleep", lambda _s: None)

    out = tmp_path / "wet_urls.txt"
    urls = pipe.stage_fetch("CC-MAIN-2026-34", 2, str(out))
    assert calls["n"] == 3
    assert len(urls) == 2
    assert all(u.startswith("https://data.commoncrawl.org/") for u in urls)
    assert out.read_text().splitlines() == urls
