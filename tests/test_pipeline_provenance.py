# tests/test_pipeline_provenance.py — C1/C2/C3 provenance gates.
#
# Each stage used to reuse an artifact merely because the file existed, so
# files from a different crawl were consumed silently. These tests pin the
# input-identity checks and the sidecars they read.
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
    (out / "b.wet.gz").write_bytes(b"stale-b")
    (out / "c.wet.gz").write_bytes(b"stale-c")

    calls = []

    def fake_download(urls_path, out_dir, workers):
        calls.append((urls_path, out_dir, workers))
        for name in ("a.wet.gz", "b.wet.gz"):
            (tmp_path / "wet" / name).write_bytes(b"fresh-" + name.encode())

    monkeypatch.setattr("scripts.download_wet.main", fake_download)

    pipe.stage_download(str(urls), str(out), workers=2)

    assert calls, "foreign file set was reused instead of re-downloading"
    assert not (out / "c.wet.gz").exists(), "foreign WET file survived"
    assert (out / "a.wet.gz").read_bytes() == b"fresh-a.wet.gz"

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
