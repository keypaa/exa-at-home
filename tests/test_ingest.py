# tests/test_ingest.py
import json

import pytest

# warcio is a declared dependency and exa_home.ingest imports it directly
# (brief mandates warcio-only, no vendored fallback in ingest.py). In this
# offline sandbox warcio is not installed, so skip with a clear message;
# on CI with warcio present the tests run for real.
pytest.importorskip(
    "warcio", reason="warcio not installed; ingest.py is warcio-only per brief"
)

from fixtures.synth import make_docs, make_wet_gz

from exa_home.ingest import RawDoc, read_wet, write_docs_jsonl


def test_read_wet_yields_raw_docs(tmp_path):
    p = str(tmp_path / "t.wet.gz")
    make_wet_gz(p, make_docs(5))
    docs = list(read_wet(p))
    assert len(docs) == 5
    assert docs[0].url == "https://example.com/0"
    assert "synthetic document 0" in docs[0].text


def _make_wet_raw(path: str, records: list[tuple[str, bytes]]) -> None:
    """WET with raw (possibly invalid UTF-8) payloads."""
    import gzip
    import io

    from warcio.warcwriter import WARCWriter

    with open(path, "wb") as f:
        for url, payload in records:
            with gzip.GzipFile(fileobj=f, mode="wb") as gz:
                writer = WARCWriter(gz, gzip=False)
                rec = writer.create_warc_record(
                    url,
                    "conversion",
                    payload=io.BytesIO(payload),
                    warc_content_type="text/plain",
                )
                writer.write_record(rec)


def test_undecodable_payload_is_counted_as_skipped(tmp_path):
    """BUG: a payload that failed UTF-8 decoding was dropped inside the
    generator, so it never reached the stats — the reported kept/skipped
    counts understated the loss."""
    p = str(tmp_path / "t.wet.gz")
    good = b"This is a synthetic document about topic 1, padded past fifty characters."
    bad = b"\xff\xfe\x00 not valid utf-8 \x80"
    _make_wet_raw(
        p,
        [
            ("https://example.com/0", good),
            ("https://example.com/bad", bad),
        ],
    )
    out = str(tmp_path / "docs.jsonl")
    stats = write_docs_jsonl(read_wet(p), out)
    assert stats.kept == 1
    assert stats.skipped == 1
    assert stats.reasons["undecodable"] == 1


def test_ingest_skips_empty_and_counts(tmp_path):
    # NOTE(deviation from brief): the brief's literal data cannot yield (3, 1)
    # against its own implementation — make_docs() texts are 43 chars, below
    # the 50-char too_short gate, and read_wet drops the whitespace-only doc
    # before write_docs_jsonl ever counts it. Preserving intent (kept + skip
    # counting) with long kept docs and one short-but-nonempty skipped doc.
    p = str(tmp_path / "t.wet.gz")
    docs = [
        {
            "id": f"d-{i}",
            "url": f"https://example.com/{i}",
            "title": f"Title {i}",
            "text": f"This is synthetic document {i} about topic {i % 7}, padded past fifty characters.",
            "date": "2026-01-01",
        }
        for i in range(3)
    ] + [
        {
            "id": "x",
            "url": "https://e.co/x",
            "title": "t",
            "text": "too short",
            "date": "",
        }
    ]
    make_wet_gz(p, docs)
    out = str(tmp_path / "docs.jsonl")
    stats = write_docs_jsonl(read_wet(p), out)
    assert (stats.kept, stats.skipped) == (3, 1)


def test_ingest_dedup_is_disk_backed_and_correct(tmp_path, monkeypatch):
    """60k records / 20k unique digests: dedup must stay exact without
    keeping every digest in a Python set (finding P2-05)."""
    import exa_home.ingest as ingest_mod

    opened: list[str] = []
    real_connect = ingest_mod.sqlite3.connect

    def record_connect(path, *a, **k):
        opened.append(str(path))
        return real_connect(path, *a, **k)

    monkeypatch.setattr(ingest_mod.sqlite3, "connect", record_connect)

    def docs():
        for rep in range(3):
            for i in range(20_000):
                digest = f"{i:040x}"
                yield RawDoc(
                    id=f"sha1:{digest}",
                    url=f"https://e.co/{i}",
                    title="t",
                    text=(
                        f"document {i} repetition {rep}, padded with enough "
                        "text to pass the fifty-character keep gate."
                    ),
                    date="",
                    digest=digest,
                )

    out = str(tmp_path / "docs.jsonl")
    stats = write_docs_jsonl(docs(), out, progress_every=0)
    assert stats.kept == 20_000
    assert stats.skipped == 40_000
    assert stats.reasons["duplicate"] == 40_000
    with open(out, encoding="utf-8") as f:
        assert len(f.read().splitlines()) == 20_000
    # The dedup structure really is a temp SQLite DB, not an in-RAM set.
    assert len(opened) == 1 and opened[0].endswith("seen.sqlite")
    assert not __import__("os").path.exists(opened[0]), "temp DB was not cleaned up"


def test_ingest_duplicate_short_doc_counts_as_duplicate(tmp_path):
    """Ordering preserved: the digest is recorded before the length gate,
    so a repeat of a too-short doc is reported as duplicate, as before."""
    doc = RawDoc(id="sha1:x", url="u", title="", text="short", date="", digest="x")
    out = str(tmp_path / "docs.jsonl")
    stats = write_docs_jsonl(iter([doc, doc]), out, progress_every=0)
    assert stats.kept == 0
    assert stats.reasons == {"too_short": 1, "duplicate": 1}


def test_stage_ingest_excludes_by_url_not_by_page_text(tmp_path):
    """The subset exclusion applies to the URL, never to the whole JSON line.

    `if rx.search(line): continue` matched the page TEXT: any document whose
    body merely mentioned robots.txt/robotstxt was silently dropped from the
    corpus, and the reported stats came from the pre-dedup pass, so the loss
    was invisible in the counts.
    """
    from scripts.run_pipeline import stage_ingest

    docs = [
        {
            "id": "keep-mentions",
            "url": "https://example.com/article",
            "title": "",
            # The BODY mentions robotstxt — the URL does not match.
            "text": (
                "This page explains that every site should publish a robotstxt "
                "file so crawlers know what to skip; here is the full story."
            ),
            "date": "",
        },
        {
            "id": "exclude-url",
            "url": "https://example.com/robotstxt-policy",
            "title": "",
            "text": "A perfectly ordinary page body that has nothing to exclude.",
            "date": "",
        },
        {
            "id": "exclude-crawldiagnostics",
            "url": "https://example.com/crawldiagnostics/42",
            "title": "",
            "text": "Another ordinary page body, but the URL is a crawl diagnostic.",
            "date": "",
        },
        {
            "id": "keep-plain",
            "url": "https://example.com/plain",
            "title": "",
            "text": "A plain page with no trigger words anywhere in the document.",
            "date": "",
        },
    ]
    wet_dir = tmp_path / "wet"
    wet_dir.mkdir()
    make_wet_gz(str(wet_dir / "seg-0.warc.wet.gz"), docs)

    out = tmp_path / "docs.jsonl"
    res = stage_ingest(str(wet_dir), str(out))

    kept = [json.loads(line) for line in out.read_text().splitlines()]
    assert [d["url"] for d in kept] == [
        "https://example.com/article",
        "https://example.com/plain",
    ]
    # The loss is visible in the reported stats.
    assert res["excluded"] == 2
    assert res["docs"] == 2


def test_title_from_url_percent_decodes_and_splits():
    from exa_home.ingest import _title_from_url

    assert _title_from_url("https://e.co/how-to-build-an-ann.html") == (
        "How To Build An Ann"
    )
    # %2F decodes to "/" and must split words, not leak into the title.
    assert _title_from_url("https://e.co/foo%2Fbar.html") == "Foo Bar"
    assert _title_from_url("https://e.co/") == ""
    # One very long unbroken segment is not a usable title.
    assert _title_from_url("https://e.co/" + "x" * 61 + ".html") == ""
