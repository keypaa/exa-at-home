# tests/test_ingest.py
import pytest

# warcio is a declared dependency and exa_home.ingest imports it directly
# (brief mandates warcio-only, no vendored fallback in ingest.py). In this
# offline sandbox warcio is not installed, so skip with a clear message;
# on CI with warcio present the tests run for real.
pytest.importorskip("warcio", reason="warcio not installed; ingest.py is warcio-only per brief")

from fixtures.synth import make_docs, make_wet_gz
from exa_home.ingest import read_wet, write_docs_jsonl

def test_read_wet_yields_raw_docs(tmp_path):
    p = str(tmp_path / "t.wet.gz")
    make_wet_gz(p, make_docs(5))
    docs = list(read_wet(p))
    assert len(docs) == 5
    assert docs[0].url == "https://example.com/0"
    assert "synthetic document 0" in docs[0].text

def test_ingest_skips_empty_and_counts(tmp_path):
    # NOTE(deviation from brief): the brief's literal data cannot yield (3, 1)
    # against its own implementation — make_docs() texts are 43 chars, below
    # the 50-char too_short gate, and read_wet drops the whitespace-only doc
    # before write_docs_jsonl ever counts it. Preserving intent (kept + skip
    # counting) with long kept docs and one short-but-nonempty skipped doc.
    p = str(tmp_path / "t.wet.gz")
    docs = [
        {"id": f"d-{i}", "url": f"https://example.com/{i}", "title": f"Title {i}",
         "text": f"This is synthetic document {i} about topic {i % 7}, padded past fifty characters.",
         "date": "2026-01-01"}
        for i in range(3)
    ] + [{"id": "x", "url": "https://e.co/x", "title": "t", "text": "too short", "date": ""}]
    make_wet_gz(p, docs)
    out = str(tmp_path / "docs.jsonl")
    stats = write_docs_jsonl(read_wet(p), out)
    assert (stats.kept, stats.skipped) == (3, 1)
