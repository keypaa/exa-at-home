# tests/test_scaffold.py — Task 1 scaffold gate (Tier 0 only).
import gzip

import numpy as np
import pytest

from fixtures.synth import make_vectors, make_docs, make_wet_gz


def _conversion_texts(path: str) -> list[bytes]:
    """Payloads of WARC-Type conversion records in a .wet.gz file.

    Uses real warcio when installed; otherwise a spec-faithful minimal
    WARC/1.0 parse (Content-Length framing) so Tier-0 stays green offline.
    """
    try:
        from warcio.archiveiterator import ArchiveIterator
    except ImportError:
        texts: list[bytes] = []
        with gzip.open(path, "rb") as gz:
            raw = gz.read()
        pos = 0
        while True:
            start = raw.find(b"WARC/1.0\r\n", pos)
            if start == -1:
                break
            hend = raw.find(b"\r\n\r\n", start)
            assert hend != -1, "truncated WARC headers"
            headers = raw[start:hend].decode("utf-8", errors="replace")
            is_conv = "WARC-Type: conversion" in headers
            clen = 0
            for line in headers.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    clen = int(line.split(":", 1)[1].strip())
            if is_conv:
                texts.append(raw[hend + 4:hend + 4 + clen])
            pos = hend + 4 + clen
        return texts
    with open(path, "rb") as f:
        return [r.content_stream().read() for r in ArchiveIterator(f)
                if r.rec_type == "conversion"]


def test_synth_vectors_are_normalized():
    v = make_vectors(10, 256, seed=1)
    assert v.shape == (10, 256)
    assert v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_synth_wet_roundtrip(tmp_path):
    p = str(tmp_path / "t.wet.gz")
    docs = make_docs(3)
    make_wet_gz(p, docs)
    recs = _conversion_texts(p)
    assert len(recs) == 3
    assert recs[0].decode("utf-8") == docs[0]["text"]


@pytest.mark.cloud_only
def test_scaffold_cloud_marker_placeholder():
    """Proves the cloud_only marker is registered; never runs in local CI."""
    assert True
