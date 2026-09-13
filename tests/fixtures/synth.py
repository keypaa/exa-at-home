# tests/fixtures/synth.py — Tier-0 synthetic fixtures (seeded, no model, no network).
import gzip
import io

import numpy as np


def make_vectors(n: int, dim: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v


def make_docs(n: int, seed: int = 0) -> list[dict]:
    return [
        {"id": f"doc-{i}", "url": f"https://example.com/{i}",
         "title": f"Title {i}", "text": f"This is synthetic document {i} about topic {i % 7}.",
         "date": "2026-01-01"}
        for i in range(n)
    ]


def make_wet_gz(path: str, docs: list[dict]) -> None:
    """Minimal real WET-format gzip: warcio can parse it back.

    Primary path uses ``warcio.warcwriter.WARCWriter`` exactly per the Task 1
    brief, which emits spec-compliant RFC 1123 ``WARC-Date`` values. warcio is
    a declared dependency (pyproject.toml) and is present in CI. If the import
    is unavailable (e.g. this sandbox, where PyPI egress is denied), fall back
    to a hand-rolled WARC/1.0 writer whose byte layout matches warcio's
    (``gzip=False`` file-level gzip, CRLF headers, Content-Length framing) so
    the Tier-0 roundtrip test still exercises a parser in offline sandboxes.
    The function signature is identical either way.
    """
    try:
        from warcio.warcwriter import WARCWriter
    except ImportError:
        _make_wet_gz_fallback(path, docs)
        return

    with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb") as gz:
        writer = WARCWriter(gz, gzip=False)
        for d in docs:
            payload = d["text"].encode("utf-8")
            rec = writer.create_warc_record(
                d["url"], "conversion",
                payload=io.BytesIO(payload),
                warc_content_type="text/plain")
            writer.write_record(rec)


def _make_wet_gz_fallback(path: str, docs: list[dict]) -> None:
    """Offline WARC/1.0 writer mirroring WARCWriter's wire layout."""
    import hashlib
    import uuid
    from email.utils import formatdate

    with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
        for d in docs:
            payload = d["text"].encode("utf-8")
            digest = hashlib.sha1(payload).hexdigest()
            header = (
                "WARC/1.0\r\n"
                "WARC-Type: conversion\r\n"
                f"WARC-Target-URI: {d['url']}\r\n"
                f"WARC-Date: {formatdate(usegmt=True)}\r\n"
                f"WARC-Record-ID: <urn:uuid:{uuid.uuid4()}>\r\n"
                "Content-Type: text/plain\r\n"
                f"WARC-Block-Digest: sha1:{digest}\r\n"
                f"Content-Length: {len(payload)}\r\n"
                "\r\n"
            )
            gz.write(header.encode("utf-8") + payload + b"\r\n\r\n")
