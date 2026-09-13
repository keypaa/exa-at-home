# tests/fixtures/synth.py — Tier-0 synthetic fixtures (seeded, no model, no network).
import gzip
import hashlib
import io  # noqa: F401 (kept for API symmetry with warcio-based writers)
import uuid
from datetime import datetime, timezone

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

    Writes spec-compliant WARC/1.0 ``conversion`` records by hand (no warcio
    import needed) so Tier-0 stays dependency-light; the byte layout is what
    ``warcio.warcwriter.WARCWriter`` itself emits (file-level gzip, CRLF
    headers, Content-Length framing), so ``warcio.archiveiterator`` parses it
    wherever warcio is installed (cloud box / Task 2+).
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
        for d in docs:
            payload = d["text"].encode("utf-8")
            digest = hashlib.sha1(payload).hexdigest()
            header = (
                "WARC/1.0\r\n"
                "WARC-Type: conversion\r\n"
                f"WARC-Target-URI: {d['url']}\r\n"
                f"WARC-Date: {now}\r\n"
                f"WARC-Record-ID: <urn:uuid:{uuid.uuid4()}>\r\n"
                "Content-Type: text/plain\r\n"
                f"WARC-Block-Digest: sha1:{digest}\r\n"
                f"Content-Length: {len(payload)}\r\n"
                "\r\n"
            )
            gz.write(header.encode("utf-8") + payload + b"\r\n\r\n")
