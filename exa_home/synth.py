# exa_home/synth.py — deterministic synthetic backends (seeded, no model, no network).
#
# Canonical home for Tier-0 construction: corpus builders, the numpy
# brute-force ANN stand-in, and a shared backend assembler. Scripts and
# notebooks import from HERE — never from tests/fixtures (prod code must
# not depend on test paths; tests/fixtures/synth.py is a thin re-export
# shim kept so existing test imports keep working).
import gzip
import io
import tempfile

import numpy as np


def make_vectors(n: int, dim: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v


def make_docs(n: int, seed: int = 0) -> list[dict]:
    return [
        {
            "id": f"doc-{i}",
            "url": f"https://example.com/{i}",
            "title": f"Title {i}",
            "text": f"This is synthetic document {i} about topic {i % 7}.",
            "date": "2026-01-01",
        }
        for i in range(n)
    ]


def make_wet_gz(path: str, docs: list[dict]) -> None:
    """Minimal real WET-format gzip: warcio can parse it back.

    Primary path uses ``warcio.warcwriter.WARCWriter`` exactly per the Task 1
    brief, which emits spec-compliant RFC 1123 ``WARC-Date`` values. warcio is
    a declared dependency (pyproject.toml) and is present in CI. If the import
    is unavailable (offline sandboxes with no PyPI egress), fall back to a
    hand-rolled WARC/1.0 writer whose byte layout matches warcio's
    (``gzip=False`` file-level gzip, CRLF headers, Content-Length framing) so
    the Tier-0 roundtrip test still exercises a parser offline. The function
    signature is identical either way.
    """
    try:
        from warcio.warcwriter import WARCWriter
    except ImportError:
        _make_wet_gz_fallback(path, docs)
        return

    # Multi-member gzip: one gzip member per record, matching real WET files.
    # warcio's ArchiveIterator strictly requires this layout and rejects a
    # single gzip member spanning multiple records.
    with open(path, "wb") as f:
        for d in docs:
            with gzip.GzipFile(fileobj=f, mode="wb") as gz:
                writer = WARCWriter(gz, gzip=False)
                payload = d["text"].encode("utf-8")
                rec = writer.create_warc_record(
                    d["url"],
                    "conversion",
                    payload=io.BytesIO(payload),
                    warc_content_type="text/plain",
                )
                writer.write_record(rec)


def _make_wet_gz_fallback(path: str, docs: list[dict]) -> None:
    """Offline WARC/1.0 writer mirroring WARCWriter's wire layout."""
    import hashlib
    import uuid
    from email.utils import formatdate

    with open(path, "wb") as f:
        for d in docs:
            with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
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


class NumpyFakeANN:
    """Tier-0 ANN stand-in: exact brute-force dot over doc vectors.

    Same injected-backend shape as the orch tests' FakeANN
    (`search(q, nprobe, n_coarse, **filters) -> (ids, scores)`), but scores are
    real numpy dots so the retrieve span is representative. Filters are
    accepted and ignored (documented): the fake corpus has no filter index.
    """

    def __init__(self, ids: list[str], mat):
        self.ids = ids
        self._mat = mat

    def search(self, q, nprobe=8, n_coarse=200, **filters):
        scores = self._mat @ np.asarray(q, dtype=np.float32)
        k = min(n_coarse, len(self.ids))
        idx = np.argsort(scores, kind="stable")[::-1][:k]
        return [self.ids[i] for i in idx], [float(scores[i]) for i in idx]


def build_synth_backends(n_docs: int = 200, seed: int = 0) -> dict:
    """Shared Tier-0 backend assembler (single construction site).

    Returns {"embedder", "ann", "reranker", "reranker_factory", "store"}.
    Both `e2e_latency.build_synth_dag` and the sweep suite delegate here —
    the wiring exists exactly once.
    """
    from exa_home.embed import HashEmbedder
    from exa_home.rerank import MockReranker
    from exa_home.store import ContentStore

    docs = make_docs(n_docs, seed=seed)
    store = ContentStore(tempfile.mkdtemp(prefix="synth-store-"))
    for d in docs:
        store.put(d)
    embedder = HashEmbedder(dim=256)
    mat = embedder.encode_docs([d["text"] for d in docs])
    ann = NumpyFakeANN([d["id"] for d in docs], mat)
    return {
        "embedder": embedder,
        "ann": ann,
        "reranker": MockReranker(),
        "reranker_factory": lambda pt, b: MockReranker(),
        "store": store,
    }
