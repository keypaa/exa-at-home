# tests/fixtures/backends.py — the shared test doubles.
#
# These fakes used to be copy-pasted into three test files (test_orch.py,
# test_mcp.py, test_mcp_logic.py) and then drift from the production
# contracts: FakeEmbedder returned 1-D "length of text" vectors, FakeANN
# returned fixed descending scores, and no fake modelled degraded rerank.
# A green Tier-0 suite could therefore assert orderings and shapes no real
# backend would ever produce (finding T3/P2-14).
#
# Parity rules enforced by tests/test_synth_parity.py:
#   FakeEmbedder  -> 256-D normalized float32 vectors, like Embedder.
#   FakeANN       -> the Rust `binary_dot_packed` score (sign-bit vs query
#                    sign), best-first, truncated to n_coarse, and the Rust
#                    filter semantics (domains OR / terms AND / month range
#                    inclusive / allow row ids).
#   FakeReranker  -> degraded flag semantics identical to Reranker.rerank.
from __future__ import annotations

import hashlib

import numpy as np

# Prod helpers the fakes must NOT re-implement (parity by construction).
from scripts.build_filter import _month_key, domain_of, tokenize

DIM = 256  # exa_home.config.DIM — kept local so fixtures stay import-light
DEFAULT_IDS = ("d0", "d1", "d2")


def _stable_vector(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic, normalized 256-D vector for `text` (hash-seeded)."""
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    )
    v = rng.normal(size=dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


class FakeEmbedder:
    """Like Embedder.encode_queries: returns (n, 256) float32 vectors.

    The old 1-D "length of text" vector had nothing in common with the real
    model, so downstream scoring was exercised at a shape the server never
    sees. Deterministic per text (hash-seeded), so tests stay reproducible.
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.calls: list[str] = []

    def encode_queries(self, texts):
        self.calls.extend(texts)
        return (
            np.vstack([_stable_vector(t, self.dim) for t in texts])
            if texts
            else (np.zeros((0, self.dim), dtype=np.float32))
        )


class FakeANN:
    """Brute-force stand-in for IvfIndex with its real scoring + filters.

    - score(row, q) = Σ over the 256 code bits: +q[i] where the build-time
      sign bit was set, -q[i] where it was not. That is exactly the Rust
      `quant::binary_dot_packed` / `lut::score` contract (and the numpy
      reference in tests/test_ann.py).
    - results are best-first and truncated to ``n_coarse``.
    - filters use the same semantics as crates/ann-core/src/filter.rs:
      domains OR (hostname, www. stripped), terms AND (lowercase tokens),
      month_range inclusive over YYYY-MM buckets (docs without a month are
      excluded), allow is a row-id whitelist.
    """

    def __init__(self, ids=DEFAULT_IDS, vecs=None, docs=None):
        self.ids = list(ids)
        mat = (
            np.asarray(vecs, dtype=np.float32)
            if vecs is not None
            else np.vstack([_stable_vector(i) for i in self.ids])
        )
        assert mat.shape[0] == len(self.ids), "one vector per id"
        self.vecs = mat
        self.bits = mat > 0  # the build-time code the index would store
        self.docs = docs or {}  # id -> doc dict (url/date/text), for filters
        self.seen_kwargs: list[dict] = []

    def _rows_after_filters(self, **filters) -> np.ndarray:
        keep = np.ones(len(self.ids), dtype=bool)
        if "allow" in filters and filters["allow"] is not None:
            allowed = np.zeros(len(self.ids), dtype=bool)
            for row in filters["allow"]:
                if 0 <= row < len(self.ids):
                    allowed[row] = True
            keep &= allowed
        domains = filters.get("domains")
        if domains:
            want = {d.lower() for d in domains}
            keep &= np.array(
                [
                    domain_of((self.docs.get(doc_id) or {}).get("url", "")) in want
                    for doc_id in self.ids
                ]
            )
        terms = filters.get("terms")
        if terms:
            want_terms = [t.lower() for t in terms]
            keep &= np.array(
                [
                    all(
                        t in tokenize((self.docs.get(doc_id) or {}).get("text", ""))
                        for t in want_terms
                    )
                    for doc_id in self.ids
                ]
            )
        mr = filters.get("month_range")
        if mr:
            lo, hi = mr

            def in_range(doc_id: str) -> bool:
                month = _month_key((self.docs.get(doc_id) or {}).get("date", ""))
                return bool(month) and lo <= month <= hi

            keep &= np.array([in_range(doc_id) for doc_id in self.ids])

        return np.flatnonzero(keep)

    def search(self, q, nprobe=8, n_coarse=200, **filters):
        self.seen_kwargs.append(dict(filters))
        rows = self._rows_after_filters(**filters)
        q = np.asarray(q, dtype=np.float32).ravel()
        # score = Σ where(bit, q, -q) — the binary dot product
        contrib = np.where(self.bits[rows], q, -q)
        scores = contrib.sum(axis=1) if len(rows) else np.zeros(0)
        order = np.argsort(-scores, kind="stable")[:n_coarse]
        picked = rows[order]
        return [self.ids[i] for i in picked], [float(s) for s in scores[order]]


class FakeReranker:
    """Success path sorts by a scored order and attaches a score; the
    degraded path returns the input order and NO score, exactly like the
    real reranker (orch falls back to the coarse score in that case)."""

    def __init__(self, degraded=False):
        self.degraded = degraded
        self.calls: list[str] = []

    def rerank(self, query, candidates):
        self.calls.append(query)
        if self.degraded:
            return list(candidates), True
        ranked = sorted(candidates, key=lambda c: c["coarse_score"], reverse=True)
        return [{**c, "score": c["coarse_score"] + 0.01} for c in ranked], False


class FakeStore:
    """In-memory content store."""

    def __init__(self, docs):
        self.docs = docs

    def get(self, doc_id):
        return self.docs.get(doc_id)

    def get_many(self, ids):
        return {i: self.docs.get(i) for i in ids}


def make_docs(n: int = 3) -> dict:
    """`n` fake documents with url/title/text (and a date, filtering-ready)."""
    return {
        f"d{i}": {
            "id": f"d{i}",
            "url": f"https://example.com/{i}",
            "title": f"Title {i}",
            "text": f"passage {i} about neural search " * 20,
            "date": "2026-01-01",
        }
        for i in range(n)
    }


def wire_mcp(n: int = 3, degraded: bool = False, docs: dict | None = None):
    """Wire the MCP server with fakes; returns the FakeANN (so tests can
    assert on the filter kwargs it received)."""
    from exa_home.mcp_server import wire

    ann = FakeANN([f"d{i}" for i in range(n)], docs=docs)
    wire(
        FakeEmbedder(),
        ann,
        FakeReranker(degraded=degraded),
        FakeStore(docs if docs is not None else make_docs(n)),
    )
    return ann
