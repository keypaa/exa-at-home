# tests/fixtures/backends.py — the shared test doubles.
#
# These four fakes were copy-pasted into three test files (test_orch.py,
# test_mcp.py, test_mcp_logic.py). When the orchestrator's backend contract
# changed, three blocks had to change and forgetting one silently tested a
# different contract than production. One home, imported everywhere.
from __future__ import annotations


class FakeEmbedder:
    """Returns a 1-D "embedding" per text (length of the text)."""

    def __init__(self):
        self.calls: list[str] = []

    def encode_queries(self, texts):
        self.calls.extend(texts)
        return [[float(len(t))] for t in texts]


class FakeANN:
    """Mimics the ANN backend's search() shape: (ids, scores)."""

    def __init__(self, ids=("d0", "d1", "d2")):
        self.ids = list(ids)
        self.seen_kwargs: list[dict] = []

    def search(self, q, nprobe=8, top_k=200, **filters):
        self.seen_kwargs.append(dict(filters))
        ids = self.ids[:top_k]
        return ids, [1.0 - 0.1 * i for i in range(len(ids))]


class FakeReranker:
    """Success path sorts by coarse score and attaches a score; the
    degraded path omits the score (the real reranker's contract)."""

    def __init__(self, degraded=False):
        self.degraded = degraded

    def rerank(self, query, candidates):
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


def make_docs(n: int = 3) -> dict:
    """`n` fake documents with url/title/text."""
    return {
        f"d{i}": {
            "id": f"d{i}",
            "url": f"https://example.com/{i}",
            "title": f"Title {i}",
            "text": f"passage {i} about neural search " * 20,
        }
        for i in range(n)
    }


def wire_mcp(n: int = 3, degraded: bool = False):
    """Wire the MCP server with fakes; returns the FakeANN (so tests can
    assert on the filter kwargs it received)."""
    from exa_home.mcp_server import wire

    ann = FakeANN([f"d{i}" for i in range(n)])
    wire(
        FakeEmbedder(),
        ann,
        FakeReranker(degraded=degraded),
        FakeStore(make_docs(n)),
    )
    return ann
