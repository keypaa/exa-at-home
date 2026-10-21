# tests/test_embed.py
import re

import numpy as np
import pytest

from exa_home.embed import Embedder, HashEmbedder, length_sorted_order


def test_hash_embedder_shape_and_deterministic():
    e = HashEmbedder(dim=256)
    a = e.encode_queries(["hello world"])
    b = e.encode_queries(["hello world"])
    assert a.shape == (1, 256) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert not np.array_equal(a, e.encode_queries(["goodbye"]))


def test_embedder_defaults_to_the_configured_model():
    """The model id lives in exa_home.config — one place, not four."""
    from exa_home.config import EMBEDDER_MODEL

    assert EMBEDDER_MODEL == "mixedbread-ai/mxbai-embed-large-v1"


def test_length_sorted_order_is_stable_and_ascending():
    """Length bucketing exists to cut padding waste; ties must keep their
    input order (stable) so the restored order is exact."""
    order = length_sorted_order([5, 1, 1, 9, 3])
    assert list(order) == [1, 2, 4, 0, 3]
    lengths = [5, 1, 1, 9, 3]
    assert [lengths[i] for i in order] == [1, 1, 3, 5, 9]


class _StubTokenizer:
    def __call__(self, texts, **kwargs):
        return {"length": [len(t) // 10 for t in texts]}


class _StubModel:
    """Minimal SentenceTransformer stand-in.

    Records the batches it saw and marks each vector with the ORIGINAL
    document index parsed from the text, so the test can prove the output
    order survived the internal sort.
    """

    def __init__(self):
        self.tokenizer = _StubTokenizer()
        self.seen_batches = []
        self.seen_texts = []

    def encode(self, texts, **kwargs):
        self.seen_batches.append((len(texts), kwargs.get("batch_size")))
        self.seen_texts.extend(texts)
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            m = re.search(r"doc-(\d+)", t)
            # The marker lives in a RATIO: the implementation normalizes
            # every vector, so only a direction survives normalization.
            out[i, 0] = float(m.group(1)) if m else 0.0
            out[i, 1] = 1.0
        return out


def _stub_embedder():
    e = Embedder.__new__(Embedder)
    e.dim = 256
    e.model = _StubModel()
    e.batch_size = 1024
    e.precision = "float32"
    e.max_seq_length = 512
    return e


def test_encode_docs_restores_input_order_after_bucketing():
    """The whole point of sorting by length is that the OUTPUT must still
    be in input order — a silent reorder would corrupt every vector."""
    e = _stub_embedder()
    texts = [f"doc-{i}" * (10 - i) for i in range(6)]
    v = e.encode_docs(texts)
    assert v.shape == (6, 256) and v.dtype == np.float32
    # The stub saw the texts grouped by non-decreasing TOKEN length ...
    token_lens = [len(t) // 10 for t in e.model.seen_texts]
    assert token_lens == sorted(token_lens)
    # ... and the returned vectors still line up with the INPUT order
    # (marker read as a ratio: normalization preserves direction only).
    assert list(v[:, 0] / v[:, 1]) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_encode_docs_uses_the_configured_batch_size():
    e = _stub_embedder()
    e.encode_docs([f"doc-{i}" for i in range(10)])
    assert e.model.seen_batches[0][1] == 1024


def test_encode_queries_does_not_bucket_single_queries():
    """The serving path is one query at a time: bucketing would only add
    latency."""
    e = _stub_embedder()
    v = e.encode_queries(["a query"])
    assert v.shape == (1, 256)
    assert e.model.seen_batches[0][0] == 1


@pytest.mark.cloud_only
def test_real_embedder_truncates_to_256():
    e = Embedder()
    v = e.encode_queries(["what is a vector database?"])
    assert v.shape == (1, 256) and v.dtype == np.float32
    assert abs(np.linalg.norm(v[0]) - 1.0) < 1e-3
