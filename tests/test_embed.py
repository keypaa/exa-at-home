# tests/test_embed.py
import numpy as np
import pytest

from exa_home.embed import Embedder, HashEmbedder


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


@pytest.mark.cloud_only
def test_real_embedder_truncates_to_256():
    e = Embedder()
    v = e.encode_queries(["what is a vector database?"])
    assert v.shape == (1, 256) and v.dtype == np.float32
    assert abs(np.linalg.norm(v[0]) - 1.0) < 1e-3
