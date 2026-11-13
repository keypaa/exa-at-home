# tests/test_ground_truth.py — C7: random-vector --nq path warns.
import json

import numpy as np

from scripts.ground_truth import main


def test_nq_without_queries_warns_about_unscorable_ground_truth(tmp_path, capsys):
    vecs = tmp_path / "vecs.npy"
    ids = tmp_path / "ids.json"
    np.save(vecs, np.random.default_rng(0).normal(size=(5, 256)).astype(np.float32))
    ids.write_text(json.dumps([f"d{i}" for i in range(5)]))
    out = tmp_path / "gt.jsonl"

    assert (
        main(["--vecs", str(vecs), "--ids", str(ids), "--nq", "2", "--out", str(out)])
        == 0
    )
    err = capsys.readouterr().err
    assert "WARNING" in err and "--queries" in err and "~0.0000" in err
    assert len(out.read_text().splitlines()) == 2


def test_queries_path_does_not_warn(tmp_path, capsys, monkeypatch):
    """The aligned text-query path is the supported one: no warning."""
    vecs = tmp_path / "vecs.npy"
    ids = tmp_path / "ids.json"
    queries = tmp_path / "q.jsonl"
    np.save(vecs, np.random.default_rng(0).normal(size=(5, 256)).astype(np.float32))
    ids.write_text(json.dumps([f"d{i}" for i in range(5)]))
    queries.write_text(json.dumps({"query": "neural search"}) + "\n")

    class FakeEmbedder:
        def encode_queries(self, texts):
            return np.zeros((len(texts), 256), dtype=np.float32)

    import scripts.ground_truth as gt

    monkeypatch.setattr(gt, "Embedder", FakeEmbedder, raising=False)
    # ground_truth imports Embedder lazily inside the function, so patch the
    # source module instead.
    import exa_home.embed

    monkeypatch.setattr(exa_home.embed, "Embedder", FakeEmbedder)

    assert (
        main(
            [
                "--vecs",
                str(vecs),
                "--ids",
                str(ids),
                "--out",
                str(tmp_path / "gt.jsonl"),
                "--queries",
                str(queries),
            ]
        )
        == 0
    )
    assert "WARNING" not in capsys.readouterr().err
