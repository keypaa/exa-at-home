# tests/test_embed_resumable.py — C14: resume after the final chunk.
import json

import numpy as np

from scripts.embed_resumable import _sha256_file, run_embed


class _ExplodingEmbedder:
    def encode_docs(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("a completed run must not re-embed documents")


def test_resume_accepts_state_at_n_docs_without_a_done_marker(tmp_path, monkeypatch):
    """A crash after the final chunk but before the .done marker is complete.

    The old guard rejected `rows % chunk != 0` unless rows == n_docs was
    special-cased, forcing a full --force re-embed of the whole corpus.
    """
    import exa_home.embed

    monkeypatch.setattr(exa_home.embed, "Embedder", _ExplodingEmbedder)

    docs = tmp_path / "docs.jsonl"
    docs.write_text(
        "\n".join(json.dumps({"id": f"d{i}", "text": f"body {i}"}) for i in range(3))
        + "\n"
    )
    vecs = tmp_path / "vecs.npy"
    ids = tmp_path / "ids.json"
    vecs_part = tmp_path / "vecs.npy.part"
    ids_part = tmp_path / "ids.json.part.ids"
    state = tmp_path / "vecs.npy.state.json"

    mm = np.lib.format.open_memmap(
        vecs_part, mode="w+", dtype=np.float32, shape=(3, 256)
    )
    mm[:] = 0.5
    mm.flush()
    del mm
    ids_part.write_text("".join(json.dumps(f"d{i}") + "\n" for i in range(3)))
    state.write_text(
        json.dumps(
            {
                "n_docs": 3,
                "chunk": 2,
                "batch": 2,
                "source_sha256": _sha256_file(str(docs)),
                "rows": 3,
            }
        )
    )

    shape, _dtype = run_embed(
        str(docs), str(vecs), str(ids), batch=2, chunk=2, verbose=False
    )
    assert shape == (3, 256)
    assert vecs.exists() and (tmp_path / "vecs.npy.done").exists()
    assert json.loads(ids.read_text()) == ["d0", "d1", "d2"]
    assert not state.exists() and not vecs_part.exists() and not ids_part.exists()
