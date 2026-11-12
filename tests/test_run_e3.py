# tests/test_run_e3.py — C10: docs/index validation + corpus-fingerprinted targets.
import json

import pytest

from scripts.build_index import write_source
from scripts.run_e3 import (
    docs_fingerprint,
    infer_docs_path,
    targets_path_for,
    validate_docs_for_index,
)


def test_infer_docs_prefers_corpus_named_docs(tmp_path):
    """index_1wet must resolve docs_1wet.jsonl, not the toy root docs.jsonl."""
    (tmp_path / "docs.jsonl").write_text("toy\n")
    (tmp_path / "docs_1wet.jsonl").write_text("full\n")
    index = tmp_path / "index_1wet"
    index.mkdir()

    assert infer_docs_path(tmp_path, index) == tmp_path / "docs_1wet.jsonl"

    # index/ (the toy) still resolves the canonical docs.jsonl.
    toy_index = tmp_path / "index"
    toy_index.mkdir()
    assert infer_docs_path(tmp_path, toy_index) == tmp_path / "docs.jsonl"


def test_validate_docs_accepts_matching_source(tmp_path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text('{"id": "a"}\n')
    index = tmp_path / "index"
    index.mkdir()
    write_source(
        str(index),
        {
            "docs_sha256": docs_fingerprint(docs),
            "docs_size": docs.stat().st_size,
        },
    )
    assert validate_docs_for_index(docs, index) == docs_fingerprint(docs)


def test_validate_docs_rejects_corpus_mismatch(tmp_path):
    """An explicit --docs from another corpus must be an error, not a
    silently mismatched target set."""
    other = tmp_path / "other.jsonl"
    other.write_text('{"id": "other"}\n')
    docs = tmp_path / "docs.jsonl"
    docs.write_text('{"id": "a"}\n')
    index = tmp_path / "index"
    index.mkdir()
    write_source(
        str(index),
        {
            "docs_sha256": docs_fingerprint(other),
            "docs_size": other.stat().st_size,
        },
    )
    with pytest.raises(ValueError, match="does not match the corpus"):
        validate_docs_for_index(docs, index)


def test_validate_docs_legacy_source_checks_size(tmp_path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text('{"id": "a"}\n')
    index = tmp_path / "index"
    index.mkdir()
    write_source(str(index), {"docs_size": docs.stat().st_size + 1})
    with pytest.raises(ValueError, match="size"):
        validate_docs_for_index(docs, index)


def test_validate_docs_without_provenance_warns_but_allows(tmp_path, capsys):
    docs = tmp_path / "docs.jsonl"
    docs.write_text('{"id": "a"}\n')
    index = tmp_path / "index"
    index.mkdir()
    assert validate_docs_for_index(docs, index) == docs_fingerprint(docs)
    assert "no source provenance" in capsys.readouterr().err


def test_targets_path_is_corpus_fingerprinted(tmp_path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text(json.dumps({"id": "a"}) + "\n")
    fp = docs_fingerprint(docs)
    p = targets_path_for(docs, fp)
    assert p.name == f"sampled_targets.docs.{fp[:12]}.json"
    # No fingerprint (legacy call) keeps the old file name.
    assert targets_path_for(docs).name == "sampled_targets.docs.json"
