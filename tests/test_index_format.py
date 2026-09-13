"""Tier 0: versioned index/ dir schema + manifest verify (Task 6 / M3)."""
import json
import struct

import numpy as np
import pytest

from exa_home.index_format import (
    SCHEMA,
    SCHEMA_VERSION,
    DIM,
    CODE_BYTES,
    MANIFEST,
    CENTROIDS_FILE,
    CODES_FILE,
    LISTS_FILE,
    DOC_IDS_FILE,
    sha256,
    write_manifest,
    verify_manifest,
)


def test_schema_constants_match_locked_layout():
    """SCHEMA shared by both sides: version 1, dim 256, 32-byte codes,
    little-endian posting lists of u32 row ids."""
    assert SCHEMA_VERSION == 1
    assert DIM == 256
    assert CODE_BYTES == 32  # 256 bits / 8
    assert SCHEMA["version"] == 1
    assert SCHEMA["dim"] == 256
    assert SCHEMA["code_bytes"] == 32
    assert SCHEMA["files"]["manifest"] == MANIFEST == "manifest.json"
    assert SCHEMA["files"]["centroids"] == CENTROIDS_FILE == "centroids.f32"
    assert SCHEMA["files"]["codes"] == CODES_FILE == "codes.bin"
    assert SCHEMA["files"]["lists"] == LISTS_FILE == "lists.bin"
    assert SCHEMA["files"]["doc_ids"] == DOC_IDS_FILE == "doc_ids.json"
    assert SCHEMA["lists"]["int"] == "u32"
    assert SCHEMA["lists"]["endian"] == "little"


def test_manifest_verify_roundtrip(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"hello")
    write_manifest(str(tmp_path), {"dim": 256, "n_docs": 1, "n_centroids": 1})
    m = verify_manifest(str(tmp_path))
    assert m["version"] == 1
    assert m["dim"] == 256
    assert m["files"]["a.bin"] == sha256(str(tmp_path / "a.bin"))
    # manifest.json itself is never checksummed (it holds the checksums).
    assert MANIFEST not in m["files"]


def test_manifest_verify_rejects_corruption(tmp_path):
    from exa_home.index_format import write_manifest, verify_manifest
    (tmp_path / "a.bin").write_bytes(b"hello")
    write_manifest(str(tmp_path), {"dim": 256})
    (tmp_path / "a.bin").write_bytes(b"evil!")
    try:
        verify_manifest(str(tmp_path))
        assert False, "should have raised"
    except ValueError:
        pass


def test_manifest_verify_rejects_version_mismatch(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"hello")
    write_manifest(str(tmp_path), {"dim": 256})
    mpath = tmp_path / MANIFEST
    m = json.loads(mpath.read_text())
    m["version"] = 999
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="version"):
        verify_manifest(str(tmp_path))


def test_index_dir_byte_layout(tmp_path):
    """The locked on-disk layout parses back exactly: Kx256 LE centroids,
    n×32 codes, u32-K posting lists, row-order doc ids."""
    rng = np.random.default_rng(0)
    n, k, dim = 64, 4, 256
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    centroids = rng.normal(size=(k, dim)).astype(np.float32)
    assign = np.argmax(vecs @ centroids.T, axis=1)
    ids = [f"d{i}" for i in range(n)]

    centroids.astype("<f4").tofile(tmp_path / CENTROIDS_FILE)
    pack = np.packbits((vecs > 0).astype(np.uint8), axis=1, bitorder="little")
    assert pack.shape == (n, 32)
    pack.tofile(tmp_path / CODES_FILE)
    with open(tmp_path / LISTS_FILE, "wb") as f:
        f.write(struct.pack("<I", k))
        for c in range(k):
            rows = np.where(assign == c)[0].astype("<u4")
            f.write(struct.pack("<I", len(rows)))
            f.write(rows.tobytes())
    (tmp_path / DOC_IDS_FILE).write_text(json.dumps(ids))
    write_manifest(str(tmp_path), {"dim": dim, "n_docs": n, "n_centroids": k})
    m = verify_manifest(str(tmp_path))
    assert m["n_docs"] == n and m["n_centroids"] == k

    # Read back independent of the writer above.
    got_c = np.fromfile(tmp_path / CENTROIDS_FILE, dtype="<f4").reshape(k, dim)
    assert np.array_equal(got_c, centroids)
    got_codes = np.fromfile(tmp_path / CODES_FILE, dtype=np.uint8).reshape(n, 32)
    assert np.array_equal(got_codes, pack)
    raw = (tmp_path / LISTS_FILE).read_bytes()
    off = 0
    (k2,) = struct.unpack_from("<I", raw, off)
    off += 4
    assert k2 == k
    total = 0
    for _ in range(k):
        (ln,) = struct.unpack_from("<I", raw, off)
        off += 4
        off += 4 * ln
        total += ln
    assert total == n and off == len(raw)
    assert json.loads((tmp_path / DOC_IDS_FILE).read_text()) == ids
