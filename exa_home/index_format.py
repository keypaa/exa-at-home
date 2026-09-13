# exa_home/index_format.py
"""Versioned index/ dir: manifest.json with sha256 per file. verify fails fast.

SCHEMA locks the on-disk layout (Task 6, M3); the Rust loader in
crates/ann-core/src/index.rs implements to these same constants.
"""
from __future__ import annotations
import hashlib, json, os

MANIFEST = "manifest.json"
CENTROIDS_FILE = "centroids.f32"
CODES_FILE = "codes.bin"
LISTS_FILE = "lists.bin"
DOC_IDS_FILE = "doc_ids.json"

SCHEMA_VERSION = 1
DIM = 256
CODE_BYTES = DIM // 8  # 32: 256 sign bits, Task 5 bit-order contract

# Shared by both sides (Python + Rust index.rs):
#   centroids.f32  K×256 float32 little-endian
#   codes.bin      n×32 bytes, M2 packing (bit i = vec[i] > 0,
#                  byte i/8, bit i%8, LSB-first == packbits bitorder="little")
#   lists.bin      u32 K, then per list: u32 len + len×u32 row ids (LE)
#   doc_ids.json   ["...", ...] in row order
SCHEMA = {
    "version": SCHEMA_VERSION,
    "dim": DIM,
    "code_bytes": CODE_BYTES,
    "files": {
        "manifest": MANIFEST,
        "centroids": CENTROIDS_FILE,
        "codes": CODES_FILE,
        "lists": LISTS_FILE,
        "doc_ids": DOC_IDS_FILE,
    },
    "centroids": {"dtype": "float32", "endian": "little", "shape": ["K", DIM]},
    "codes": {"packing": "sign-bit", "bytes_per_row": CODE_BYTES,
              "bitorder": "little"},
    "lists": {"int": "u32", "endian": "little",
              "layout": "u32 K, then per list: u32 len + len×u32 row ids"},
    "doc_ids": {"format": "json-list", "order": "row"},
}

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_manifest(index_dir: str, meta: dict) -> None:
    files = {}
    for name in sorted(os.listdir(index_dir)):
        if name == MANIFEST:
            continue
        files[name] = sha256(os.path.join(index_dir, name))
    with open(os.path.join(index_dir, MANIFEST), "w") as f:
        json.dump({"version": SCHEMA_VERSION, **meta, "files": files}, f, indent=2)

def verify_manifest(index_dir: str) -> dict:
    with open(os.path.join(index_dir, MANIFEST)) as f:
        m = json.load(f)
    if m.get("version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported index version {m.get('version')!r}: "
            f"this code reads version {SCHEMA_VERSION}"
        )
    for name, want in m["files"].items():
        got = sha256(os.path.join(index_dir, name))
        if got != want:
            raise ValueError(f"corrupt index file {name}: want {want[:12]} got {got[:12]}")
    return m
