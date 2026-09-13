# exa_home/index_format.py
"""Versioned index/ dir: manifest.json with sha256 per file. verify fails fast."""
from __future__ import annotations
import hashlib, json, os

MANIFEST = "manifest.json"

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
        json.dump({"version": 1, **meta, "files": files}, f, indent=2)

def verify_manifest(index_dir: str) -> dict:
    with open(os.path.join(index_dir, MANIFEST)) as f:
        m = json.load(f)
    for name, want in m["files"].items():
        got = sha256(os.path.join(index_dir, name))
        if got != want:
            raise ValueError(f"corrupt index file {name}: want {want[:12]} got {got[:12]}")
    return m
