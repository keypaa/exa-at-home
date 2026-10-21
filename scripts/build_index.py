# scripts/build_index.py
"""Assemble a versioned index/ dir from vectors + centroids + docs.

Usage:
    python scripts/build_index.py --vecs vecs.npy --ids ids.json \\
        --centroids centroids.npy --docs docs.jsonl --out index/ [--filter]

Writes the locked on-disk layout (see exa_home/index_format.py SCHEMA):
  centroids.f32  Kx256 float32 little-endian
  codes.bin      n×32 bytes, M2 sign-bit packing (bitorder="little")
  lists.bin      u32 K, then per list: u32 len + len×u32 row ids (LE)
  doc_ids.json   row-order id list (copy of --ids)
  store/         ContentStore shards populated from --docs (default
                 <out>/store; the e2e --ann ivf path defaults to this)
  filter/        only with --filter (built BEFORE the manifest is written)
  manifest.json  sha256 per file, written LAST (fail-fast verify)

Tier-0 safe: stdlib + numpy only at import; `ann_core` is never imported
(the layout is asserted via exa_home.index_format, same pattern as Task 12).
--filter needs pyroaring (declared dep); centroid training lives in
scripts/train_centroids.py (sklearn, cloud-only) — this script only serves
pre-trained centroids.

Centroid assignment runs in row chunks so the 5M-doc full build never
materializes the n×K score matrix. The chunk size is a function of K:
the transient chunk×K float32 score matrix stays inside CHUNK_BYTES
(200 MiB) — the old fixed 50k-row chunk needed 0.2GB at K=1024 but 2.0GB
at K=10k and 6.0GB at K=30k.
"""

import argparse
import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from exa_home.config import CRAWL, MAX_TERMS

# Transient score-matrix budget: the centroid-assignment chunk (rows × K
# float32 scores) stays inside this many bytes.
CHUNK_BYTES = 200 * 1024 * 1024
# Floor for the chunk size: below this, the per-chunk numpy call overhead
# dominates (and the bound above is exceeded only for K > 50k).
MIN_CHUNK = 1_000

# Provenance sidecar: WHICH corpus an index was built from. It is NOT part
# of the locked index layout (manifest.json keys, centroids.f32, codes.bin,
# lists.bin, doc_ids.json stay exactly as specified) and it is written AFTER
# the manifest, so the manifest's file set never changes. The pipeline uses
# it to decide whether an existing index may be reused at all.
SOURCE_FILE = "source.json"


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def source_identity(vecs_path: str, ids_path: str, docs_path: str) -> dict:
    """Identity of the corpus an index is built from.

    The paths alone are NOT an identity: the same file name can hold a
    different corpus after a re-ingest (the toy/full pipelines share the
    working directory). So the corpus size plus the size + content hash of
    the vectors and the ids go in as well.
    """
    from exa_home.index_format import sha256

    return {
        "docs": docs_path,
        "docs_size": _file_size(docs_path),
        "vecs": vecs_path,
        "vecs_size": _file_size(vecs_path),
        "vecs_sha256": sha256(vecs_path),
        "ids": ids_path,
        "ids_size": _file_size(ids_path),
        "ids_sha256": sha256(ids_path),
    }


def same_source(recorded: dict | None, current: dict) -> bool:
    """True when `recorded` describes exactly the corpus `current` does.

    The paths are deliberately NOT compared (the same corpus may be reached
    through a relative or an absolute path); the sizes + hashes are.
    """
    if not recorded:
        return False
    return all(
        recorded.get(key) == current.get(key)
        for key in ("docs_size", "vecs_size", "vecs_sha256", "ids_size", "ids_sha256")
    )


def read_source(index_dir: str) -> dict | None:
    """The recorded provenance of an index dir (None when unknown)."""
    try:
        with open(os.path.join(index_dir, SOURCE_FILE)) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def write_source(index_dir: str, source: dict) -> None:
    with open(os.path.join(index_dir, SOURCE_FILE), "w") as f:
        json.dump(source, f, indent=2, sort_keys=True)


def chunk_size(k: int) -> int:
    """Rows per centroid-assignment chunk for K centroids.

    The transient score matrix is chunk × K float32, so the chunk shrinks
    as K grows: the transient stays within CHUNK_BYTES (200 MiB) unless
    the MIN_CHUNK floor dominates (K > ~52k).
    """
    if k <= 0:
        return MIN_CHUNK
    return max(MIN_CHUNK, CHUNK_BYTES // (k * 4))


def assign(vecs, centroids, verbose: bool = False):
    import numpy as np

    n = len(vecs)
    out = np.empty(n, dtype=np.int64)
    chunk = chunk_size(len(centroids))
    n_chunks = (n + chunk - 1) // chunk
    for i, s in enumerate(range(0, n, chunk)):
        e = min(s + chunk, n)
        out[s:e] = np.argmax(vecs[s:e] @ centroids.T, axis=1)
        if verbose and (i + 1) % max(1, n_chunks // 10) == 0:
            print(f"assign: chunk {i + 1}/{n_chunks} ({e}/{n} docs)", flush=True)
    return out


def write_lists(path: str, assign_v, k: int, verbose: bool = False) -> None:
    """Write lists.bin: u32 K, then per list: u32 len + len×u32 row ids (LE).

    ONE stable argsort of the assignment vector plus searchsorted
    boundaries, instead of one `np.where(assign_v == c)` per cluster —
    that scanned the WHOLE assignment array K times (O(n·K): 10.3s at
    n=1M/K=10k vs 0.08s here).

    The row ids are identical to the np.where build: the argsort is
    stable, so within a cluster the rows keep their ascending order.
    """
    import numpy as np

    order = np.argsort(assign_v, kind="stable")
    bounds = np.searchsorted(assign_v[order], np.arange(k + 1))
    with open(path, "wb") as f:
        f.write(struct.pack("<I", k))
        for c in range(k):
            rows = order[bounds[c] : bounds[c + 1]].astype("<u4")
            f.write(struct.pack("<I", len(rows)))
            f.write(rows.tobytes())
            if verbose and (c + 1) % max(1, k // 10) == 0:
                print(f"lists: {c + 1}/{k} posting lists", flush=True)


def main(
    vecs_path: str,
    ids_path: str,
    centroids_path: str,
    docs_path: str,
    out: str,
    store: str | None = None,
    no_store: bool = False,
    filter: bool = False,
    max_terms: int = 50_000,
    crawl: str = "CC-MAIN-2026-34",
    verbose: bool = False,
    source: dict | None = None,
) -> dict:
    import numpy as np

    from exa_home.index_format import (
        CENTROIDS_FILE,
        CODES_FILE,
        DIM,
        DOC_IDS_FILE,
        LISTS_FILE,
        write_manifest,
    )

    vecs = np.load(vecs_path)
    if vecs.ndim != 2 or vecs.shape[1] != DIM:
        raise ValueError(f"{vecs_path}: want n×{DIM} float32, got {vecs.shape}")
    vecs = vecs.astype(np.float32, copy=False)
    with open(ids_path) as f:
        ids = json.load(f)
    if len(ids) != len(vecs):
        raise ValueError(f"{ids_path}: {len(ids)} ids vs {len(vecs)} vecs")
    centroids = np.load(centroids_path).astype(np.float32, copy=False)
    if centroids.ndim != 2 or centroids.shape[1] != DIM:
        raise ValueError(f"{centroids_path}: want K×{DIM}, got {centroids.shape}")
    n, k = len(vecs), len(centroids)

    os.makedirs(out, exist_ok=True)
    if verbose:
        print(
            f"assign: {n} docs -> {k} clusters "
            f"({chunk_size(k)} rows/chunk, {CHUNK_BYTES // (1024 * 1024)} MiB budget)",
            flush=True,
        )
    assign_v = assign(vecs, centroids, verbose=verbose)

    centroids.astype("<f4").tofile(os.path.join(out, CENTROIDS_FILE))
    pack = np.packbits((vecs > 0).astype(np.uint8), axis=1, bitorder="little")
    assert pack.shape == (n, DIM // 8), pack.shape
    pack.tofile(os.path.join(out, CODES_FILE))
    write_lists(os.path.join(out, LISTS_FILE), assign_v, k, verbose=verbose)
    with open(os.path.join(out, DOC_IDS_FILE), "w") as f:
        json.dump(ids, f)

    n_stored = 0
    if not no_store:
        from exa_home.store import ContentStore
        from scripts.jsonl import read_jsonl

        store_dir = store or os.path.join(out, "store")
        cs = ContentStore(store_dir)
        # Track offsets during the bulk write so no rescan is needed:
        # save_index() persists store/offsets.json for O(1) cold gets.
        cs.begin_bulk()
        for d in read_jsonl(docs_path):
            cs.put(d)
            n_stored += 1
            if verbose and n_stored % 100_000 == 0:
                print(f"store: {n_stored} docs", flush=True)
        if n_stored != n:
            raise ValueError(f"{docs_path}: {n_stored} docs vs {n} vecs")
        cs.save_index()

    if filter:
        from scripts.build_filter import build as build_filter
        from scripts.jsonl import JsonlFile

        # Streamed: the docs are handed over one at a time (the old code
        # built the whole list first — GBs of text as Python dicts at 5M
        # docs). The filter layout is unchanged.
        build_filter(JsonlFile(docs_path), os.path.join(out, "filter"), max_terms)

    # Manifest LAST so filter/ + store shards are covered by verify.
    # The provenance sidecar is written after it (and any stale copy is
    # removed before it) so the manifest's file set — and with it the
    # locked on-disk index layout — never changes.
    src = (
        source
        if source is not None
        else source_identity(vecs_path, ids_path, docs_path)
    )
    stale_source = os.path.join(out, SOURCE_FILE)
    if os.path.exists(stale_source):
        os.remove(stale_source)
    write_manifest(out, {"dim": DIM, "n_docs": n, "n_centroids": k, "crawl": crawl})
    write_source(out, src)
    print(f"index {n} docs, K={k} -> {out} (stored {n_stored}, filter={filter})")
    return {"n_docs": n, "n_centroids": k, "n_stored": n_stored}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--centroids", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--store", default=None, help="content store root (default <out>/store)"
    )
    ap.add_argument("--no-store", action="store_true")
    ap.add_argument(
        "--filter",
        action="store_true",
        help="also build index/filter/ (needs pyroaring)",
    )
    ap.add_argument("--max-terms", type=int, default=MAX_TERMS)
    ap.add_argument("--crawl", default=CRAWL)
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="progress output (assign/store/lists stages)",
    )
    a = ap.parse_args()
    main(
        a.vecs,
        a.ids,
        a.centroids,
        a.docs,
        a.out,
        a.store,
        a.no_store,
        a.filter,
        a.max_terms,
        a.crawl,
        verbose=a.verbose,
    )
