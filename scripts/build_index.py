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
materializes the n×K score matrix.
"""
import argparse
import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHUNK = 50_000


def assign(vecs, centroids, verbose: bool = False):
    import numpy as np

    n = len(vecs)
    out = np.empty(n, dtype=np.int64)
    n_chunks = (n + CHUNK - 1) // CHUNK
    for i, s in enumerate(range(0, n, CHUNK)):
        e = min(s + CHUNK, n)
        out[s:e] = np.argmax(vecs[s:e] @ centroids.T, axis=1)
        if verbose and (i + 1) % max(1, n_chunks // 10) == 0:
            print(f"assign: chunk {i + 1}/{n_chunks} ({e}/{n} docs)", flush=True)
    return out


def main(vecs_path: str, ids_path: str, centroids_path: str, docs_path: str,
         out: str, store: str | None = None, no_store: bool = False,
         filter: bool = False, max_terms: int = 50_000,
         crawl: str = "CC-MAIN-2026-34", verbose: bool = False) -> dict:
    import numpy as np

    from exa_home.index_format import (
        CENTROIDS_FILE, CODES_FILE, DIM, DOC_IDS_FILE, LISTS_FILE,
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
        print(f"assign: {n} docs -> {k} clusters (50k-row chunks)", flush=True)
    assign_v = assign(vecs, centroids, verbose=verbose)

    centroids.astype("<f4").tofile(os.path.join(out, CENTROIDS_FILE))
    pack = np.packbits((vecs > 0).astype(np.uint8), axis=1, bitorder="little")
    assert pack.shape == (n, DIM // 8), pack.shape
    pack.tofile(os.path.join(out, CODES_FILE))
    with open(os.path.join(out, LISTS_FILE), "wb") as f:
        f.write(struct.pack("<I", k))
        for c in range(k):
            rows = np.where(assign_v == c)[0].astype("<u4")
            f.write(struct.pack("<I", len(rows)))
            f.write(rows.tobytes())
            if verbose and (c + 1) % max(1, k // 10) == 0:
                print(f"lists: {c + 1}/{k} posting lists", flush=True)
    with open(os.path.join(out, DOC_IDS_FILE), "w") as f:
        json.dump(ids, f)

    n_stored = 0
    if not no_store:
        from exa_home.store import ContentStore

        store_dir = store or os.path.join(out, "store")
        cs = ContentStore(store_dir)
        # Track offsets during the bulk write so no rescan is needed:
        # save_index() persists store/offsets.json for O(1) cold gets.
        cs.begin_bulk()
        with open(docs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cs.put(json.loads(line))
                    n_stored += 1
                    if verbose and n_stored % 100_000 == 0:
                        print(f"store: {n_stored} docs", flush=True)
        if n_stored != n:
            raise ValueError(f"{docs_path}: {n_stored} docs vs {n} vecs")
        cs.save_index()

    if filter:
        from scripts.build_filter import build as build_filter

        docs = []
        with open(docs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        build_filter(docs, os.path.join(out, "filter"), max_terms)

    # Manifest LAST so filter/ + store shards are covered by verify.
    write_manifest(out, {"dim": DIM, "n_docs": n, "n_centroids": k,
                         "crawl": crawl})
    print(f"index {n} docs, K={k} -> {out} (stored {n_stored}, filter={filter})")
    return {"n_docs": n, "n_centroids": k, "n_stored": n_stored}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--centroids", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default=None,
                    help="content store root (default <out>/store)")
    ap.add_argument("--no-store", action="store_true")
    ap.add_argument("--filter", action="store_true",
                    help="also build index/filter/ (needs pyroaring)")
    ap.add_argument("--max-terms", type=int, default=50_000)
    ap.add_argument("--crawl", default="CC-MAIN-2026-34")
    ap.add_argument("--verbose", action="store_true",
                    help="progress output (assign/store/lists stages)")
    a = ap.parse_args()
    main(a.vecs, a.ids, a.centroids, a.docs, a.out, a.store, a.no_store,
         a.filter, a.max_terms, a.crawl, verbose=a.verbose)
