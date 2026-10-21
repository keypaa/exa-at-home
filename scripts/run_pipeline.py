#!/usr/bin/env python3
"""One-command pipeline: slice -> download -> ingest -> embed -> index -> queries -> gate.

Each stage is a subcommand (runnable/ retryable independently) plus two
presets: `toy` (4 WET, K=1024, fast) and `full` (N WET files, K centroids).
Every stage prints its own wall time (pytest-style summary line).

Stages are idempotent: existing outputs are reused unless --force is passed.
All stage functions are importable with injectable paths (Tier-0 testable);
only the `main()` CLI touches argparse / the network.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import time

WET_PATHS_URL = "https://data.commoncrawl.org/crawl-data/{crawl}/wet.paths.gz"
BASE = "https://data.commoncrawl.org"
# Absolute script path: stage subprocesses must not depend on the CWD
# (notebook kernels run at the workspace root, not the repo).
_HERE = os.path.dirname(os.path.abspath(__file__))


def _log(stage: str, t0: float, extra: str = "") -> None:
    print(f"[{stage}] done in {time.perf_counter() - t0:.1f}s{extra}", flush=True)


def _skip(path: str, force: bool) -> bool:
    return os.path.exists(path) and not force


def stage_fetch(
    crawl: str, n_wet: int, out: str, force: bool = False, verbose: bool = False
) -> list[str]:
    """First N WET URLs from wet.paths.gz (no density ranking; see RUNBOOK §1).

    Network-dependent (urllib stdlib only). Returns the URL list.
    """
    import gzip
    import urllib.request

    t0 = time.perf_counter()
    if _skip(out, force):
        if verbose:
            print(f"[fetch] reusing {out}", flush=True)
        with open(out) as f:
            return [line.strip() for line in f if line.strip()]
    url = WET_PATHS_URL.format(crawl=crawl)
    if verbose:
        print(f"[fetch] {url} -> first {n_wet}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read()
    lines = gzip.decompress(raw).decode("utf-8").splitlines()[:n_wet]
    urls = [f"{BASE}/{line}" for line in lines if line.strip()]
    with open(out, "w") as f:
        f.writelines(u + "\n" for u in urls)
    _log("fetch", t0, f" ({len(urls)} URLs -> {out})")
    return urls


def stage_download(
    urls_file: str,
    out_dir: str,
    workers: int = 16,
    force: bool = False,
    verbose: bool = False,
) -> str:
    """WET URLs -> data dir. Delegates to download_wet.py."""
    from scripts.download_wet import main as dl_main

    t0 = time.perf_counter()
    with open(urls_file) as f:
        urls = [line.strip() for line in f if line.strip()]
    if not force and os.path.isdir(out_dir):
        have = len(
            glob.glob(os.path.join(out_dir, "*.warc.wet.gz"))
            + glob.glob(os.path.join(out_dir, "*.wet.gz"))
        )
        if have >= len(urls):
            if verbose:
                print(f"[download] reusing {out_dir} ({have} files)", flush=True)
            return out_dir
    os.makedirs(out_dir, exist_ok=True)
    # download_wet takes a urls file; reuse the caller's file directly.
    dl_main(urls_file, out_dir, workers)
    _log("download", t0, f" ({out_dir})")
    return out_dir


def stage_ingest(
    wet_dir: str, docs_path: str, force: bool = False, verbose: bool = False
) -> dict:
    """WET dir -> deduped docs.jsonl (digest-first + URL-second)."""
    from exa_home.ingest import read_wet, write_docs_jsonl

    t0 = time.perf_counter()
    if _skip(docs_path, force):
        with open(docs_path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        if verbose:
            print(f"[ingest] reusing {docs_path} ({n} docs)", flush=True)
        return {"docs": n, "reused": True}

    def all_docs():
        pats = [
            os.path.join(wet_dir, "*.warc.wet.gz"),
            os.path.join(wet_dir, "*.wet.gz"),
        ]
        files = sorted(f for p in pats for f in glob.glob(p))
        if verbose:
            print(f"[ingest] {len(files)} WET files", flush=True)
        for p in files:
            yield from read_wet(p)

    tmp = docs_path + ".tmp"
    stats = write_docs_jsonl(all_docs(), tmp)
    # URL-second dedup + subset exclusion (digest-first is in write_docs_jsonl).
    import re

    rx = re.compile(r"robotstxt|crawldiagnostics", re.IGNORECASE)
    seen: set[str] = set()
    kept = 0
    excluded = 0
    dup_urls = 0
    with (
        open(tmp, encoding="utf-8") as fin,
        open(docs_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            # The exclusion applies to the URL (and an optional `subset`
            # field) — NEVER to the whole JSON line: the line carries the
            # page TEXT, so a page whose body merely mentions robots.txt
            # was silently dropped from the corpus.
            d = json.loads(line)
            url = d.get("url", "")
            subset = d.get("subset", "")
            if rx.search(url) or (subset and rx.search(subset)):
                excluded += 1
                continue
            if url in seen:
                dup_urls += 1
                continue
            seen.add(url)
            fout.write(line)
            kept += 1
            if verbose and kept % 100_000 == 0:
                print(f"[ingest] url-dedup: {kept}", flush=True)
    os.remove(tmp)
    _log(
        "ingest",
        t0,
        f" ({kept} docs -> {docs_path}; {stats}; "
        f"excluded={excluded}; dup_url={dup_urls})",
    )
    return {"docs": kept, "stats": stats, "excluded": excluded, "dup_urls": dup_urls}


def _count_lines(path: str) -> int:
    """Number of non-empty lines in a text file (one cheap scan)."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def stage_embed(
    docs_path: str,
    vecs_path: str,
    ids_path: str,
    batch: int = 256,
    force: bool = False,
    verbose: bool = False,
    show_progress: bool = False,
) -> tuple:
    """docs.jsonl -> vecs.npy + ids.json (real Embedder; GPU box only).

    The corpus is STREAMED: the docs are read in `batch`-sized chunks and
    embedded chunk by chunk, so the whole corpus is never held in RAM as
    Python dicts (at 5M docs that is multiple GB of text — an order of
    magnitude more than the JSONL itself). The vectors are written through
    a numpy memmap, so even the n×256 float32 output never has to fit in
    RAM at once.

    Returns (shape, dtype) — unchanged.
    """
    import numpy as np

    from exa_home.embed import Embedder

    t0 = time.perf_counter()
    if _skip(vecs_path, force) and _skip(ids_path, force):
        v = np.load(vecs_path, mmap_mode="r")
        if verbose:
            print(f"[embed] reusing {vecs_path} {v.shape}", flush=True)
        return v.shape, str(v.dtype)
    emb = Embedder()
    n_docs = _count_lines(docs_path)
    if n_docs == 0:
        raise ValueError(f"{docs_path}: no documents to embed")
    tmp_vecs = vecs_path + ".tmp.npy"
    tmp_ids = ids_path + ".tmp"
    mm = None
    rows = 0
    ids_written = 0
    dim = 0
    dtype = None

    def flush(texts: list[str], ids: list[str]) -> None:
        """Embed one chunk into the memmap + append its ids."""
        nonlocal mm, rows, ids_written, dim, dtype
        vecs = emb.encode_docs(texts)
        if mm is None:
            dim = int(vecs.shape[1])
            dtype = vecs.dtype
            mm = np.lib.format.open_memmap(
                tmp_vecs,
                mode="w+",
                dtype=dtype,
                shape=(n_docs, dim),
            )
        mm[rows : rows + len(vecs)] = vecs
        rows += len(vecs)
        for doc_id in ids:
            fid.write(("" if ids_written == 0 else ",") + json.dumps(doc_id))
            ids_written += 1

    try:
        with (
            open(docs_path, encoding="utf-8") as fin,
            open(tmp_ids, "w", encoding="utf-8") as fid,
        ):
            fid.write("[")
            chunk_texts: list[str] = []
            chunk_ids: list[str] = []
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                chunk_texts.append(d["text"][:12000])
                chunk_ids.append(d["id"])
                if len(chunk_texts) < batch:
                    continue
                flush(chunk_texts, chunk_ids)
                chunk_texts, chunk_ids = [], []
                if show_progress:
                    print(f"[embed] {rows}/{n_docs} docs", flush=True)
            if chunk_texts:
                flush(chunk_texts, chunk_ids)
            fid.write("]")
        if mm is None:
            raise ValueError(f"{docs_path}: no documents to embed")
        if rows != n_docs:
            raise ValueError(f"{docs_path}: embedded {rows} docs, expected {n_docs}")
        mm.flush()
        del mm
        os.replace(tmp_vecs, vecs_path)
        os.replace(tmp_ids, ids_path)
    finally:
        for tmp in (tmp_vecs, tmp_ids):
            if os.path.exists(tmp):
                os.remove(tmp)
    shape = (n_docs, dim)
    _log("embed", t0, f" ({shape} {dtype})")
    return shape, str(dtype)


def stage_index(
    vecs_path: str,
    ids_path: str,
    docs_path: str,
    out: str,
    k: int,
    sample: int = 500_000,
    use_filter: bool = True,
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """vecs + ids + docs -> versioned index/ (centroids, codes, lists, store).

    Reuse is PROVENANCE-GATED: an existing index is reused only when it was
    built from exactly these inputs (corpus size + vecs/ids hashes — see
    scripts/build_index.py source_identity). A different corpus, or an index
    whose provenance is unknown, forces a REBUILD (loudly) — the old code
    short-circuited on the mere EXISTENCE of <out>/manifest.json, so the
    toy index was silently reused by the full pipeline and the gate then
    measured a mismatched index.
    """
    from exa_home.index_format import verify_manifest
    from scripts.build_index import main as build_main
    from scripts.build_index import read_source, same_source, source_identity
    from scripts.train_centroids import main as train_main

    t0 = time.perf_counter()
    cent_path = os.path.join(os.path.dirname(out.rstrip("/")) or ".", "centroids.npy")
    # One hash pass over the vectors per call (5 GB at full scale): that is
    # the price of a reuse decision that can actually detect a mismatch.
    src = source_identity(vecs_path, ids_path, docs_path)
    if _skip(os.path.join(out, "manifest.json"), force):
        recorded = read_source(out)
        if same_source(recorded, src):
            m = verify_manifest(out)
            if verbose:
                print(
                    f"[index] reusing {out} ({m['n_docs']} docs, K={m['n_centroids']})",
                    flush=True,
                )
            return m
        # Mismatch (or unknown provenance): NEVER reuse silently.
        print(
            f"[index] {out} does not match the current corpus "
            f"(recorded source: {recorded}) — rebuilding instead of reusing "
            f"a mismatched index",
            file=sys.stderr,
            flush=True,
        )
    train_main(vecs_path, k, sample, cent_path, verbose=verbose)
    build_main(
        vecs_path,
        ids_path,
        cent_path,
        docs_path,
        out,
        filter=use_filter,
        verbose=verbose,
        source=src,
    )
    m = verify_manifest(out)
    _log("index", t0, f" ({m['n_docs']} docs, K={m['n_centroids']} -> {out})")
    return m


def stage_queries(
    docs_path: str,
    vecs_path: str,
    ids_path: str,
    nq: int,
    queries_path: str = "q.jsonl",
    gt_path: str = "gt.jsonl",
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """docs/vecs/ids -> q.jsonl + gt.jsonl (gate queries + ground truth).

    ALIGNED by construction (M5 lesson, commit 1445aa0): q.jsonl is
    written first from docs[:nq], then ground_truth --queries embeds
    those exact texts — gt query k is always pred query k. The old
    --nq random-vector path is never used here (different query sets
    score recall@10 = 0.0000 by construction).
    """
    import subprocess

    t0 = time.perf_counter()
    if _skip(queries_path, force) and _skip(gt_path, force):
        if verbose:
            print(f"[queries] reusing {queries_path} + {gt_path}", flush=True)
        return {"reused": True}
    # Only the first `nq` docs are needed: read them and stop. The old code
    # materialised the WHOLE corpus as a list of dicts to use docs[:nq].
    with open(docs_path, encoding="utf-8") as f:
        docs = [json.loads(line) for line in itertools.islice(f, nq)]
    if len(docs) < nq:
        print(
            f"[queries] warning: corpus has only {len(docs)} docs, {nq} requested",
            file=sys.stderr,
            flush=True,
        )
    with open(queries_path, "w") as f:
        f.writelines(json.dumps({"query": d["text"][:200]}) + "\n" for d in docs)
    subprocess.run(
        [
            sys.executable,
            os.path.join(_HERE, "ground_truth.py"),
            "--vecs",
            vecs_path,
            "--ids",
            ids_path,
            "--queries",
            queries_path,
            "--out",
            gt_path,
        ],
        check=True,
    )
    _log("queries", t0, f" ({nq} queries -> {queries_path}, {gt_path})")
    return {"nq": nq}


def cmd_toy(a) -> int:
    # The toy owns the CANONICAL gate paths (index/ + q.jsonl + gt.jsonl at
    # the repo root); the full build writes its own corpus-named artifacts
    # (see cmd_full), so the two never share an index dir.
    stage_fetch("CC-MAIN-2026-34", 4, "wet_urls.txt", force=a.force, verbose=a.verbose)
    stage_download(
        "wet_urls.txt", "data/wet", workers=a.workers, force=a.force, verbose=a.verbose
    )
    stage_ingest("data/wet", "docs.jsonl", force=a.force, verbose=a.verbose)
    stage_embed(
        "docs.jsonl",
        "vecs.npy",
        "ids.json",
        batch=a.batch,
        force=a.force,
        verbose=a.verbose,
        show_progress=a.verbose,
    )
    stage_index(
        "vecs.npy",
        "ids.json",
        "docs.jsonl",
        "index/",
        k=1024,
        use_filter=True,
        force=a.force,
        verbose=a.verbose,
    )
    stage_queries(
        "docs.jsonl", "vecs.npy", "ids.json", 200, force=a.force, verbose=a.verbose
    )
    print(
        "toy pipeline complete: index/ + q.jsonl + gt.jsonl ready "
        "(run the gate: scripts/e2e_latency.py ...)"
    )
    return 0


def cmd_full(a) -> int:
    tag = f"{a.wet_files}wet"
    urls = f"wet_urls_{tag}.txt"
    wet_dir = f"data/wet_{tag}"
    # The index + query artifacts are named after the CORPUS: the toy writes
    # index/ + q.jsonl + gt.jsonl (the canonical gate paths), the full build
    # writes its own. Sharing one index dir meant the full pipeline silently
    # reused the 4-doc toy index while stage_queries built q.jsonl/gt.jsonl
    # from the FULL corpus — the gate then measured a mismatched index.
    index_dir = f"index_{tag}"
    q_path = f"q_{tag}.jsonl"
    gt_path = f"gt_{tag}.jsonl"
    stage_fetch("CC-MAIN-2026-34", a.wet_files, urls, force=a.force, verbose=a.verbose)
    stage_download(urls, wet_dir, workers=a.workers, force=a.force, verbose=a.verbose)
    docs = f"docs_{tag}.jsonl"
    vecs = f"vecs_{tag}.npy"
    ids = f"ids_{tag}.json"
    stage_ingest(wet_dir, docs, force=a.force, verbose=a.verbose)
    stage_embed(
        docs,
        vecs,
        ids,
        batch=a.batch,
        force=a.force,
        verbose=a.verbose,
        show_progress=True,
    )
    stage_index(
        vecs,
        ids,
        docs,
        f"{index_dir}/",
        k=a.k,
        use_filter=True,
        force=a.force,
        verbose=a.verbose,
    )
    stage_queries(
        docs,
        vecs,
        ids,
        1000,
        queries_path=q_path,
        gt_path=gt_path,
        force=a.force,
        verbose=a.verbose,
    )
    print(
        f"full pipeline complete ({a.wet_files} WET, K={a.k}): "
        f"{index_dir}/ + {q_path} + {gt_path} ready "
        f"(gate: scripts/e2e_latency.py --ann ivf --index {index_dir}/ "
        f"--queries {q_path} ...)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--force", action="store_true", help="redo stages even if outputs exist"
    )
    ap.add_argument("--verbose", action="store_true", help="per-stage progress output")
    ap.add_argument("--workers", type=int, default=16, help="download workers")
    ap.add_argument("--batch", type=int, default=256, help="embed batch size")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="WET dir -> docs.jsonl")
    p.add_argument("--wet", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=lambda a: stage_ingest(a.wet, a.out, a.force, a.verbose))

    p = sub.add_parser("embed", help="docs.jsonl -> vecs.npy + ids.json")
    p.add_argument("--docs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", required=True)
    p.set_defaults(
        fn=lambda a: stage_embed(
            a.docs, a.out, a.ids, a.batch, a.force, a.verbose, a.verbose
        )
    )

    p = sub.add_parser("index", help="assemble versioned index/")
    p.add_argument("--vecs", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--docs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--no-filter", action="store_true")
    p.set_defaults(
        fn=lambda a: stage_index(
            a.vecs,
            a.ids,
            a.docs,
            a.out,
            a.k,
            use_filter=not a.no_filter,
            force=a.force,
            verbose=a.verbose,
        )
    )

    p = sub.add_parser("queries", help="ground truth + gate queries")
    p.add_argument("--docs", required=True)
    p.add_argument("--vecs", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--nq", type=int, default=200)
    p.set_defaults(
        fn=lambda a: stage_queries(
            a.docs, a.vecs, a.ids, a.nq, force=a.force, verbose=a.verbose
        )
    )

    p = sub.add_parser("toy", help="full toy pipeline (4 WET, K=1024)")
    p.set_defaults(fn=cmd_toy)

    p = sub.add_parser("full", help="full pipeline: N WET files, K centroids")
    p.add_argument("--wet-files", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.set_defaults(fn=cmd_full)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
