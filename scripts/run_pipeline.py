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
import json
import os
import sys
import time

WET_PATHS_URL = ("https://data.commoncrawl.org/crawl-data/{crawl}/"
                 "wet.paths.gz")
BASE = "https://data.commoncrawl.org"


def _log(stage: str, t0: float, extra: str = "") -> None:
    print(f"[{stage}] done in {time.perf_counter() - t0:.1f}s{extra}",
          flush=True)


def _skip(path: str, force: bool) -> bool:
    return os.path.exists(path) and not force


def stage_fetch(crawl: str, n_wet: int, out: str, force: bool = False,
                verbose: bool = False) -> list[str]:
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
            return [l.strip() for l in f if l.strip()]
    url = WET_PATHS_URL.format(crawl=crawl)
    if verbose:
        print(f"[fetch] {url} -> first {n_wet}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read()
    lines = gzip.decompress(raw).decode("utf-8").splitlines()[:n_wet]
    urls = [f"{BASE}/{l}" for l in lines if l.strip()]
    with open(out, "w") as f:
        for u in urls:
            f.write(u + "\n")
    _log("fetch", t0, f" ({len(urls)} URLs -> {out})")
    return urls


def stage_download(urls_file: str, out_dir: str, workers: int = 16,
                   force: bool = False, verbose: bool = False) -> str:
    """WET URLs -> data dir. Delegates to download_wet.py."""
    from scripts.download_wet import main as dl_main
    t0 = time.perf_counter()
    with open(urls_file) as f:
        urls = [l.strip() for l in f if l.strip()]
    if not force and os.path.isdir(out_dir):
        have = len(glob.glob(os.path.join(out_dir, "*.warc.wet.gz")) +
                   glob.glob(os.path.join(out_dir, "*.wet.gz")))
        if have >= len(urls):
            if verbose:
                print(f"[download] reusing {out_dir} ({have} files)",
                      flush=True)
            return out_dir
    os.makedirs(out_dir, exist_ok=True)
    import tempfile
    # download_wet takes a urls file; reuse the caller's file directly.
    dl_main(urls_file, out_dir, workers)
    _log("download", t0, f" ({out_dir})")
    return out_dir


def stage_ingest(wet_dir: str, docs_path: str, force: bool = False,
                 verbose: bool = False) -> dict:
    """WET dir -> deduped docs.jsonl (digest-first + URL-second)."""
    from exa_home.ingest import read_wet, write_docs_jsonl
    t0 = time.perf_counter()
    if _skip(docs_path, force):
        n = sum(1 for _ in open(docs_path, encoding="utf-8"))
        if verbose:
            print(f"[ingest] reusing {docs_path} ({n} docs)", flush=True)
        return {"docs": n, "reused": True}

    def all_docs():
        pats = [os.path.join(wet_dir, "*.warc.wet.gz"),
                os.path.join(wet_dir, "*.wet.gz")]
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
    with open(tmp, encoding="utf-8") as fin, open(docs_path, "w",
                                                  encoding="utf-8") as fout:
        for line in fin:
            if rx.search(line):
                continue
            d = json.loads(line)
            if d["url"] in seen:
                continue
            seen.add(d["url"])
            fout.write(line)
            kept += 1
            if verbose and kept % 100_000 == 0:
                print(f"[ingest] url-dedup: {kept}", flush=True)
    os.remove(tmp)
    _log("ingest", t0, f" ({kept} docs -> {docs_path}; {stats})")
    return {"docs": kept, "stats": stats}


def stage_embed(docs_path: str, vecs_path: str, ids_path: str,
                batch: int = 256, force: bool = False,
                verbose: bool = False, show_progress: bool = False) -> tuple:
    """docs.jsonl -> vecs.npy + ids.json (real Embedder; GPU box only)."""
    import numpy as np
    from exa_home.embed import Embedder
    t0 = time.perf_counter()
    if _skip(vecs_path, force) and _skip(ids_path, force):
        v = np.load(vecs_path, mmap_mode="r")
        if verbose:
            print(f"[embed] reusing {vecs_path} {v.shape}", flush=True)
        return v.shape, str(v.dtype)
    docs = [json.loads(l) for l in open(docs_path, encoding="utf-8")]
    emb = Embedder()
    if show_progress:
        # Per-batch ticks for long bulk runs (silent by default).
        parts = []
        for i in range(0, len(docs), batch):
            parts.append(emb.encode_docs(
                [d["text"][:12000] for d in docs[i:i + batch]]))
            done = min(i + batch, len(docs))
            print(f"[embed] {done}/{len(docs)} docs", flush=True)
        vecs = np.vstack(parts)
    else:
        vecs = np.vstack([emb.encode_docs(
            [d["text"][:12000] for d in docs[i:i + batch]])
            for i in range(0, len(docs), batch)])
    np.save(vecs_path, vecs)
    with open(ids_path, "w") as f:
        json.dump([d["id"] for d in docs], f)
    _log("embed", t0, f" ({vecs.shape} {vecs.dtype})")
    return vecs.shape, str(vecs.dtype)


def stage_index(vecs_path: str, ids_path: str, docs_path: str, out: str,
                k: int, sample: int = 500_000, use_filter: bool = True,
                force: bool = False, verbose: bool = False) -> dict:
    """vecs + ids + docs -> versioned index/ (centroids, codes, lists, store)."""
    from scripts.train_centroids import main as train_main
    from scripts.build_index import main as build_main
    from exa_home.index_format import verify_manifest
    t0 = time.perf_counter()
    cent_path = os.path.join(os.path.dirname(out.rstrip("/")) or ".",
                             "centroids.npy")
    if _skip(os.path.join(out, "manifest.json"), force):
        m = verify_manifest(out)
        if verbose:
            print(f"[index] reusing {out} "
                  f"({m['n_docs']} docs, K={m['n_centroids']})", flush=True)
        return m
    train_main(vecs_path, k, sample, cent_path, verbose=verbose)
    build_main(vecs_path, ids_path, cent_path, docs_path, out,
               filter=use_filter, verbose=verbose)
    m = verify_manifest(out)
    _log("index", t0, f" ({m['n_docs']} docs, K={m['n_centroids']} -> {out})")
    return m


def stage_queries(docs_path: str, vecs_path: str, ids_path: str, nq: int,
                  queries_path: str = "q.jsonl", gt_path: str = "gt.jsonl",
                  force: bool = False, verbose: bool = False) -> dict:
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
            print(f"[queries] reusing {queries_path} + {gt_path}",
                  flush=True)
        return {"reused": True}
    docs = [json.loads(l) for l in open(docs_path, encoding="utf-8")]
    with open(queries_path, "w") as f:
        for d in docs[:nq]:
            f.write(json.dumps({"query": d["text"][:200]}) + "\n")
    subprocess.run([sys.executable, "scripts/ground_truth.py",
                    "--vecs", vecs_path, "--ids", ids_path,
                    "--queries", queries_path,
                    "--out", gt_path], check=True)
    _log("queries", t0, f" ({nq} queries -> {queries_path}, {gt_path})")
    return {"nq": nq}


def cmd_toy(a) -> int:
    stage_fetch("CC-MAIN-2026-34", 4, "wet_urls.txt",
                force=a.force, verbose=a.verbose)
    stage_download("wet_urls.txt", "data/wet", workers=a.workers,
                   force=a.force, verbose=a.verbose)
    stage_ingest("data/wet", "docs.jsonl",
                 force=a.force, verbose=a.verbose)
    stage_embed("docs.jsonl", "vecs.npy", "ids.json", batch=a.batch,
                force=a.force, verbose=a.verbose,
                show_progress=a.verbose)
    stage_index("vecs.npy", "ids.json", "docs.jsonl", "index/",
                k=1024, use_filter=True,
                force=a.force, verbose=a.verbose)
    stage_queries("docs.jsonl", "vecs.npy", "ids.json", 200,
                  force=a.force, verbose=a.verbose)
    print("toy pipeline complete: index/ + q.jsonl + gt.jsonl ready "
          "(run the gate: scripts/e2e_latency.py ...)")
    return 0


def cmd_full(a) -> int:
    tag = f"{a.wet_files}wet"
    urls = f"wet_urls_{tag}.txt"
    wet_dir = f"data/wet_{tag}"
    stage_fetch("CC-MAIN-2026-34", a.wet_files, urls,
                force=a.force, verbose=a.verbose)
    stage_download(urls, wet_dir, workers=a.workers,
                   force=a.force, verbose=a.verbose)
    docs = f"docs_{tag}.jsonl"
    vecs = f"vecs_{tag}.npy"
    ids = f"ids_{tag}.json"
    stage_ingest(wet_dir, docs, force=a.force, verbose=a.verbose)
    stage_embed(docs, vecs, ids, batch=a.batch,
                force=a.force, verbose=a.verbose,
                show_progress=True)
    stage_index(vecs, ids, docs, "index/", k=a.k, use_filter=True,
                force=a.force, verbose=a.verbose)
    stage_queries(docs, vecs, ids, 1000, force=a.force, verbose=a.verbose)
    print(f"full pipeline complete ({a.wet_files} WET, K={a.k}): "
          "index/ + q.jsonl + gt.jsonl ready")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="redo stages even if outputs exist")
    ap.add_argument("--verbose", action="store_true",
                    help="per-stage progress output")
    ap.add_argument("--workers", type=int, default=16,
                    help="download workers")
    ap.add_argument("--batch", type=int, default=256,
                    help="embed batch size")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="WET dir -> docs.jsonl")
    p.add_argument("--wet", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=lambda a: stage_ingest(a.wet, a.out, a.force,
                                             a.verbose))

    p = sub.add_parser("embed", help="docs.jsonl -> vecs.npy + ids.json")
    p.add_argument("--docs", required=True); p.add_argument("--out", required=True)
    p.add_argument("--ids", required=True)
    p.set_defaults(fn=lambda a: stage_embed(a.docs, a.out, a.ids, a.batch,
                                            a.force, a.verbose, a.verbose))

    p = sub.add_parser("index", help="assemble versioned index/")
    p.add_argument("--vecs", required=True); p.add_argument("--ids", required=True)
    p.add_argument("--docs", required=True); p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--no-filter", action="store_true")
    p.set_defaults(fn=lambda a: stage_index(a.vecs, a.ids, a.docs, a.out,
                                            a.k, use_filter=not a.no_filter,
                                            force=a.force,
                                            verbose=a.verbose))

    p = sub.add_parser("queries", help="ground truth + gate queries")
    p.add_argument("--docs", required=True); p.add_argument("--vecs", required=True)
    p.add_argument("--ids", required=True); p.add_argument("--nq", type=int,
                                                           default=200)
    p.set_defaults(fn=lambda a: stage_queries(a.docs, a.vecs, a.ids, a.nq,
                                              force=a.force,
                                              verbose=a.verbose))

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
