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

from exa_home.config import (
    CENTROID_SAMPLE,
    DOWNLOAD_WORKERS,
    EMBED_BATCH,
    EMBED_CHUNK,
    K_TOY,
)

WET_PATHS_URL = "https://data.commoncrawl.org/crawl-data/{crawl}/wet.paths.gz"
BASE = "https://data.commoncrawl.org"
# The WET file naming: Common Crawl ships `*.warc.wet.gz` (the sibling of
# the WARC segment) and `*.wet.gz`. ONE place, so the stages that COUNT
# the WET files and the stage that INGESTS them cannot drift apart.
WET_GLOBS = ("*.warc.wet.gz", "*.wet.gz")


def wet_files(wet_dir: str) -> list[str]:
    """Every WET file in `wet_dir` (both naming conventions), sorted.

    A `*.warc.wet.gz` file matches BOTH patterns, so the result is
    de-duplicated: the old `glob + glob` count counted such files twice
    and could then skip a download whose files were still missing.
    """
    seen: dict[str, None] = {}
    for pattern in WET_GLOBS:
        for path in glob.glob(os.path.join(wet_dir, pattern)):
            seen.setdefault(path, None)
    return sorted(seen)


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
        have = len(wet_files(out_dir))
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
        files = wet_files(wet_dir)
        if verbose:
            print(f"[ingest] {len(files)} WET files", flush=True)
        for i, p in enumerate(files, 1):
            if verbose:
                print(f"[ingest] WET file {i}/{len(files)}: {os.path.basename(p)}", flush=True)
            yield from read_wet(p)

    tmp = docs_path + ".tmp"
    stats = write_docs_jsonl(all_docs(), tmp)
    # URL-second dedup + subset exclusion (digest-first is in write_docs_jsonl).
    import re

    from scripts.jsonl import read_jsonl

    rx = re.compile(r"robotstxt|crawldiagnostics", re.IGNORECASE)
    seen: set[str] = set()
    kept = 0
    excluded = 0
    dup_urls = 0
    with open(docs_path, "w", encoding="utf-8") as fout:
        for d in read_jsonl(tmp):
            # The exclusion applies to the URL (and an optional `subset`
            # field) — NEVER to the whole JSON line: the line carries the
            # page TEXT, so a page whose body merely mentions robots.txt
            # was silently dropped from the corpus.
            url = d.get("url", "")
            subset = d.get("subset", "")
            if rx.search(url) or (subset and rx.search(subset)):
                excluded += 1
                continue
            if url in seen:
                dup_urls += 1
                continue
            seen.add(url)
            fout.write(json.dumps(d) + "\n")
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
    batch: int = EMBED_BATCH,
    chunk: int = EMBED_CHUNK,
    force: bool = False,
    verbose: bool = False,
    show_progress: bool = False,
    resume: bool = True,
) -> tuple:
    """Resumable wrapper around scripts.embed_resumable.run_embed (item 2)."""
    from scripts.embed_resumable import run_embed

    return run_embed(
        docs_path,
        vecs_path,
        ids_path,
        batch=batch,
        chunk=chunk,
        force=force,
        verbose=verbose,
        show_progress=show_progress,
        resume=resume,
    )


def stage_index(
    vecs_path: str,
    ids_path: str,
    docs_path: str,
    out: str,
    k: int,
    sample: int = CENTROID_SAMPLE,
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
    embedder=None,
) -> dict:
    """docs/vecs/ids -> q.jsonl + gt.jsonl (gate queries + ground truth).

    ALIGNED by construction (M5 lesson, commit 1445aa0): q.jsonl is
    written first from docs[:nq], then ground_truth --queries embeds
    those exact texts — gt query k is always pred query k. The old
    --nq random-vector path is never used here (different query sets
    score recall@10 = 0.0000 by construction).

    The ground truth is computed IN-PROCESS (scripts.ground_truth is
    imported, not shelled out): one interpreter instead of two, and the
    stage becomes testable without a subprocess.

    `embedder`: injectable query embedder (default: the real one) so the
    Tier-0 suite can build a ground truth without loading a 335M-param
    model.
    """
    t0 = time.perf_counter()
    if _skip(queries_path, force) and _skip(gt_path, force):
        if verbose:
            print(f"[queries] reusing {queries_path} + {gt_path}", flush=True)
        return {"reused": True}
    from scripts.ground_truth import ground_truth
    from scripts.jsonl import read_jsonl

    # Only the first `nq` docs are needed: read them and stop. The old code
    # materialised the WHOLE corpus as a list of dicts to use docs[:nq].
    docs = list(itertools.islice(read_jsonl(docs_path), nq))
    if len(docs) < nq:
        print(
            f"[queries] warning: corpus has only {len(docs)} docs, {nq} requested",
            file=sys.stderr,
            flush=True,
        )
    with open(queries_path, "w") as f:
        f.writelines(json.dumps({"query": d["text"][:200]}) + "\n" for d in docs)
    ground_truth(vecs_path, ids_path, gt_path, queries=queries_path, embedder=embedder)
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
        k=K_TOY,
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
    ap.add_argument(
        "--workers", type=int, default=DOWNLOAD_WORKERS, help="download workers"
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=EMBED_BATCH,
        help="GPU batch size for bulk embedding",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=EMBED_CHUNK,
        help="docs held in RAM per flush (streaming boundary)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="WET dir -> docs.jsonl")
    p.add_argument("--wet", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=lambda a: stage_ingest(a.wet, a.out, a.force, a.verbose))

    p = sub.add_parser("embed", help="docs.jsonl -> vecs.npy + ids.json")
    p.add_argument("--docs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(
        fn=lambda a: stage_embed(
            a.docs, a.out, a.ids, a.batch, a.chunk, a.force, a.verbose, a.verbose,
            resume=a.resume,
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
