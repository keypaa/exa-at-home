#!/usr/bin/env python3
"""Knob-grid benchmark: sweep ANN/serving params with full reproducibility.

Each grid point runs the latency gate + serving-stack predictions + recall
(where ground truth exists) and banks one ledger row carrying the FULL
config — no more "which row was nprobe=4" archaeology.

Reproducibility contract (Tier-0 tested):
- timestamped out dir (`sweeps/YYYYMMDD-HHMMSS/`, `-1` suffixed on clash):
  NEVER overwritten — `pred_<slug>.jsonl` per combo, `grid.json` with the
  exact grid, `sweep.log` with every line also printed to stdout.
- the ledger (`--ledger`, default `runs.jsonl`) is APPEND-only; each row
  holds config + p50/p99 + stages + breakdown + recall_at_10 (null when
  gt is unavailable, e.g. synth).

Grid dimensions: nprobe x top_coarse x max-pair-tokens x batch-size.
Backends are built once and shared; reranker instances are constructed
per (pair-tokens, batch) combo and cached across the nprobe x top_coarse
plane (each build reloads the ~90MB model — seconds, logged).

Usage (box):
    python scripts/sweep.py --index index/ --queries q.jsonl --gt gt.jsonl \\
        --nprobe 1 2 4 8 16 --top-coarse 50 --max-pair-tokens 32 64 128 \\
        --out sweeps/ --ledger runs.jsonl
Usage (Tier-0):
    python scripts/sweep.py --synth-docs 50 --synth-queries 3 \\
        --nprobe 1 2 --top-coarse 10 20 --out /tmp/s
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from exa_home.config import OPERATING_POINT

# The ONE home of the recall@10 formula (scripts/measure_recall.py).
from scripts.measure_recall import recall_at_10 as _recall_at_10


def _timestamp() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def fresh_out_dir(base: str) -> str:
    """Timestamped out dir; `-N` suffix on clash — never reuse."""
    cand, i = os.path.join(base, _timestamp()), 0
    while os.path.exists(cand):
        i += 1
        cand = f"{os.path.join(base, _timestamp())}-{i}"
    return cand


def build_synth_backends(n_docs: int = 200, seed: int = 0) -> dict:
    """Tier-0 backends. Thin delegate: the wiring lives exactly once in
    exa_home.synth (kept here so sweep call sites don't churn)."""
    from exa_home.synth import build_synth_backends as _build

    return _build(n_docs, seed)


def build_real_backends(
    index_dir: str, store_dir: str | None, max_query_tokens: int = 32
) -> dict:
    """Cloud-only backends. Lazy imports: ann_core + models need the box.

    The reranker arrives as a FACTORY (cached per combo in run_grid):
    rebuilding per (pair-tokens, batch) combo reloads the ~90MB model,
    so instances are shared across the nprobe x top_coarse plane.
    """
    from ann_core import IvfIndex

    from exa_home.embed import Embedder
    from exa_home.rerank import Reranker
    from exa_home.store import ContentStore

    def _factory(max_pair_tokens: int, batch_size: int) -> Reranker:
        return Reranker(
            max_pair_tokens=max_pair_tokens,
            batch_size=batch_size,
            max_query_tokens=max_query_tokens,
        )

    return {
        "embedder": Embedder(),
        "ann": IvfIndex.load(index_dir),
        "reranker": _factory(64, 32),
        "reranker_factory": _factory,
        "store": ContentStore(store_dir or os.path.join(index_dir, "store")),
    }


def run_grid(
    backends: dict,
    queries: list[dict],
    gt_rows: list[dict] | None,
    nprobes: list[int],
    top_coarses: list[int],
    k: int,
    out_dir: str,
    ledger_path: str,
    budget_ms: float,
    logger: logging.Logger | None = None,
    pair_tokens_list: list[int] | None = None,
    batch_sizes: list[int] | None = None,
) -> list[dict]:
    """Run the full grid. Returns the ledger rows (also appended to file)."""
    from exa_home.orch import build_search_dag
    from scripts.e2e_latency import append_ledger, pct, run_latency

    os.makedirs(out_dir, exist_ok=True)
    if logger is None:
        logger = logging.getLogger(f"sweep-{os.path.basename(out_dir)}")
        logger.setLevel(logging.INFO)
    log = logger
    # run_grid always owns its file log (idempotent: one handler per path).
    want = os.path.join(out_dir, "sweep.log")
    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == want
        for h in log.handlers
    ):
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(want)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    pair_tokens_list = pair_tokens_list or [64]
    batch_sizes = batch_sizes or [32]
    combos = [
        {"nprobe": np, "top_coarse": tc, "max_pair_tokens": pt, "batch_size": b}
        for np in nprobes
        for tc in top_coarses
        for pt in pair_tokens_list
        for b in batch_sizes
    ]
    with open(os.path.join(out_dir, "grid.json"), "w") as f:
        json.dump(
            {
                "combos": combos,
                "k": k,
                "budget_ms": budget_ms,
                "n_queries": len(queries),
            },
            f,
            indent=2,
        )
    factory = backends.get("reranker_factory", lambda pt, b: backends["reranker"])
    rr_cache: dict = {}
    rows = []
    for c in combos:
        slug = (
            f"np{c['nprobe']}_tc{c['top_coarse']}"
            f"_pt{c['max_pair_tokens']}_b{c['batch_size']}"
        )
        key = (c["max_pair_tokens"], c["batch_size"])
        if key not in rr_cache:
            rr_cache[key] = factory(*key)
        dag = build_search_dag(
            backends["embedder"],
            backends["ann"],
            rr_cache[key],
            backends["store"],
            top_coarse=c["top_coarse"],
            nprobe=c["nprobe"],
        )
        # ONE pass over the query set per grid point: run_latency collects
        # the per-query result ids on the way, so the predictions are the
        # ids of the very same searches that were timed. (The old code ran
        # every query TWICE per grid point — once for the latency, once
        # for the predictions — doubling the GPU/CPU time of every sweep.)
        stats = run_latency(dag, queries, k=k)
        pred_rows = [
            {"query_id": qid, "top10": top10}
            for qid, top10 in enumerate(stats["top_ids"])
        ]
        with open(os.path.join(out_dir, f"pred_{slug}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in pred_rows)
        recall, skipped = _recall_at_10(gt_rows, pred_rows) if gt_rows else (None, 0)
        if skipped:
            print(
                f"sweep: warning: {skipped} ground-truth row(s) have no "
                f"prediction — skipped (recall over the comparable queries)",
                file=sys.stderr,
            )
        cfg = {"sweep": "knob-grid", **c}
        if gt_rows is not None:
            cfg["gt_skipped"] = skipped
        row = append_ledger(ledger_path, cfg, stats, recall=recall)
        p50, viol = pct(stats["total"], 0.5), len(stats["violations"])
        gate = (
            "PASS"
            if p50 <= budget_ms and not viol
            else ("OVER BUDGET" if p50 > budget_ms else "VIOLATIONS")
        )
        rec = f" recall={recall:.4f}" if recall is not None else ""
        skip_note = f" ({skipped} gt skipped)" if skipped else ""
        log.info(
            f"{slug}: p50={p50:.2f}ms {gate}{rec}{skip_note} "
            f"({stats['n']} queries, {viol} violations)"
        )
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--nprobe", type=int, nargs="+", default=[OPERATING_POINT["nprobe"]]
    )
    ap.add_argument(
        "--top-coarse", type=int, nargs="+", default=[OPERATING_POINT["top_coarse"]]
    )
    ap.add_argument(
        "--max-pair-tokens",
        type=int,
        nargs="+",
        default=[OPERATING_POINT["max_pair_tokens"]],
    )
    ap.add_argument(
        "--batch-size", type=int, nargs="+", default=[OPERATING_POINT["batch_size"]]
    )
    ap.add_argument("--k", type=int, default=OPERATING_POINT["k"])
    ap.add_argument("--budget-ms", type=float, default=OPERATING_POINT["budget_ms"])
    ap.add_argument("--out", default="sweeps/")
    ap.add_argument("--ledger", default="runs.jsonl")
    ap.add_argument("--synth-docs", type=int, default=0)
    ap.add_argument("--synth-queries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--gt", default=None)
    ap.add_argument(
        "--max-query-tokens", type=int, default=OPERATING_POINT["max_query_tokens"]
    )
    a = ap.parse_args(argv)

    out_dir = fresh_out_dir(a.out)
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(f"sweep-{os.path.basename(out_dir)}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    t0 = time.perf_counter()
    if a.synth_docs:
        backends = build_synth_backends(a.synth_docs, a.seed)
        queries = [
            {"query": f"synthetic topic {i} about neural search", "filters": {}}
            for i in range(a.synth_queries)
        ]
        gt_rows = None
    else:
        if not a.index or not a.queries:
            print(
                "sweep: --index + --queries required (or --synth-docs)", file=sys.stderr
            )
            return 2
        from scripts.e2e_latency import load_queries

        backends = build_real_backends(a.index, None, a.max_query_tokens)
        queries = load_queries(a.queries)
        if a.gt:
            from scripts.jsonl import read_jsonl

            gt_rows = list(read_jsonl(a.gt))
        else:
            gt_rows = None
    logger.info(
        f"grid: nprobe={a.nprobe} top_coarse={a.top_coarse} "
        f"pair_tokens={a.max_pair_tokens} batch={a.batch_size} "
        f"k={a.k} queries={len(queries)} -> {out_dir}"
    )
    run_grid(
        backends,
        queries,
        gt_rows,
        a.nprobe,
        a.top_coarse,
        a.k,
        out_dir,
        a.ledger,
        a.budget_ms,
        logger,
        pair_tokens_list=a.max_pair_tokens,
        batch_sizes=a.batch_size,
    )
    logger.info(f"sweep done in {time.perf_counter() - t0:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
