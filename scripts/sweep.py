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
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))


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
    """Tier-0 backends: HashEmbedder + numpy brute-force ANN + mock rerank."""
    from fixtures.synth import make_docs  # noqa: E402 (tests/ on sys.path)

    from exa_home.embed import HashEmbedder
    from exa_home.rerank import MockReranker
    from exa_home.store import ContentStore
    from scripts.e2e_latency import NumpyFakeANN

    docs = make_docs(n_docs, seed=seed)
    store = ContentStore(tempfile.mkdtemp(prefix="sweep-store-"))
    for d in docs:
        store.put(d)
    embedder = HashEmbedder(dim=256)
    mat = embedder.encode_docs([d["text"] for d in docs])
    ann = NumpyFakeANN([d["id"] for d in docs], mat)
    return {"embedder": embedder, "ann": ann,
            "reranker": MockReranker(),
            "reranker_factory": lambda pt, b: MockReranker(),
            "store": store}


def build_real_backends(index_dir: str, store_dir: str | None,
                        max_query_tokens: int = 32) -> dict:
    """Cloud-only backends. Lazy imports: ann_core + models need the box.

    The reranker arrives as a FACTORY (cached per combo in run_grid):
    rebuilding per (pair-tokens, batch) combo reloads the ~90MB model,
    so instances are shared across the nprobe x top_coarse plane.
    """
    from exa_home.embed import Embedder
    from exa_home.rerank import Reranker
    from exa_home.store import ContentStore

    from ann_core import IvfIndex  # noqa: E402 (maturin develop on-box)

    def _factory(max_pair_tokens: int, batch_size: int) -> Reranker:
        return Reranker(max_pair_tokens=max_pair_tokens,
                        batch_size=batch_size,
                        max_query_tokens=max_query_tokens)

    return {"embedder": Embedder(),
            "ann": IvfIndex.load(index_dir),
            "reranker": _factory(64, 32),
            "reranker_factory": _factory,
            "store": ContentStore(store_dir or os.path.join(index_dir, "store"))}


def _recall_at_10(gt_rows: list[dict], pred_rows: list[dict]) -> float:
    pr = {r["query_id"]: r["top10"] for r in pred_rows}
    recs = [len(set(g["top10"]) & set(pr[g["query_id"]])) / 10 for g in gt_rows]
    return sum(recs) / len(recs)


def run_grid(backends: dict, queries: list[dict],
             gt_rows: list[dict] | None,
             nprobes: list[int], top_coarses: list[int], k: int,
             out_dir: str, ledger_path: str, budget_ms: float,
             logger: logging.Logger | None = None,
             pair_tokens_list: list[int] | None = None,
             batch_sizes: list[int] | None = None) -> list[dict]:
    """Run the full grid. Returns the ledger rows (also appended to file)."""
    from exa_home.orch import build_search_dag, run_search
    from scripts.e2e_latency import append_ledger, pct, run_latency

    os.makedirs(out_dir, exist_ok=True)
    if logger is None:
        logger = logging.getLogger(f"sweep-{os.path.basename(out_dir)}")
        logger.setLevel(logging.INFO)
    log = logger
    # run_grid always owns its file log (idempotent: one handler per path).
    want = os.path.join(out_dir, "sweep.log")
    if not any(isinstance(h, logging.FileHandler) and
               getattr(h, "baseFilename", "") == want
               for h in log.handlers):
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(want)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    pair_tokens_list = pair_tokens_list or [64]
    batch_sizes = batch_sizes or [32]
    combos = [{"nprobe": np, "top_coarse": tc,
               "max_pair_tokens": pt, "batch_size": b}
              for np in nprobes for tc in top_coarses
              for pt in pair_tokens_list for b in batch_sizes]
    with open(os.path.join(out_dir, "grid.json"), "w") as f:
        json.dump({"combos": combos, "k": k, "budget_ms": budget_ms,
                   "n_queries": len(queries)}, f, indent=2)
    factory = backends.get("reranker_factory",
                           lambda pt, b: backends["reranker"])
    rr_cache: dict = {}
    rows = []
    for c in combos:
        slug = (f"np{c['nprobe']}_tc{c['top_coarse']}"
                f"_pt{c['max_pair_tokens']}_b{c['batch_size']}")
        key = (c["max_pair_tokens"], c["batch_size"])
        if key not in rr_cache:
            rr_cache[key] = factory(*key)
        dag = build_search_dag(backends["embedder"], backends["ann"],
                               rr_cache[key], backends["store"],
                               top_coarse=c["top_coarse"], nprobe=c["nprobe"])
        stats = run_latency(dag, queries, k=k)
        pred_rows = []
        for qid, q in enumerate(queries):
            res = run_search(dag, q["query"], q.get("filters"), top_k=k)
            pred_rows.append({"query_id": qid,
                              "top10": [r["id"] for r in res["results"]]})
        with open(os.path.join(out_dir, f"pred_{slug}.jsonl"), "w") as f:
            for r in pred_rows:
                f.write(json.dumps(r) + "\n")
        recall = _recall_at_10(gt_rows, pred_rows) if gt_rows else None
        row = append_ledger(ledger_path, {"sweep": "knob-grid", **c},
                            stats, recall=recall)
        p50, viol = pct(stats["total"], 0.5), len(stats["violations"])
        gate = "PASS" if p50 <= budget_ms and not viol else \
            ("OVER BUDGET" if p50 > budget_ms else "VIOLATIONS")
        rec = f" recall={recall:.4f}" if recall is not None else ""
        log.info(f"{slug}: p50={p50:.2f}ms {gate}{rec} "
                 f"({stats['n']} queries, {viol} violations)")
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nprobe", type=int, nargs="+", default=[8])
    ap.add_argument("--top-coarse", type=int, nargs="+", default=[50])
    ap.add_argument("--max-pair-tokens", type=int, nargs="+", default=[64])
    ap.add_argument("--batch-size", type=int, nargs="+", default=[32])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--budget-ms", type=float, default=100.0)
    ap.add_argument("--out", default="sweeps/")
    ap.add_argument("--ledger", default="runs.jsonl")
    ap.add_argument("--synth-docs", type=int, default=0)
    ap.add_argument("--synth-queries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--max-query-tokens", type=int, default=32)
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
        queries = [{"query": f"synthetic topic {i} about neural search",
                    "filters": {}} for i in range(a.synth_queries)]
        gt_rows = None
    else:
        if not a.index or not a.queries:
            print("sweep: --index + --queries required (or --synth-docs)",
                  file=sys.stderr)
            return 2
        from scripts.e2e_latency import load_queries
        backends = build_real_backends(a.index, None, a.max_query_tokens)
        queries = load_queries(a.queries)
        gt_rows = [json.loads(l) for l in open(a.gt)] if a.gt else None
    logger.info(f"grid: nprobe={a.nprobe} top_coarse={a.top_coarse} "
                f"pair_tokens={a.max_pair_tokens} batch={a.batch_size} "
                f"k={a.k} queries={len(queries)} -> {out_dir}")
    run_grid(backends, queries, gt_rows, a.nprobe, a.top_coarse, a.k,
             out_dir, a.ledger, a.budget_ms, logger,
             pair_tokens_list=a.max_pair_tokens, batch_sizes=a.batch_size)
    logger.info(f"sweep done in {time.perf_counter() - t0:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
