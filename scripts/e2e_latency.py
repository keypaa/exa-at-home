# scripts/e2e_latency.py
"""End-to-end p50/p99 with profile waterfall check.

Tier-0 safe: imports only pure-Python pieces (orch/embed/store/rerank).
The ANN backend arrives INJECTED (same fake-backend pattern as Task 10's
tests) — `ann_core` is NEVER imported at module top. The real-index path
(`--ann ivf`) lazy-imports `ann_core` + the ST model inside its builder, so
it only runs on the cloud box (maturin + GPU + HF download).

Exit codes: 0 = pass; 1 = total p50 over budget; 2 = usage error or
waterfall-integrity failure (profile stage-sum != wall).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

# Spec §7 profile keys from run_search (filter_ms is in-scan, always 0.0 —
# excluded from the waterfall sum so the integrity check is unchanged).
STAGE_ORDER = ("embed_ms", "ann_ms", "rerank_ms", "snippet_ms")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def load_queries(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = d.get("query", d.get("text", ""))
            if not q:
                raise ValueError(f"{path}:{ln}: query line needs a 'query' key")
            out.append({"query": q, "filters": d.get("filters") or {}})
    if not out:
        raise ValueError(f"{path}: no queries found")
    return out


class NumpyFakeANN:
    """Tier-0 ANN stand-in: exact brute-force dot over doc vectors.

    Same injected-backend shape as Task 10's FakeANN
    (`search(q, nprobe, top_k, **filters) -> (ids, scores)`), but scores are
    real numpy dots so the retrieve span is representative. Filters are
    accepted and ignored (documented): the fake corpus has no filter index.
    """

    def __init__(self, ids: list[str], mat):
        self.ids = ids
        self._mat = mat

    def search(self, q, nprobe=8, top_k=200, **filters):
        import numpy as np

        scores = self._mat @ np.asarray(q, dtype=np.float32)
        k = min(top_k, len(self.ids))
        idx = np.argsort(scores, kind="stable")[::-1][:k]
        return [self.ids[i] for i in idx], [float(scores[i]) for i in idx]


def build_synth_dag(n_docs: int = 500, seed: int = 0,
                    top_coarse: int = 200, nprobe: int = 8):
    """500-doc synth index from existing pieces: HashEmbedder + ContentStore
    + NumpyFakeANN + MockReranker, wired through build_search_dag."""
    from fixtures.synth import make_docs  # noqa: E402  (tests/ on sys.path)

    from exa_home.embed import HashEmbedder
    from exa_home.orch import build_search_dag
    from exa_home.rerank import MockReranker
    from exa_home.store import ContentStore

    docs = make_docs(n_docs, seed=seed)
    store = ContentStore(tempfile.mkdtemp(prefix="e2e-store-"))
    for d in docs:
        store.put(d)
    embedder = HashEmbedder(dim=256)
    mat = embedder.encode_docs([d["text"] for d in docs])
    ann = NumpyFakeANN([d["id"] for d in docs], mat)
    return build_search_dag(embedder, ann, MockReranker(), store,
                            top_coarse=top_coarse, nprobe=nprobe)


def build_real_dag(index_dir: str, store_dir: str | None, embedder_kind: str,
                   reranker_kind: str, top_coarse: int, nprobe: int,
                   max_pair_tokens: int = 160, batch_size: int = 128):
    """Cloud-only path: IvfIndex over a built index/ dir. Lazy imports."""
    from exa_home.orch import build_search_dag
    from exa_home.store import ContentStore

    from ann_core import IvfIndex  # noqa: E402  (maturin develop on-box)

    if embedder_kind in ("arctic", "real"):  # "arctic" kept as alias (pre-switch runs)
        from exa_home.embed import Embedder
        embedder = Embedder()
    else:
        from exa_home.embed import HashEmbedder
        embedder = HashEmbedder(dim=256)
    if reranker_kind == "real":
        from exa_home.rerank import Reranker
        reranker = Reranker(max_pair_tokens=max_pair_tokens,
                            batch_size=batch_size)
    else:
        from exa_home.rerank import MockReranker
        reranker = MockReranker()
    ann = IvfIndex.load(index_dir)
    store = ContentStore(store_dir or os.path.join(index_dir, "store"))

    return build_search_dag(embedder, ann, reranker, store,
                            top_coarse=top_coarse, nprobe=nprobe)


def run_latency(dag, queries: list[dict], k: int) -> dict:
    """Run every query with profile=True; collect per-stage ms + totals."""
    from exa_home.orch import run_search

    per_stage = {s: [] for s in STAGE_ORDER}
    totals: list[float] = []
    violations: list[tuple] = []
    for i, q in enumerate(queries):
        res = run_search(dag, q["query"], q.get("filters"), top_k=k,
                         profile=True)
        prof = res["profile"]
        wall = prof["total_ms"]
        stage_sum = sum(prof[s] for s in STAGE_ORDER)
        tol = 0.2 * wall + 5.0
        if abs(stage_sum - wall) > tol:
            violations.append((i, stage_sum, wall))
        for s in STAGE_ORDER:
            per_stage[s].append(prof[s])
        totals.append(wall)
    return {"stages": per_stage, "total": totals,
            "violations": violations, "n": len(queries)}


def report(stats: dict, budget_ms: float) -> str:
    lines = [f"queries: {stats['n']}"]
    lines.append(f"{'stage':<10}{'p50 (ms)':>12}{'p99 (ms)':>12}")
    for s in STAGE_ORDER:
        xs = stats["stages"][s]
        lines.append(f"{s:<10}{pct(xs, 0.5):>12.3f}{pct(xs, 0.99):>12.3f}")
    t = stats["total"]
    lines.append(f"{'total':<10}{pct(t, 0.5):>12.3f}{pct(t, 0.99):>12.3f}")
    n_viol = len(stats["violations"])
    lines.append(f"waterfall integrity: "
                 f"{'FAIL ' + str(stats['violations'][:3]) if n_viol else 'OK'} "
                 f"({n_viol} violations, tol 20%+5ms)")
    lines.append(f"budget: total p50 {pct(t, 0.5):.3f}ms vs "
                 f"{budget_ms:.1f}ms -> "
                 f"{'OVER BUDGET' if pct(t, 0.5) > budget_ms else 'PASS'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ann", choices=["fake", "ivf"], default="fake")
    ap.add_argument("--index", default=None,
                    help="index/ dir (required for --ann ivf)")
    ap.add_argument("--store", default=None,
                    help="content store root (default <index>/store, ivf mode)")
    ap.add_argument("--queries", default=None, help="q.jsonl path")
    ap.add_argument("--synth-queries", type=int, default=20,
                    help="used when --queries is absent")
    ap.add_argument("--synth-docs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--budget-ms", type=float, default=100.0)
    ap.add_argument("--embedder", choices=["hash", "arctic", "real"], default="hash")
    ap.add_argument("--reranker", choices=["mock", "real"], default="mock")
    ap.add_argument("--nprobe", type=int, default=8)
    ap.add_argument("--top-coarse", type=int, default=200)
    ap.add_argument("--max-pair-tokens", type=int, default=160,
                    help="reranker pair truncation (real reranker only)")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="reranker batch size (real reranker only)")
    a = ap.parse_args(argv)

    try:
        if a.queries:
            queries = load_queries(a.queries)
        else:
            queries = [{"query": f"synthetic topic {i} about neural "
                                 f"search {i % 7}", "filters": {}}
                       for i in range(a.synth_queries)]
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"e2e_latency: {e}", file=sys.stderr)
        return 2
    if not queries:
        print("e2e_latency: no queries (--synth-queries must be >= 1 "
              "without --queries)", file=sys.stderr)
        return 2

    if a.ann == "fake":
        dag = build_synth_dag(n_docs=a.synth_docs, seed=a.seed,
                              top_coarse=a.top_coarse, nprobe=a.nprobe)
    else:
        if not a.index:
            print("e2e_latency: --index is required for --ann ivf",
                  file=sys.stderr)
            return 2
        try:
            dag = build_real_dag(a.index, a.store, a.embedder, a.reranker,
                                 a.top_coarse, a.nprobe,
                                 max_pair_tokens=a.max_pair_tokens,
                                 batch_size=a.batch_size)
        except ImportError as e:
            print(f"e2e_latency: cloud-only backend unavailable: {e}",
                  file=sys.stderr)
            return 2

    stats = run_latency(dag, queries, a.k)
    print(report(stats, a.budget_ms))
    if stats["violations"]:
        return 2
    if pct(stats["total"], 0.5) > a.budget_ms:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
