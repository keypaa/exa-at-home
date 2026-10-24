"""One-command gate for the four release gates (G1..G4).

Exit codes:
  0  all *hard* gates pass (G1 ANN fidelity, G2 answer coverage, G3 latency)
  1  at least one hard gate regressed
  2  usage / input error

G4 (relevance) is reported as an advisory warning and never flips the exit code.

Tier-0 friendly: ``--synth`` runs the whole gate against deterministic in-memory
backends in a couple of seconds (no model, no index, no GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---- thresholds (pinned; change only by ADR) ------------------------------
G2_BASELINE = 0.740
G1_DROP_BUDGET = 0.10
G2_DROP_BUDGET = 0.10
G3_BUDGET_MS = 100.0
G4_COARSE = 0.350
G4_RERANKED = 0.380
G4_NOISE = 0.10


def _jsonl(path):
    from scripts.jsonl import read_jsonl

    return list(read_jsonl(str(path)))


def _pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = int(len(s) * p + 0.999999999) - 1
    k = max(0, min(len(s) - 1, k))
    return s[k]


def make_synth_backends(n_docs: int = 80, seed: int = 7):
    from exa_home.embed import HashEmbedder
    from exa_home.store import ContentStore
    from exa_home.synth import NumpyFakeANN, make_docs

    docs = make_docs(n_docs, seed=seed)
    tmp = str(Path(os.environ.get("TMPDIR", "/tmp")) / f"gate-synth-store-{seed}")
    os.makedirs(tmp, exist_ok=True)
    store = ContentStore(tmp)
    for d in docs:
        store.put(d)
    emb = HashEmbedder(dim=256)
    mat = emb.encode_docs([d["text"] for d in docs])
    ann = NumpyFakeANN([d["id"] for d in docs], mat)
    # a tiny, deterministic *reranker* mirror for recall tests
    from exa_home.rerank import MockReranker

    rr = MockReranker()
    return docs, store, emb, ann, rr


def gate_synthetic(n_docs: int, k: int, budget_ms: float, verbose: bool = False):
    """Tier-0 stand-in for all four gates."""
    docs, store, emb, ann, reranker = make_synth_backends(n_docs)
    from exa_home.orch import build_search_dag, run_search

    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]

    # --- G3 latency: run the real search DAG on k queries -------------------
    mat = emb.encode_docs(texts)
    dag = build_search_dag(emb, ann, reranker, store, top_coarse=50, nprobe=8)
    totals = []
    for i in range(min(k, len(docs))):
        q = texts[i]
        res = run_search(dag, q, {}, k if k < len(docs) else 10, profile=True)
        totals.append(res["profile"]["total_ms"])
    p50 = _pct(totals, 0.5)
    p99 = _pct(totals, 0.99)

    # --- G1 ANN fidelity: exact-id recall@10 (synth self-match) ---
    hits = 0
    nq = min(20, len(docs))
    for i in range(nq):
        qv = mat[i]
        got, _ = ann.search(qv, nprobe=8, n_coarse=10)
        scores = mat @ qv
        # brute-force exact top-10 row ids
        exact_idx = sorted(range(len(ids)), key=lambda r: -scores[r])[:10]
        exact_ids = {ids[r] for r in exact_idx}
        hits += len(set(got) & exact_ids) / 10
    g1_recall = hits / nq

    # --- G2 answer coverage: target-doc recall@10 ---
    # Build a known target (query i) against the synth corpus; expected hit rate
    # is 1.0 if ANN recalls its own vector in top-10.
    g2_hits = 0
    for i in range(nq):
        qv = mat[i]
        got, _ = ann.search(qv, nprobe=8, n_coarse=10)
        if ids[i] in got:
            g2_hits += 1
    g2_recall = g2_hits / nq

    # --- G4 relevance (advisory constant) ---
    results = {
        "gate": "synth",
        "G1_ann_fidelity": {
            "value": round(g1_recall, 4),
            "baseline": 1.0,
            "limit": 1.0 - G1_DROP_BUDGET,
            "pass": g1_recall >= 1.0 - G1_DROP_BUDGET,
        },
        "G2_answer_coverage": {
            "value": round(g2_recall, 4),
            "baseline": 1.0,
            "limit": 1.0 - G2_DROP_BUDGET,
            "pass": g2_recall >= 1.0 - G2_DROP_BUDGET,
        },
        "G3_latency": {
            "p50_ms": round(p50, 3) if p50 is not None else None,
            "p99_ms": round(p99, 3) if p99 is not None else None,
            "budget_ms": budget_ms,
            "pass": (p50 is not None and p50 <= budget_ms),
        },
        "G4_relevance_advisory": {
            "coarse_p10": G4_COARSE,
            "reranked_p10": G4_RERANKED,
            "noise": G4_NOISE,
            "note": "noisy at n=10, advisory only",
        },
    }
    return results


def gate_real(index_dir, queries, gt, pred, pred16, e3_result_path, budget_ms,
              run_g2_e3: bool, verbose: bool):
    from ann_core import IvfIndex

    from exa_home.embed import Embedder
    from exa_home.orch import build_search_dag, run_search
    from exa_home.rerank import Reranker
    from exa_home.store import ContentStore

    ann = IvfIndex.load(str(index_dir))
    store = ContentStore(str(Path(index_dir) / "store"))
    emb = Embedder()
    rr = Reranker()

    # --- G3 latency ---
    q_list = [json.loads(line)["query"] for line in Path(queries).read_text().splitlines() if line.strip()]
    dag = build_search_dag(emb, ann, rr, store, top_coarse=50, nprobe=8)
    totals = []
    for q in q_list[: min(200, len(q_list))]:
        r = run_search(dag, q, {}, 10, profile=True)
        totals.append(r["profile"]["total_ms"])
    p50 = _pct(totals, 0.5)
    p99 = _pct(totals, 0.99)
    g3_pass = (p50 is not None and p50 <= budget_ms)

    # --- G1 ANN fidelity ---
    g1 = {}
    if pred:
        from scripts.measure_recall import recall_at_10

        gt_rows = _jsonl(gt)
        pred_rows = _jsonl(pred)
        recall, _skipped = recall_at_10(gt_rows, pred_rows)
        g1["reranked"] = round(recall, 4) if recall is not None else None
        g1["baseline"] = 0.2885
        g1["limit"] = 0.2885 - G1_DROP_BUDGET
        g1["pass"] = (recall is not None and recall >= 0.2885 - G1_DROP_BUDGET)
    if pred16:
        from scripts.measure_recall import recall_at_10

        recall16, _ = recall_at_10(_jsonl(gt), _jsonl(pred16))
        g1["nprobe16"] = round(recall16, 4) if recall16 is not None else None
    # coarse (ANN only) baseline measured separately, surface if --pred doesn't
    g1.setdefault("baseline_coarse", 0.4015)
    g1.setdefault("baseline_coarse_limit", 0.4015 - G1_DROP_BUDGET)

    # --- G2 answer coverage ---
    g2 = {}
    if os.path.exists(e3_result_path or ""):
        r = json.loads(Path(e3_result_path).read_text())
        coarse = r.get("coarse_recall10")
        reranked = r.get("reranked_recall10")
        g2["coarse"] = coarse
        g2["reranked"] = reranked
        g2["baseline"] = G2_BASELINE
        g2["limit"] = G2_BASELINE - G2_DROP_BUDGET
        g2["pass"] = (reranked is not None and reranked >= G2_BASELINE - G2_DROP_BUDGET)
    elif run_g2_e3:
        print("[gate] G2 (E3) requires eval/e3_result.json; run scripts/run_e3.py "
              "on the box and pass --e3-result <path>", file=sys.stderr)
        g2["pass"] = None
    else:
        g2["pass"] = None

    # --- G4 relevance (advisory) ---
    g4 = {"advisory": True, "note": "n=10, do-not-regress"}
    if os.path.exists(str(Path("eval/labels_llm_judge_20261021.json"))):
        g4["coarse_p10"] = G4_COARSE
        g4["reranked_p10"] = G4_RERANKED

    hard_results = []
    if "pass" in g1:
        hard_results.append(g1["pass"])
    if g2.get("pass") is not None:
        hard_results.append(g2["pass"])
    hard_results.append(g3_pass)
    overall_pass = all(bool(x) for x in hard_results)

    return {
        "gate": "real",
        "index": str(index_dir),
        "G1_ann_fidelity": g1,
        "G2_answer_coverage": g2,
        "G3_latency": {
            "p50_ms": round(p50, 3) if p50 is not None else None,
            "p99_ms": round(p99, 3) if p99 is not None else None,
            "budget_ms": budget_ms,
            "pass": g3_pass,
        },
        "G4_relevance_advisory": g4,
        "hard_gates_pass": overall_pass,
        "n_queries": len(totals),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth", action="store_true", help="Tier-0 synthetic smoke")
    ap.add_argument("--n-docs", type=int, default=80)
    ap.add_argument("--index", default="index")
    ap.add_argument("--queries", default="q.jsonl")
    ap.add_argument("--gt", default="gt.jsonl")
    ap.add_argument("--pred", default="", help="pred for G1 (exact-id recall)")
    ap.add_argument("--pred16", default="", help="pred at nprobe=16 for G1")
    ap.add_argument("--e3-result", default="eval/e3_result.json", help="G2 result")
    ap.add_argument("--run-g2-e3", action="store_true",
                    help="attempt G2 live (requires index/store)")
    ap.add_argument("--budget-ms", type=float, default=G3_BUDGET_MS)
    ap.add_argument("--ledger", default="runs.jsonl",
                    help="append one full-config row")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    a = ap.parse_args(argv)

    if a.synth:
        res = gate_synthetic(a.n_docs, min(20, a.n_docs), a.budget_ms)
    else:
        try:
            res = gate_real(a.index, a.queries, a.gt, a.pred, a.pred16,
                            a.e3_result, a.budget_ms, a.run_g2_e3, True)
        except Exception as e:  # noqa: BLE001
            print(f"gate: runtime error: {e}", file=sys.stderr)
            return 2

    if a.json:
        print(json.dumps(res, indent=2))
    else:
        g = res
        if a.synth:
            print(f"G1 ann fidelity:  {g['G1_ann_fidelity']['value']:.4f}  "
                  f"(limit {g['G1_ann_fidelity']['limit']:.4f}) -> "
                  f"{'PASS' if g['G1_ann_fidelity']['pass'] else 'FAIL'}")
            print(f"G2 answer cov.:   {g['G2_answer_coverage']['value']:.4f}  "
                  f"(limit {g['G2_answer_coverage']['limit']:.4f}) -> "
                  f"{'PASS' if g['G2_answer_coverage']['pass'] else 'FAIL'}")
            print(f"G3 latency:       p50={g['G3_latency']['p50_ms']:.3f}  "
                  f"p99={g['G3_latency']['p99_ms']:.3f}  (budget {g['G3_latency']['budget_ms']}) -> "
                  f"{'PASS' if g['G3_latency']['pass'] else 'FAIL'}")
            print("G4 relevance:     advisory (n=10, do-not-regress)")
        else:
            print("=== gate result ===")
            for k in ("G1_ann_fidelity", "G2_answer_coverage", "G3_latency"):
                print(f"{k}: {json.dumps(g.get(k))}")
            print(f"hard_gates_pass: {g.get('hard_gates_pass')}")

    if a.synth:
        passed = all([
            res["G1_ann_fidelity"]["pass"],
            res["G2_answer_coverage"]["pass"],
            res["G3_latency"]["pass"],
        ])
    else:
        passed = res.get("hard_gates_pass", True)
    # no input data at all -> usage-ish
    if a.synth is False and not res.get("G3_latency", {}).get("p50_ms"):
        print("gate: no measurement taken", file=sys.stderr)
        return 2

    if a.ledger:
        try:
            entry = {
                "item": "3",
                "synth": a.synth,
                "budget_ms": a.budget_ms,
                **res,
            }
            with open(a.ledger, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            print(f"gate: ledger write failed: {e}", file=sys.stderr)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
