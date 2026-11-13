# tests/test_e2e_tier0.py (Task 12)
"""Tier-0 e2e latency: synth 500-doc index via existing pieces.

Note: the Tier-0 ANN tests need `ann_core` built (`maturin develop`);
the fake backend here mirrors its search() shape so the harness can run
without the Rust extension. The cloud_only real-budget test (--ann ivf,
budget 100ms) runs on the cloud box.
"""

import subprocess
import sys

import pytest

from exa_home.orch import run_search
from scripts.e2e_latency import build_synth_dag, pct, run_latency

SCRIPTS_E2E = "scripts/e2e_latency.py"


def test_synth_e2e_exits_zero_with_generous_budget():
    r = subprocess.run(
        [
            sys.executable,
            SCRIPTS_E2E,
            "--ann",
            "fake",
            "--synth-docs",
            "500",
            "--synth-queries",
            "20",
            "--k",
            "10",
            "--budget-ms",
            "5000",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "waterfall integrity: OK" in r.stdout
    assert "total" in r.stdout and "p50 (ms)" in r.stdout


def test_synth_e2e_waterfall_and_shape_in_process():
    dag = build_synth_dag()
    queries = [{"query": f"topic {i} neural search", "filters": {}} for i in range(5)]
    stats = run_latency(dag, queries, k=10)
    assert stats["violations"] == []
    assert len(stats["total"]) == 5
    assert pct(stats["total"], 0.5) >= 0.0
    res = run_search(dag, "topic 1 neural search", {}, top_k=10)
    assert len(res["results"]) <= 10 and res["degraded"] is False


def test_ledger_appends_one_json_row_per_run(tmp_path):
    from scripts.e2e_latency import append_ledger, build_synth_dag, run_latency

    dag = build_synth_dag(n_docs=100, seed=1)
    queries = [{"query": f"topic {i} neural search", "filters": {}} for i in range(5)]
    stats = run_latency(dag, queries, k=5)
    ledger = tmp_path / "runs.jsonl"
    append_ledger(str(ledger), {"ann": "fake", "n_docs": 100}, stats)
    import json

    with open(ledger, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 1
    row = rows[0]
    assert row["config"]["ann"] == "fake" and row["n"] == 5
    assert "p50_ms" in row and "p99_ms" in row and "git" in row


def test_pct_p50_is_the_50th_smallest_not_the_51st():
    """p50 of 1..100 is 50 (the 50th smallest), not 51.

    The old helper indexed `int(p * len(xs))` = 50 (0-based) = the 51st
    smallest value: one rank high, and the same helper decides the budget
    gate. Nearest-rank definition: 1-based rank ceil(p*n) -> 0-based
    index ceil(p*n)-1.
    """
    import math

    xs = list(range(1, 101))
    want_rank = math.ceil(0.5 * len(xs))  # 50 -> the 50th smallest
    assert want_rank == 50
    assert pct(xs, 0.5) == 50
    assert pct(xs, 0.5) == sorted(xs)[want_rank - 1]
    # The old (buggy) answer was one rank high.
    assert sorted(xs)[int(0.5 * len(xs))] == 51


def test_pct_p99_is_nearest_rank():
    """p99 of 1..100 is the 99th smallest (99), not the max (100)."""
    import math

    xs = list(range(1, 101))
    assert pct(xs, 0.99) == sorted(xs)[math.ceil(0.99 * len(xs)) - 1] == 99


def test_pct_edge_cases():
    assert pct([5.0], 0.5) == 5.0
    assert pct([5.0], 0.99) == 5.0
    assert pct([1.0, 2.0, 3.0], 0.0) == 1.0
    assert pct([1.0, 2.0, 3.0], 1.0) == 3.0
    with pytest.raises(ValueError):
        pct([], 0.5)


def test_budget_gate_uses_the_corrected_p50():
    """50 fast + 50 slow queries: the p50 is the fast half (10ms).

    With the old off-by-one the gate saw 20ms and failed a 15ms budget.
    """
    from scripts.e2e_latency import STAGE_ORDER, report

    totals = [10.0] * 50 + [20.0] * 50
    stats = {
        "n": len(totals),
        "total": totals,
        "stages": {s: list(totals) for s in STAGE_ORDER},
        "breakdown": {},
        "violations": [],
    }
    out = report(stats, 15.0)
    assert "total p50 10.000ms vs 15.0ms -> PASS" in out
    assert "OVER BUDGET" not in out


def test_over_budget_exits_one():
    r = subprocess.run(
        [
            sys.executable,
            SCRIPTS_E2E,
            "--ann",
            "fake",
            "--synth-docs",
            "500",
            "--synth-queries",
            "5",
            "--budget-ms",
            "0.000001",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert r.returncode == 1, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "OVER BUDGET" in r.stdout


@pytest.mark.cloud_only
def test_real_index_under_100ms_budget():
    """Cloud box: real IvfIndex + Arctic + reranker under the 100ms gate."""
    r = subprocess.run(
        [
            sys.executable,
            SCRIPTS_E2E,
            "--ann",
            "ivf",
            "--index",
            "index/",
            "--queries",
            "q.jsonl",
            "--k",
            "10",
            "--budget-ms",
            "100",
            "--embedder",
            "arctic",
            "--reranker",
            "real",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"


def test_latency_defaults_come_from_operating_point(monkeypatch):
    """C4: the CLI must measure the shipped operating point by default."""
    import scripts.e2e_latency as e2e
    from exa_home.config import OPERATING_POINT

    seen = {}

    def fake_build(*args, **kwargs):
        seen.update(kwargs)
        return object()

    def fake_run(dag, queries, k, verbose=False):
        return {
            "n": len(queries),
            "stages": {s: [1.0] for s in e2e.STAGE_ORDER},
            "breakdown": {},
            "total": [1.0],
            "violations": [],
            "top_ids": [],
        }

    monkeypatch.setattr(e2e, "build_real_dag", fake_build)
    monkeypatch.setattr(e2e, "run_latency", fake_run)
    rc = e2e.main(["--ann", "ivf", "--index", "unused", "--synth-queries", "1"])
    assert rc == 0
    assert seen["max_pair_tokens"] == OPERATING_POINT["max_pair_tokens"]
    assert seen["batch_size"] == OPERATING_POINT["batch_size"]


def test_run_latency_excludes_two_warmup_queries(monkeypatch):
    """C13: cold-start queries must not enter the percentiles."""
    import scripts.e2e_latency as e2e

    calls = []

    def fake_run_search(dag, query, filters, top_k, profile=False):
        calls.append(query)
        prof = {s: 1.0 for s in e2e.STAGE_ORDER}
        prof["total_ms"] = float(len(e2e.STAGE_ORDER))
        return {"profile": prof, "results": [], "degraded": False}

    monkeypatch.setattr("exa_home.orch.run_search", fake_run_search)
    queries = [{"query": f"q{i}", "filters": {}} for i in range(5)]
    stats = e2e.run_latency(object(), queries, k=10)
    assert len(calls) == 7, f"expected 2 warmup + 5 measured calls, got {len(calls)}"
    assert calls[:2] == ["q0", "q1"]
    assert stats["n"] == 5 and len(stats["total"]) == 5
