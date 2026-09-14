# tests/test_e2e_tier0.py (Task 12)
"""Tier-0 e2e latency: synth 500-doc index via existing pieces.

Offline note (same contract as test_orch.py): ann_core is not importable in
this sandbox (no maturin/PyPI), so the ANN backend is an injected numpy
fake with the exact search() shape. The cloud_only real-budget test
(--ann ivf, budget 100ms) runs on the cloud box later.
"""
import subprocess
import sys

import pytest

from exa_home.orch import run_search
from scripts.e2e_latency import build_synth_dag, pct, run_latency

SCRIPTS_E2E = "scripts/e2e_latency.py"


def test_synth_e2e_exits_zero_with_generous_budget():
    r = subprocess.run(
        [sys.executable, SCRIPTS_E2E, "--ann", "fake",
         "--synth-docs", "500", "--synth-queries", "20",
         "--k", "10", "--budget-ms", "5000"],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "waterfall integrity: OK" in r.stdout
    assert "total" in r.stdout and "p50 (ms)" in r.stdout


def test_synth_e2e_waterfall_and_shape_in_process():
    dag = build_synth_dag()
    queries = [{"query": f"topic {i} neural search", "filters": {}}
               for i in range(5)]
    stats = run_latency(dag, queries, k=10)
    assert stats["violations"] == []
    assert len(stats["total"]) == 5
    assert pct(stats["total"], 0.5) >= 0.0
    res = run_search(dag, "topic 1 neural search", {}, top_k=10)
    assert len(res["results"]) <= 10 and res["degraded"] is False


def test_over_budget_exits_one():
    r = subprocess.run(
        [sys.executable, SCRIPTS_E2E, "--ann", "fake",
         "--synth-docs", "500", "--synth-queries", "5",
         "--budget-ms", "0.000001"],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 1, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "OVER BUDGET" in r.stdout


@pytest.mark.cloud_only
def test_real_index_under_100ms_budget():
    """Cloud box: real IvfIndex + Arctic + reranker under the 100ms gate."""
    r = subprocess.run(
        [sys.executable, SCRIPTS_E2E, "--ann", "ivf",
         "--index", "index/", "--queries", "q.jsonl",
         "--k", "10", "--budget-ms", "100",
         "--embedder", "arctic", "--reranker", "real"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
