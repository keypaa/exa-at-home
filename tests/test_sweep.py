# tests/test_sweep.py — Tier-0 for scripts/sweep.py (knob-grid benchmark).
#
# Offline: tiny synth grid (HashEmbedder + NumpyFakeANN + MockReranker),
# asserts the reproducibility contract — timestamped out dir (never
# overwritten), per-combo pred files, file+stdout logging, ledger rows
# carrying full config + recall field. The cloud_only real-backend path
# runs on the box later.
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_synth_grid_writes_all_artifacts(tmp_path):
    from scripts.sweep import run_grid, build_synth_backends
    backends = build_synth_backends(n_docs=50, seed=0)
    queries = [{"query": f"synthetic topic {i} about neural search",
                "filters": {}} for i in range(3)]
    out = tmp_path / "out"
    ledger = tmp_path / "runs.jsonl"
    rows = run_grid(backends, queries, gt_rows=None,
                    nprobes=[1, 2], top_coarses=[10, 20], k=5,
                    pair_tokens_list=[64], batch_sizes=[32],
                    out_dir=str(out), ledger_path=str(ledger),
                    budget_ms=5000.0)
    assert len(rows) == 4  # 2 x 2 grid
    assert (out / "sweep.log").exists()
    assert (out / "grid.json").exists()
    grid = json.loads((out / "grid.json").read_text())
    assert grid["combos"] == [
        {"nprobe": 1, "top_coarse": 10,
         "max_pair_tokens": 64, "batch_size": 32},
        {"nprobe": 1, "top_coarse": 20,
         "max_pair_tokens": 64, "batch_size": 32},
        {"nprobe": 2, "top_coarse": 10,
         "max_pair_tokens": 64, "batch_size": 32},
        {"nprobe": 2, "top_coarse": 20,
         "max_pair_tokens": 64, "batch_size": 32}]
    for r in rows:
        slug = (f"np{r['config']['nprobe']}_tc{r['config']['top_coarse']}"
                f"_pt{r['config']['max_pair_tokens']}_b{r['config']['batch_size']}")
        assert (out / f"pred_{slug}.jsonl").exists()
        assert r["config"]["nprobe"] in (1, 2)
        assert r["n"] == 3 and "p50_ms" in r and "recall_at_10" in r
    ledger_rows = [json.loads(l) for l in open(ledger)]
    assert len(ledger_rows) == 4


def test_second_run_never_overwrites_first(tmp_path, monkeypatch):
    import scripts.sweep as sw
    monkeypatch.setattr(sw, "_timestamp", lambda: "FIXED-TS")
    base = tmp_path / "sweeps"
    first = sw.fresh_out_dir(str(base))
    Path(first).mkdir(parents=True)
    second = sw.fresh_out_dir(str(base))
    assert second != first and second.endswith("-1")


def test_cli_help_lists_grid_flags():
    r = subprocess.run([sys.executable, str(SCRIPTS / "sweep.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for flag in ("--nprobe", "--top-coarse", "--out", "--ledger",
                 "--budget-ms", "--synth-docs", "--index",
                 "--max-pair-tokens", "--batch-size"):
        assert flag in r.stdout, flag


def test_reranker_dims_extend_grid(tmp_path):
    """pair-tokens x batch multiply the grid; reranker rebuilt per combo."""
    from scripts.sweep import run_grid, build_synth_backends
    backends = build_synth_backends(n_docs=50, seed=0)
    queries = [{"query": f"synthetic topic {i} about neural search",
                "filters": {}} for i in range(3)]
    out = tmp_path / "out"
    rows = run_grid(backends, queries, gt_rows=None,
                    nprobes=[1], top_coarses=[10], k=5,
                    pair_tokens_list=[32, 64], batch_sizes=[16, 32],
                    out_dir=str(out),
                    ledger_path=str(tmp_path / "runs.jsonl"),
                    budget_ms=5000.0)
    assert len(rows) == 4  # 1 x 1 x 2 x 2
    pts = {(r["config"]["max_pair_tokens"], r["config"]["batch_size"])
           for r in rows}
    assert pts == {(32, 16), (32, 32), (64, 16), (64, 32)}
    assert (out / "pred_np1_tc10_pt32_b16.jsonl").exists()
