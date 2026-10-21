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
    from scripts.sweep import build_synth_backends, run_grid

    backends = build_synth_backends(n_docs=50, seed=0)
    queries = [
        {"query": f"synthetic topic {i} about neural search", "filters": {}}
        for i in range(3)
    ]
    out = tmp_path / "out"
    ledger = tmp_path / "runs.jsonl"
    rows = run_grid(
        backends,
        queries,
        gt_rows=None,
        nprobes=[1, 2],
        top_coarses=[10, 20],
        k=5,
        pair_tokens_list=[64],
        batch_sizes=[32],
        out_dir=str(out),
        ledger_path=str(ledger),
        budget_ms=5000.0,
    )
    assert len(rows) == 4  # 2 x 2 grid
    assert (out / "sweep.log").exists()
    assert (out / "grid.json").exists()
    grid = json.loads((out / "grid.json").read_text())
    assert grid["combos"] == [
        {"nprobe": 1, "top_coarse": 10, "max_pair_tokens": 64, "batch_size": 32},
        {"nprobe": 1, "top_coarse": 20, "max_pair_tokens": 64, "batch_size": 32},
        {"nprobe": 2, "top_coarse": 10, "max_pair_tokens": 64, "batch_size": 32},
        {"nprobe": 2, "top_coarse": 20, "max_pair_tokens": 64, "batch_size": 32},
    ]
    for r in rows:
        slug = (
            f"np{r['config']['nprobe']}_tc{r['config']['top_coarse']}"
            f"_pt{r['config']['max_pair_tokens']}_b{r['config']['batch_size']}"
        )
        assert (out / f"pred_{slug}.jsonl").exists()
        assert r["config"]["nprobe"] in (1, 2)
        assert r["n"] == 3 and "p50_ms" in r and "recall_at_10" in r
    with open(ledger) as f:
        ledger_rows = [json.loads(line) for line in f]
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
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "sweep.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    for flag in (
        "--nprobe",
        "--top-coarse",
        "--out",
        "--ledger",
        "--budget-ms",
        "--synth-docs",
        "--index",
        "--max-pair-tokens",
        "--batch-size",
    ):
        assert flag in r.stdout, flag


def test_recall_skips_gt_rows_without_predictions():
    """A gt row with no prediction is SKIPPED, not a KeyError.

    The old `_recall_at_10` indexed `pr[g["query_id"]]` blind: a sweep with
    fewer predictions than gt rows raised a bare KeyError and lost every
    remaining grid point. scripts/measure_recall.py already skips + reports.
    """
    from scripts.sweep import _recall_at_10

    gt_rows = [
        {"query_id": 0, "top10": [f"d{i}" for i in range(10)]},
        {"query_id": 1, "top10": [f"e{i}" for i in range(10)]},
    ]
    pred_rows = [{"query_id": 0, "top10": [f"d{i}" for i in range(10)]}]
    recall, skipped = _recall_at_10(gt_rows, pred_rows)
    assert skipped == 1
    assert recall == 1.0  # the one comparable query is a perfect hit
    # No comparable query at all -> no recall, but no crash either.
    recall, skipped = _recall_at_10(gt_rows, [{"query_id": 9, "top10": []}])
    assert (recall, skipped) == (None, 2)


def test_grid_survives_gt_with_more_queries_than_predictions(tmp_path):
    """Fewer queries than gt rows: the whole grid still runs."""
    from scripts.sweep import build_synth_backends, run_grid

    backends = build_synth_backends(n_docs=50, seed=0)
    queries = [
        {"query": f"synthetic topic {i} about neural search", "filters": {}}
        for i in range(3)
    ]
    gt_rows = [
        {"query_id": i, "top10": [f"doc-{j}" for j in range(10)]} for i in range(6)
    ]
    out = tmp_path / "out"
    ledger = tmp_path / "runs.jsonl"
    rows = run_grid(
        backends,
        queries,
        gt_rows,
        nprobes=[1, 2],
        top_coarses=[10],
        k=5,
        pair_tokens_list=[64],
        batch_sizes=[32],
        out_dir=str(out),
        ledger_path=str(ledger),
        budget_ms=5000.0,
    )
    assert len(rows) == 2  # every grid point survived
    for r in rows:
        assert r["recall_at_10"] is not None
        assert r["config"]["gt_skipped"] == 3  # gt query_ids 3,4,5 have no pred
    log = (out / "sweep.log").read_text()
    assert "skipped" in log


def test_reranker_dims_extend_grid(tmp_path):
    """pair-tokens x batch multiply the grid; reranker rebuilt per combo."""
    from scripts.sweep import build_synth_backends, run_grid

    backends = build_synth_backends(n_docs=50, seed=0)
    queries = [
        {"query": f"synthetic topic {i} about neural search", "filters": {}}
        for i in range(3)
    ]
    out = tmp_path / "out"
    rows = run_grid(
        backends,
        queries,
        gt_rows=None,
        nprobes=[1],
        top_coarses=[10],
        k=5,
        pair_tokens_list=[32, 64],
        batch_sizes=[16, 32],
        out_dir=str(out),
        ledger_path=str(tmp_path / "runs.jsonl"),
        budget_ms=5000.0,
    )
    assert len(rows) == 4  # 1 x 1 x 2 x 2
    pts = {(r["config"]["max_pair_tokens"], r["config"]["batch_size"]) for r in rows}
    assert pts == {(32, 16), (32, 32), (64, 16), (64, 32)}
    assert (out / "pred_np1_tc10_pt32_b16.jsonl").exists()
