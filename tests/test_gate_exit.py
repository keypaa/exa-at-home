# tests/test_gate_exit.py — T7: gate.py exit codes are meaningful.
#
# A gate that exits 0 on failure is worse than no gate. These run the real
# CLI as a subprocess (with the ledger disabled: no synthetic rows are ever
# appended to runs.jsonl) and assert the documented exit codes.
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _gate(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.gate", *args, "--ledger", ""],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )


def test_gate_synth_passes_with_exit_zero():
    r = _gate("--synth", "--n-docs", "25", "--json")
    assert r.returncode == 0, r.stderr
    import json

    payload = json.loads(r.stdout)
    # Every hard gate reports pass=true on the synth smoke.
    for gate in ("G1_ann_fidelity", "G2_answer_coverage", "G3_latency"):
        assert payload[gate]["pass"] is True, payload[gate]


def test_gate_exits_one_when_a_hard_gate_regresses():
    # A zero latency budget makes G3 fail: the gate must exit 1, not 0.
    r = _gate("--synth", "--n-docs", "25", "--budget-ms", "0.0001", "--json")
    assert r.returncode == 1
    assert "G3" in r.stdout or "G3" in r.stderr


def test_gate_exits_two_on_usage_error():
    # Both inputs are nonexistent so the test cannot pass because a real
    # index happened to load (and it fails before loading models).
    r = _gate(
        "--index",
        "does-not-exist/index",
        "--queries",
        "does-not-exist/q.jsonl",
    )
    assert r.returncode == 2, r.stderr
    assert "runtime error" in r.stderr


def test_gate_default_does_not_append_a_synthetic_ledger_row(tmp_path):
    """`--ledger` must be opt-in: CI runs `gate.py --synth` on every push
    and used to append a synth row to the tracked evidence ledger."""
    import os

    r = subprocess.run(
        [sys.executable, "-m", "scripts.gate", "--synth", "--n-docs", "25", "--json"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "runs.jsonl").exists(), "gate appended by default"


def test_g2_without_a_result_file_is_reported_as_skipped():
    """`--run-g2-e3` with no result file must not look like a measured pass."""
    from scripts.gate import _g2_status

    assert _g2_status(True, False) == {"skipped": False, "skip_reason": None}
    got = _g2_status(False, True)
    assert got["skipped"] is True and "no e3 result" in got["skip_reason"]
    got = _g2_status(False, False)
    assert got["skipped"] is True and "not --run-g2-e3" in got["skip_reason"]


def test_gate_treats_a_zero_p50_as_a_real_measurement(monkeypatch, capsys):
    """p50 == 0.0 is a valid measurement; only None means 'no measurement'."""
    from scripts import gate

    def fake_real(*args, **kwargs):
        return {
            "gate": "real",
            "G1_ann_fidelity": {"skipped": True, "pass": None},
            "G2_answer_coverage": {"skipped": True, "pass": None},
            "G3_latency": {"p50_ms": 0.0, "p99_ms": 0.0, "pass": True},
            "hard_gates_pass": True,
            "skipped_gates": ["G1", "G2"],
        }

    monkeypatch.setattr(gate, "gate_real", fake_real)
    assert gate.main(["--json"]) == 0

    def fake_none(*args, **kwargs):
        res = fake_real()
        res["G3_latency"]["p50_ms"] = None
        return res

    monkeypatch.setattr(gate, "gate_real", fake_none)
    assert gate.main(["--json"]) == 2
