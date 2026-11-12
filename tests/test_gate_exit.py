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
    r = _gate("--queries", "does-not-exist/q.jsonl")
    assert r.returncode == 2, r.stderr
