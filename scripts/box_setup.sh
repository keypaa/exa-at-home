#!/usr/bin/env bash
# box_setup.sh — one-shot wiped-box setup: rust + repo deps + ann_core build.
#
# The box is wiped every session, so cargo is gone EVERY time. Run this
# first in the box terminal (or press the notebook's setup button, which
# performs the same steps from Python):
#
#   bash scripts/box_setup.sh
#
# Idempotent: each step checks before doing anything, so re-runs are cheap.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v cargo >/dev/null 2>&1; then
    echo "rust: no cargo — installing stable toolchain via rustup ..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
else
    echo "rust: $(command -v cargo)"
fi
export PATH="$HOME/.cargo/bin:$PATH"

if [ "$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 14) else 0)')" = "1" ]; then
    export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
    echo "python >= 3.14: PyO3 forward-compat flag set"
fi

echo "pip: installing repo + dev extras ..."
err="$(mktemp)"
if (cd "$REPO" && python3 -m pip install -q -e ".[dev]" 2>"$err"); then
    rm -f "$err"
elif grep -q "externally-managed" "$err" 2>/dev/null; then
    echo "pip: no venv active (PEP 668) — retrying with --break-system-packages"
    (cd "$REPO" && python3 -m pip install -q --break-system-packages -e ".[dev]")
    rm -f "$err"
else
    cat "$err"; rm -f "$err"; exit 1
fi

if python3 -c "import ann_core" 2>/dev/null; then
    echo "ann_core: already importable"
else
    echo "rust: building ann_core (maturin develop, minutes) ..."
    (cd "$REPO/crates/ann-core" && python3 -m maturin develop)
    echo "ann_core: built"
fi

echo "box setup complete."
