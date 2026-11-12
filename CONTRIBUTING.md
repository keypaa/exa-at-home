# Contributing to exa-at-home

Thanks for contributing! exa-at-home is a CPU-first, binary-quantized neural
search engine (Python orchestration + Rust/PyO3 ANN core + FastMCP server)
built to reproduce Exa-style low-latency search at home. Every optimization
lands as a measured before/after experiment.

## Getting started

```bash
git clone https://github.com/keypaa/exa-at-home.git
cd exa-at-home
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1   # Python 3.14 + PyO3 0.22
cd crates/ann-core && maturin develop && cd ../..
python -m pytest tests/ -m "not cloud_only" -q   # 118 passed
cargo test -p ann-core                            # 28 passed
```

Python 3.14+ requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` because the
pinned PyO3 0.22 only advertises support up to Python 3.13.

## How changes flow

1. Fork the repo, create a branch from `main`.
2. Write or update tests for the behavior you change (the fake backends in
   `exa_home/synth.py` must stay in sync with the real contracts — see
   `tests/test_ann.py`).
3. Run the full local verification (below) and make it green.
4. Open a pull request; a maintainer will review.

## Local verification (must be green)

```bash
.venv/bin/ruff check exa_home/ scripts/ tests/ notebooks/
.venv/bin/ruff format --check exa_home/ scripts/ tests/ notebooks/
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo fmt --check -p ann-core
.venv/bin/python -m pytest tests/ -m "not cloud_only" -q
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p ann-core
```

The `cloud_only` tests need a GPU box with torch/sentence-transformers and
Common Crawl access; they run only via the CI `cloud_only` job
(workflow_dispatch / weekly schedule).

## Config discipline

`exa_home/config.py` holds the single measured operating point
(`nprobe 8, top_coarse 50, max_pair_tokens 64, batch_size 32, budget 100ms`).
`exa_home/config_loader.py` (Lane B) is a separate YAML/JSON hardware-profile
system consumed by `scripts/run_pipeline.py`. The MCP server and orchestrator
read `config.OPERATING_POINT` — do not hardcode serving knobs elsewhere.

## Release

See `ROADMAP.md` and `WORKPLAN.md`. Bump `pyproject.toml` version, update
`CHANGELOG.md`, and tag the release. No pushes to `main` without a green CI
run.

## Code of conduct

Please abide by the project [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting.
