<!-- Keep this short: the reviewer needs the WHAT/WHY/VERIFY, not a re-run of the diff. -->
## What and why

<!-- One paragraph. Cite the audit finding ID if this closes one. -->

## What changed (by area)

- [ ] `exa_home/` (library)
- [ ] `scripts/` (pipeline)
- [ ] `crates/ann-core` (Rust)
- [ ] `tests/`
- [ ] docs (README / EXPERIMENTS / RUNBOOK / HANDOFF)
- [ ] CI

## Verification (paste the commands you ran and their results)

```bash
.venv/bin/python -m pytest tests/ -m "not cloud_only" -q   # X passed, Y deselected
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p ann-core
.venv/bin/ruff check exa_home/ scripts/ tests/ notebooks/
.venv/bin/ruff format --check exa_home/ scripts/ tests/ notebooks/
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo fmt --check -p ann-core
```

## Behaviour-change coverage

<!-- Every behaviour change needs a test that FAILS without the change. -->

- [ ] New/updated test proves the fix (name it):

## Audit-finding linkage

- Finding(s): 
- zenno plan task(s): 

## Notes for the reviewer

<!-- What is deliberately NOT in this PR; follow-ups; known limitations. -->
