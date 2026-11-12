---
name: Bug report
about: Something behaves differently from what the docs claim
title: "[bug] "
labels: bug
---

## What happened

<!-- One paragraph. Include the exact command you ran. -->

## What you expected

<!-- One paragraph, citing the doc/claim that was violated. -->

## Reproduction

```bash
# minimal command / script
```

## Environment

- OS / CPU / RAM / GPU:
- Python version (`python -V`):
- `git rev-parse HEAD`:
- ann_core built? (`python -c "import ann_core"`):
- torch / sentence-transformers installed?:

## Outputs

<details>
<summary>terminal output / traceback</summary>

```
paste here (redact tokens/paths)
```

</details>

## Which stage?

- [ ] ingest / store (`exa_home/`)
- [ ] embed / rerank
- [ ] Rust ANN (`crates/ann-core`)
- [ ] pipeline (`scripts/run_pipeline.py`)
- [ ] gate / measurements
- [ ] MCP server
- [ ] docs
