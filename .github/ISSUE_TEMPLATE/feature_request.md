---
name: Feature request
about: Propose a change to the operating point, the pipeline, or the MCP surface
title: "[feature] "
labels: enhancement
---

## Problem

<!-- What cannot be measured / reproduced today? Link the doc that says it can. -->

## Proposed change

<!-- Which knob, stage, or contract changes? Reference exa_home/config.py by name. -->

## How it will be measured

<!-- The before/after experiment: metric, corpus, command, expected direction. -->

## Blast radius

- [ ] serving path (MCP / `OPERATING_POINT`)
- [ ] index format (`exa_home/index_format.py` + `crates/ann-core/src/index.rs`)
- [ ] pipeline stages
- [ ] docs (README / EXPERIMENTS / RUNBOOK)
- [ ] tests (Tier-0 / cloud_only)

## Audit-finding linkage

<!-- If this closes an audit finding, cite its ID (e.g. C9, P3, S1). -->
