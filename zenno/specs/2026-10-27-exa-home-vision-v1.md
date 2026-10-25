# Exa MCP at Home — V1 vision & release spec

## Vision
Build and ship an **open-source, single-box, text-only, measurable Exa-like
neural search engine** that any LLM agent can use through MCP, with p50 under
100 ms at 1 M documents, up to a 5 M stretch. The point is *credible
engineering on one GPU*, not replica-scale.

## Acceptance (binary end state — all true)
1. **Works** — `python scripts/run_pipeline.py all --config config.yaml` builds
   index/ and launches `home_search` over a healthy corpus.
2. **Measured** — every knob lives in `config.yaml` and the operating point in
   `exa_home/config.py`; each run writes one provenance row to `runs.jsonl`.
3. **Professional** — tier-0 (112/112) green; a Tier-2 GPU smoke on the real
   embedder+reranker path exists; ruff/cargo fmt clean; no machine paths.
4. **Documented** — `EXPERIMENTS.md`, `LEARNINGS.md`, `ROADMAP.md`,
   VISION.md all agree on the same numbers.
5. **Open** — Apache-2.0, model-cache setup documented, no auth secret in git.

## Product scope (decisions — user locked 2026-10-24)
- **Text only.** No multimodal ingests; a boolean config gate keeps the door shut.
- **Hardware-adaptive.** `usable_vram_gb`, `usable_ram_gb` (or presets: `tiny`,
  `rtx-3090-classic`, `rtx-pro-6000-classic`) set chunk size, batch size, and
  the on-disk ingest width. The test/CI profile selects `tiny`.
- **Two serving modes**: `lite` (coarse-only) and `auto` (rerank on top-coarse).
  `deep` is a stretch — work on autopilot first.
- **Production as a MCP tool**: localhost-first, optional auth token, rate
  limiting, health check, structured logs, input size caps.
- **Two scale contracts**: 1 M core, 5 M stretch (10 M only after the 5 M gate
  passes).

## Release and licensing
- Apache-2.0
- All config keys documented; a single `.yaml` + a small `--help` table.
- No model weights committed; HF cache and the pin in `config.yaml`.
- No personal paths; any sync artifact happens explicitly by PR.

## Definition of "done" — per hard exit criteria
- G1 exact-id recall@10: no more than −0.10 from the *same corpus's own*
  measured baseline.
- G2 target-doc recall@10 on the frozen 50 questions ≥ 0.64.
- G3 total p50 ≤ 100 ms; see the fallback: on > 90 ms, ship lite-mode first.
- G4 advisory-only (precision@10 on judged set does not block a merge).
- On a `SIGKILL`: embed must resume within one committed chunk (item 2) or the
  index must re-provenance check.

## Guardrails (scope creep defense)
- Do **not** add multimodal inputs or another embedder without a measured A–B.
- Do **not** change history (`runs.jsonl` is evidence — appended, never rewritten).
- New models or libraries come in only on measurement; new libraries on the
  hot path are bounded to one PR with proven bench numbers.

## Next PR boundary (after the 1M gate is green)
1. Unify config via `exa_home/config_loader.py` + a shipped `config.yaml`.
2. `Makefile` / `scripts/make.py` one-shot with runtime detection.
3. Logging/health/rate-limit in the MCP server.
4. Tier-2 real-model smoke test.
5. Ingest URL dedup as a disk pass (P0 ingest throughput).
6. All P0 thread/process audit findings.
