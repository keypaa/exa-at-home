# Experiments (M1–M6 before/after)

Every optimization lands as a measured before/after row (spec §9).

| Milestone | Change | p50 before → after | recall@10 before → after | Notes |
|-----------|--------|--------------------|--------------------------|-------|
| M1 baseline | float brute force, serial | numpy exact 200×256 synth: ~1.42ms/query (wall, single query, laptop CPU) | 1.0 by construction (correctness ref) | `ann_core.AnnIndex` exact float dot; `cargo bench` latency to follow (Task 12). Rust unit + pytest green on networked toolchain; sandbox run limited to numpy-exact reference (see task-4-report). |
