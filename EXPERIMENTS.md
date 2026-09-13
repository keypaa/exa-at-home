# Experiments (M1–M6 before/after)

Every optimization lands as a measured before/after row (spec §9).

| Milestone | Change | p50 before → after | recall@10 before → after | Notes |
|-----------|--------|--------------------|--------------------------|-------|
| M1 baseline | float brute force, serial | numpy exact 200×256 synth: ~1.42ms/query (wall, single query, laptop CPU) | 1.0 by construction (correctness ref) | `ann_core.AnnIndex` exact float dot; `cargo bench` latency to follow (Task 12). Rust unit + pytest green on networked toolchain; sandbox run limited to numpy-exact reference (see task-4-report). |
| M2 binary quant | sign bit per dim (256f → 32B, 32x), naive per-bit `binary_dot_packed` | Rust latency pending on-box bench (Task 12); numpy exact 500×256 here: ~0.043ms/query for scale context | 0.4460 (numpy reference of `binary_dot_packed` vs `ground_truth.py` exact, 500-doc synth, 100 queries) | Expected dip vs M1 that M5 recovers. Bit-order contract: LSB-first (`packbits bitorder="little"`); bare-packbits default verified WRONG vs `binarize`. Rust `from_binary`/`search_binary` + `cargo test`/pytest need on-box `maturin develop` (sandbox: PyO3 0.22 vs Python 3.14 ABI gate, no egress). |
