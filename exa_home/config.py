# exa_home/config.py — the single source of truth for knobs and names.
#
# Twenty hardcoded values used to live in eight files (top_coarse, nprobe,
# max_pair_tokens, batch_size, budget, K, crawl, model names...). That made
# the shipped MCP path drift from the measured path — the server ran
# top_coarse=200 while every measurement used 50, i.e. 4x the work of the
# configuration we had actually validated.
#
# OPERATING_POINT is the configuration the EXPERIMENTS.md rows were measured
# at. Anything that serves or measures must read it from here, so "the
# shipped tool" and "the measured tool" cannot diverge by accident.
from __future__ import annotations

# --- the validated operating point (toy corpus, 2026-10-21) --------------
# nprobe=8 is the routing knee (M5c); top_coarse=50 beat 100 and 200 on
# both axes (M5d); pair-tokens=64 is the measured point (M5e); batch 32 is
# the batch knee (M5g: b16 57ms / b32 42ms / b64 43ms).
OPERATING_POINT: dict = {
    "nprobe": 8,
    "top_coarse": 50,
    "max_pair_tokens": 64,
    "batch_size": 32,
    "max_query_tokens": 32,
    "budget_ms": 100.0,
    "k": 10,  # final results returned (NOT the coarse candidate count)
}

# --- models --------------------------------------------------------------
EMBEDDER_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# --- corpus / index ------------------------------------------------------
CRAWL = "CC-MAIN-2026-34"
DIM = 256
K_TOY = 1024
K_1M = 10_000
K_5M = 30_000
CENTROID_SAMPLE = 500_000
EMBED_BATCH = 256
MAX_TERMS = 50_000
DOWNLOAD_WORKERS = 16
SHARD_COUNT = 64
SNIPPET_CHARS = 300


def operating_point(**overrides) -> dict:
    """The operating point with optional per-call overrides."""
    return {**OPERATING_POINT, **overrides}
