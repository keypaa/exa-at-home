# Model & data license obligations

exa-at-home bundles configuration and code that references third-party models
and data. This document lists the known obligations so downstream consumers can
remain compliant. It is **not legal advice**.

## Models

| Model | License (per model card) | Obligation |
|-------|--------------------------|------------|
| `mixedbread-ai/mxbai-embed-large-v1` | Apache-2.0 | Include license + attribution; state that truncate_dim=256 is applied |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Apache-2.0 (per model card) | Include license + attribution |

Verify the current license on each model's Hugging Face card before shipping;
model cards can change.

## Embedding / reranking libraries

| Library | License | Notes |
|---------|---------|-------|
| sentence-transformers | Apache-2.0 | `pip` dependency, unpinned (`>=2.2`) |
| torch | BSD-3-Clause | `pip` dependency, unpinned |
| fastmcp | Apache-2.0 | MCP server framework |
| duckdb | MIT | |
| warcio | Apache-2.0 | WET/WARC reading |
| pyroaring | Apache-2.0 / MIT | filter bitmaps |
| scikit-learn | BSD-3-Clause | MiniBatchKMeans for centroids |
| numpy | BSD-3-Clause | |

## Rust crates

| Crate | License |
|-------|---------|
| pyo3 | Apache-2.0 / MIT |
| roaring | Apache-2.0 / MIT |
| sha2 | MIT / Apache-2.0 |
| serde / serde_json | MIT / Apache-2.0 |
| ndarray | MIT |
| criterion | Apache-2.0 / MIT (dev-only) |

## Data (corpus)

The index is built from Common Crawl WET files (crawl `CC-MAIN-2026-34`).
Common Crawl's terms govern use of the crawled content; derived artifacts
(indexes, vectors, snippets returned by the MCP server) are produced from
that content. Do not redistribute Common Crawl-derived content without
reviewing Common Crawl's terms of use. The committed `q.jsonl`/`gt.jsonl`/
`pred.jsonl` are small eval artifacts, not a redistribution of the crawl.

## Attribution notice (suggested)

This product includes software developed by the projects listed above and
embedding/reranking models released under Apache-2.0. See each model's Hugging
Face model card for the exact license text and attribution requirements.
