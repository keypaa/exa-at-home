# Exa-at-Home Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-first, Exa-faithful neural search + MCP system (`home_search`, `home_contents`) that serves 1–5M Common Crawl pages at p50 < 100ms on a single cloud box, with every optimization landed as a measured before/after experiment.

**Architecture:** Python does orchestration, ingest, embedding, rerank, and MCP (FastMCP); Rust (PyO3) owns the hot ANN path (binary IVF + LUT scoring + filter bitsets). Offline ingest produces a versioned, checksummed `index/` directory; online serving funnels 5M docs → 200 coarse candidates → top-10 reranked.

**Tech Stack:** Python ≥3.11, `sentence-transformers` + `torch`, `fastmcp`, `numpy`, `duckdb` (slice selection), `warcio` (WET parsing), `pyroaring` (filter build); Rust edition 2021, `pyo3 0.22` + `numpy 0.22` (via `maturin`), `ndarray 0.16`, `roaring`, `sha2`, `serde/serde_json`, `criterion 0.5`; pytest with `cloud_only` marker.

**Spec:** `zenno/specs/2026-09-13-exa-at-home-design.md`

## Global Constraints

- MCP tools are namespaced `home_search` / `home_contents` — never `exa_*`.
- `home_search` p50 < 100ms end to end on the cloud box; per-stage budgets: embed 5–10ms, ANN 10–20ms, filter/fuse <5ms, rerank 20–40ms, snippet/format <10ms, orchestrator <5ms.
- Recall gate: every index change reports `recall@10` vs brute-force ground truth; below-threshold recall fails CI no matter how fast.
- Tier rule: anything needing the embedding model or >1k docs is `@pytest.mark.cloud_only` and never runs in local CI. Local CI = Tier 0 + Rust unit tests, seconds on any CPU.
- Rerank timeout (>100ms) falls back to coarse ANN order with `degraded: true`, never a failed query.
- Index shards are checksummed; corrupt index fails startup fast, never serves silent partial recall.
- Each optimization milestone (M1–M6) records a before/after row in `EXPERIMENTS.md`.

## Decisions locking spec §11 open questions

1. **Embedding model:** `Snowflake/snowflake-arctic-embed-m-v2.0`, native MRL-256 (`truncate_dim=256`), query prompt via `prompt_name="query"` (document-side prefix TBD — verify on-box; likely unprefixed). Apache-2.0, fully open. ([model page](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0)) Chosen over Nomic v2-moe / mxbai-large-v1 / Qwen3-0.6B because it is the only candidate with published quality at exactly 256 dims (~99% MTEB-R retention, beating truncated OpenAI-3-Large); jina-v3 has good 256 numbers but is CC-BY-NC (disqualified); BGE-M3 has no native MRL (disqualified). No `trust_remote_code` unless on-box load requires it (Arctic is native sentence-transformers).
2. **Rust↔Python bridge:** PyO3 via `maturin` (in-process FFI, zero per-query IPC overhead). Unix-socket sidecar rejected: adds per-query serialization latency we would then have to optimize away.
3. **Reranker:** `cross-encoder/ms-marco-MiniLM-L6-v2` — same NDCG@10 (74.30) and MRR (39.01) as L12 at ~2x throughput, so L12 buys nothing. Budget math: ~20–35K pairs/s on A100-class GPU at batch 128 fp16 short pairs → 200 pairs in ~6–10ms, inside the 40ms rerank slice (Blackwell box should match or beat A100; validate on-box). Batch guidance: `batch_size=128`, truncate pairs to 128–192 tokens, one warmup call, fp16. Apache-2.0. ([benchmark table](https://github.com/huggingface/sentence-transformers/blob/main/docs/cross_encoder/pretrained_models.md))
4. **Corpus source: WET files, not WARC+Trafilatura.** The spec says WARC slice → Trafilatura extraction, but Common Crawl ships pre-extracted plaintext as WET archives alongside every crawl, which is exactly the input our pipeline needs. V1 reads WET via `warcio` (no HTML parsing, no Trafilatura dependency); WARC+Trafilatura stays a documented upgrade. Crawl locked: `CC-MAIN-2026-34` (Aug 2026, 2.14B pages, 40.45% English, 100k WET files / 5.84 TiB, ~8.6k English pages per WET). Fetch over free HTTPS `https://data.commoncrawl.org/` (unsigned S3 disabled since 2022 — do NOT use `s3://` outside AWS). Slice sizes: 1M usable English ≈ top ~130 WET files (~8GB); 5M ≈ top ~620 files (~38GB). ([latest crawl](https://commoncrawl.org/latest-crawl), [columnar index](https://commoncrawl.org/url-index))
5. **Slice selection:** DuckDB over the columnar Parquet index, filter `crawl='CC-MAIN-2026-34' AND subset='warc' AND fetch_status=200 AND content_mime_detected='text/html' AND content_languages LIKE '%eng%'`; map chosen WARC paths to WET by replacing `/warc/` with `/wet/` (file list also at `crawl-data/CC-MAIN-2026-34/wet.paths.gz`). Dedup by `content_digest` (SHA-1 `WARC-Payload-Digest`) first, URL second; exclude `robotstxt`/`crawldiagnostics` subsets and truncated records.
6. **Centroid training:** Python `sklearn MiniBatchKMeans` on a ≤500k-doc sample (offline, cloud), export `centroids.npy`; Rust only serves. Keeps k-means deps out of the hot path.
7. **One embedding per page:** embed `text[:12000]` (~3k tokens, inside the card's 8192-token RoPE context with margin — verify effective limit on-box); full text lives in the content store. No chunking in V1.

---

## File structure

```
exa_at_home_exp/
  pyproject.toml                 # python deps, pytest markers (cloud_only), maturin build
  Cargo.toml                     # workspace root
  crates/
    ann-core/
      Cargo.toml                 # pyo3 0.22 + numpy 0.22, ndarray 0.16, roaring, sha2, serde, criterion
      src/
        lib.rs                   # PyO3 module root: AnnIndex (M1) + IvfIndex (M3+)
        quant.rs                 # dot(), binarize(), binary_dot_packed()
        lut.rs                   # M4: per-query 64×16 LUT build + nibble-gather scoring
        ivf.rs                   # M3: centroid routing, inverted lists, nprobe top-k
        filter.rs                # Task 8: roaring bitmap term/domain/date index, allow-list intersect
        index.rs                 # Task 6: index/ dir loader, manifest + sha256 verify, memmap
      benches/search_bench.rs    # criterion: binary_dot vs LUT (M4), single-query + concurrency 1/8/64 (Task 12)
  exa_home/
    __init__.py
    ingest.py                    # read_wet() -> RawDoc; write_docs_jsonl(); shard checkpoints
    store.py                     # ContentStore.put/get with RAM LRU (+ counters)
    embed.py                     # Embedder (Arctic-m, native MRL-256, ST query prompt) + HashEmbedder stub
    rerank.py                    # Reranker (MiniLM-L6-v2, batch, timeout fallback)
    orch.py                      # Canon-lite DAG: nodes, memo, cancellation, spans
    mcp_server.py                # FastMCP app: home_search, home_contents
    index_format.py              # manifest read/write/verify (py side, mirrors index.rs)
  scripts/
    select_slice.py              # DuckDB over columnar index -> wet_urls.txt (cloud-only)
    download_wet.py              # fetch N WET files with resume (cloud-only)
    train_centroids.py           # MiniBatchKMeans on sample -> centroids.npy (cloud-only)
    ground_truth.py              # brute-force exact top-10 for sample queries -> gt.jsonl
    e2e_latency.py               # end-to-end p50/p99 + profile waterfall check
  tests/
    fixtures/synth.py            # seeded RNG: random float32 vecs, fake docs, tiny synthetic WET.gz
    test_ingest.py               # Tier 0
    test_store.py                # Tier 0
    test_embed.py                # Tier 0 (hash stub) + cloud_only (real model shape test)
    test_ann.py                  # Tier 0 via PyO3 (needs `maturin develop` first)
    test_rerank.py               # Tier 0 (mock scores) + cloud_only (real model)
    test_orch.py                 # Tier 0
    test_mcp.py                  # Tier 0, mocked backends
  EXPERIMENTS.md                 # M1–M6 before/after table (the learning artifact)
  docs/RUNBOOK.md                # cloud-only: slice → toy → full build → validation
```

---

### Task 1: Scaffold + Tier-0 fixtures + tier markers

**Files:**
- Create: `pyproject.toml`, `Cargo.toml`, `crates/ann-core/Cargo.toml`, `crates/ann-core/src/lib.rs`, `exa_home/__init__.py`, `tests/fixtures/synth.py`, `tests/test_scaffold.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `tests.fixtures.synth.make_vectors(n, dim, seed) -> np.ndarray[float32]`, `make_docs(n, seed) -> list[dict]`, `make_wet_gz(path, docs)`; pytest marker `cloud_only` registered; `cargo test` + `pytest -m "not cloud_only"` both green.

- [ ] **Step 1: Write pyproject with markers and deps**

```toml
[project]
name = "exa-home"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "warcio>=1.7",
  "sentence-transformers>=2.2",
  "torch",
  "fastmcp",
  "duckdb",
  "scikit-learn",
  "pyroaring>=0.4",
  "pytest",
]

[project.optional-dependencies]
dev = ["maturin>=1.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Note: pip does NOT build the Rust module — `ann_core` is built with
`maturin develop` in Task 4 (`pip install -e .[dev]` for maturin).

[tool.pytest.ini_options]
markers = ["cloud_only: needs GPU/model or >1k docs; never runs in local CI"]
```

- [ ] **Step 2: Write workspace + crate Cargo.toml and a stub lib.rs**

```toml
# Cargo.toml (root)
[workspace]
members = ["crates/ann-core"]
resolver = "2"
```

```toml
# crates/ann-core/Cargo.toml
[package]
name = "ann-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "ann_core"
crate-type = ["cdylib", "rlib"]  # rlib so criterion benches can link internals

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"
ndarray = "0.16"
roaring = "0.10"
sha2 = "0.10"          # manifest verify (Task 6)
serde = { version = "1", features = ["derive"] }  # index JSON (Tasks 6, 8)
serde_json = "1"

[dev-dependencies]
criterion = "0.5"
```

```rust
// crates/ann-core/src/lib.rs (Task 1 stub; replaced by the M1 module in Task 4)
use pyo3::{prelude::*, Bound, types::PyModule};

#[pymodule]
fn ann_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
```

- [ ] **Step 3: Write Tier-0 synth fixtures**

```python
# tests/fixtures/synth.py
import gzip, io
import numpy as np

def make_vectors(n: int, dim: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v

def make_docs(n: int, seed: int = 0) -> list[dict]:
    return [
        {"id": f"doc-{i}", "url": f"https://example.com/{i}",
         "title": f"Title {i}", "text": f"This is synthetic document {i} about topic {i % 7}.",
         "date": "2026-01-01"}
        for i in range(n)
    ]

def make_wet_gz(path: str, docs: list[dict]) -> None:
    """Minimal real WET-format gzip: warcio can parse it back."""
    from warcio.warcwriter import WARCWriter
    with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb") as gz:
        writer = WARCWriter(gz, gzip=False)
        for d in docs:
            payload = d["text"].encode("utf-8")
            rec = writer.create_warc_record(
                d["url"], "conversion",
                payload=io.BytesIO(payload),
                warc_content_type="text/plain")
            writer.write_record(rec)
```

- [ ] **Step 4: Write a scaffold test and run everything**

```python
# tests/test_scaffold.py
import numpy as np
from fixtures.synth import make_vectors, make_docs, make_wet_gz

def test_synth_vectors_are_normalized():
    v = make_vectors(10, 256, seed=1)
    assert v.shape == (10, 256)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)

def test_synth_wet_roundtrip(tmp_path):
    from warcio.archiveiterator import ArchiveIterator
    p = str(tmp_path / "t.wet.gz")
    make_wet_gz(p, make_docs(3))
    recs = [r for r in ArchiveIterator(open(p, "rb")) if r.rec_type == "conversion"]
    assert len(recs) == 3
```

Run: `pytest tests/test_scaffold.py -v` (expect PASS), `cargo test -p ann-core` (expect PASS, 0 tests).
Also verify gate command works: `pytest -m "not cloud_only" -q`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml Cargo.toml crates exa_home tests
git commit -m "chore: scaffold workspace, Tier-0 fixtures, cloud_only marker"
```

---

### Task 2: Ingester (WET reader) + content store

**Files:**
- Create: `exa_home/ingest.py`, `exa_home/store.py`
- Test: `tests/test_ingest.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `tests.fixtures.synth.make_wet_gz` (Task 1).
- Produces: `ingest.RawDoc(id, url, title, text, date, digest)`, `ingest.read_wet(path) -> Iterator[RawDoc]`, `ingest.write_docs_jsonl(docs, path) -> IngestStats(kept, skipped, reasons)`, `store.ContentStore(root).put(doc) -> None`, `store.ContentStore.get(id) -> StoredDoc | None`, `store.ContentStore.stats -> {hits, misses}`.

- [ ] **Step 1: Write failing tests for WET ingest**

```python
# tests/test_ingest.py
from fixtures.synth import make_docs, make_wet_gz
from exa_home.ingest import read_wet, write_docs_jsonl

def test_read_wet_yields_raw_docs(tmp_path):
    p = str(tmp_path / "t.wet.gz")
    make_wet_gz(p, make_docs(5))
    docs = list(read_wet(p))
    assert len(docs) == 5
    assert docs[0].url == "https://example.com/0"
    assert "synthetic document 0" in docs[0].text

def test_ingest_skips_empty_and_counts(tmp_path):
    p = str(tmp_path / "t.wet.gz")
    docs = make_docs(3) + [{"id": "x", "url": "https://e.co/x", "title": "t", "text": "  ", "date": ""}]
    make_wet_gz(p, docs)
    out = str(tmp_path / "docs.jsonl")
    stats = write_docs_jsonl(read_wet(p), out)
    assert (stats.kept, stats.skipped) == (3, 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'exa_home.ingest'` (or ImportError on names).

- [ ] **Step 3: Implement ingester**

```python
# exa_home/ingest.py
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from typing import Iterator
from warcio.archiveiterator import ArchiveIterator

@dataclass
class RawDoc:
    id: str          # content-digest based stable id
    url: str
    title: str       # "" for WET (title comes from WARC upgrade path later)
    text: str
    date: str        # "" for WET v1
    digest: str

@dataclass
class IngestStats:
    kept: int = 0
    skipped: int = 0
    reasons: dict | None = None

def read_wet(path: str) -> Iterator[RawDoc]:
    with open(path, "rb") as f:
        for rec in ArchiveIterator(f):
            if rec.rec_type != "conversion":
                continue
            url = rec.rec_headers.get_header("WARC-Target-URI")
            payload = rec.content_stream().read()
            try:
                text = payload.decode("utf-8", errors="strict").strip()
            except UnicodeDecodeError:
                continue  # counted as skipped downstream only if wired; keep reader pure
            if not text:
                continue
            digest = hashlib.sha1(payload).hexdigest()
            yield RawDoc(id=f"sha1:{digest}", url=url, title="", text=text, date="", digest=digest)

def write_docs_jsonl(docs: Iterator[RawDoc], out_path: str) -> IngestStats:
    from collections import Counter
    stats, reasons, seen = IngestStats(), Counter(), set()
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            if d.digest in seen:
                stats.skipped += 1; reasons["duplicate"] += 1; continue
            seen.add(d.digest)
            if len(d.text) < 50:
                stats.skipped += 1; reasons["too_short"] += 1; continue
            f.write(json.dumps({"id": d.id, "url": d.url, "title": d.title,
                                "text": d.text, "date": d.date}) + "\n")
            stats.kept += 1
    stats.reasons = dict(reasons)
    return stats
```

Note: `read_wet` already drops undecodable/empty payloads; `write_docs_jsonl` enforces length + exact-digest dedup.

- [ ] **Step 4: Implement ContentStore + its tests**

```python
# tests/test_store.py
from exa_home.store import ContentStore

def test_put_get_roundtrip(tmp_path):
    s = ContentStore(str(tmp_path))
    s.put({"id": "a", "url": "https://e.co", "title": "T", "text": "hello world"})
    got = s.get("a")
    assert got["text"] == "hello world"
    assert s.get("missing") is None
    assert s.stats["hits"] == 1 and s.stats["misses"] == 1
```

```python
# exa_home/store.py
from __future__ import annotations
import json, os
from collections import OrderedDict

class ContentStore:
    """Disk shards + small RAM LRU. Serves the home_contents path."""
    def __init__(self, root: str, lru_size: int = 10_000):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._lru: OrderedDict[str, dict] = OrderedDict()
        self._lru_size = lru_size
        self.stats = {"hits": 0, "misses": 0}

    def _shard(self, doc_id: str) -> str:
        return os.path.join(self.root, f"shard-{abs(hash(doc_id)) % 64}.jsonl")

    def put(self, doc: dict) -> None:
        with open(self._shard(doc["id"]), "a", encoding="utf-8") as f:
            f.write(json.dumps(doc) + "\n")

    def get(self, doc_id: str) -> dict | None:
        if doc_id in self._lru:
            self.stats["hits"] += 1
            self._lru.move_to_end(doc_id)
            return self._lru[doc_id]
        path = self._shard(doc_id)
        if not os.path.exists(path):
            self.stats["misses"] += 1
            return None
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] == doc_id:
                    self.stats["hits"] += 1
                    self._lru[doc_id] = d
                    if len(self._lru) > self._lru_size:
                        self._lru.popitem(last=False)
                    return d
        self.stats["misses"] += 1
        return None
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_ingest.py tests/test_store.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add exa_home/ingest.py exa_home/store.py tests/test_ingest.py tests/test_store.py
git commit -m "feat: WET ingester with digest dedup + LRU content store"
```

---

### Task 3: Embedder (frozen Arctic-m + hash stub)

**Files:**
- Create: `exa_home/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `embed.Embedder(model="Snowflake/snowflake-arctic-embed-m-v2.0", dim=256)` with `.encode_queries(list[str]) -> np.ndarray[float32]`, `.encode_docs(list[str]) -> np.ndarray[float32]` (Arctic ST prompt contract: query via `prompt_name="query"`; doc-side prefix TBD on-box, likely unprefixed — marked VERIFY ON BOX; native MRL-256 truncation, slice+renormalize stays belt-and-braces for model swaps); `embed.HashEmbedder(dim=256)` with same methods, deterministic, Tier-0 safe; module-level `get_embedder()` singleton (persistent model — no cold loads at query time).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_embed.py
import numpy as np
import pytest
from exa_home.embed import HashEmbedder, get_embedder

def test_hash_embedder_shape_and_deterministic():
    e = HashEmbedder(dim=256)
    a = e.encode_queries(["hello world"])
    b = e.encode_queries(["hello world"])
    assert a.shape == (1, 256) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert not np.array_equal(a, e.encode_queries(["goodbye"]))

@pytest.mark.cloud_only
def test_real_embedder_truncates_to_256():
    e = get_embedder()
    v = e.encode_queries(["what is a vector database?"])
    assert v.shape == (1, 256) and v.dtype == np.float32
    assert abs(np.linalg.norm(v[0]) - 1.0) < 1e-3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_embed.py -v -m "not cloud_only"`
Expected: FAIL with ImportError on `exa_home.embed`.

- [ ] **Step 3: Implement**

```python
# exa_home/embed.py
from __future__ import annotations
import hashlib
import numpy as np

# Arctic-m-v2.0 prompt contract (VERIFY ON BOX: confirm doc side needs no prefix).
# ST usage: model.encode(texts, prompt_name="query") for queries, plain for docs.
QUERY_KWARGS = {"prompt_name": "query"}
DOC_KWARGS = {}

class HashEmbedder:
    """Deterministic stub for Tier 0/1. Same interface, no model."""
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.normal(size=self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one("query: " + t) for t in texts])

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts])

_singleton = None

def get_embedder(model: str = "Snowflake/snowflake-arctic-embed-m-v2.0", dim: int = 256):
    """Persistent singleton. First call loads GPU model; never call per-query cold."""
    global _singleton
    if _singleton is None:
        _singleton = Embedder(model, dim)
    return _singleton

class Embedder:
    def __init__(self, model: str, dim: int):
        from sentence_transformers import SentenceTransformer
        self.dim = dim
        # Native ST model: no trust_remote_code unless on-box load proves otherwise.
        # truncate_dim=256 maps to the model's native two-stage MRL-256 point.
        self.model = SentenceTransformer(model, truncate_dim=dim)

    def _encode(self, texts: list[str], **prompt_kwargs) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True,
                              **prompt_kwargs)
        return v[:, : self.dim].astype(np.float32) / (
            np.linalg.norm(v[:, : self.dim], axis=1, keepdims=True) + 1e-12)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, **QUERY_KWARGS)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, **DOC_KWARGS)
```

Note: `truncate_dim=256` hits the model's native two-stage MRL-256 point (~99% retention);
the slice + renormalize is belt-and-braces only for model swaps.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_embed.py -v -m "not cloud_only"`
Expected: PASS (cloud_only test deselected locally).

- [ ] **Step 5: Commit**

```bash
git add exa_home/embed.py tests/test_embed.py
git commit -m "feat: frozen Arctic-m embedder (native MRL-256) + hash stub"
```

---

### Task 4: ANN core M1 — float brute force + PyO3 + recall harness

**Files:**
- Create/Modify: `crates/ann-core/src/quant.rs`, `crates/ann-core/src/index.rs`, rewrite `crates/ann-core/src/lib.rs`, `scripts/ground_truth.py`, `exa_home/index_format.py`
- Test: `tests/test_ann.py` (grows in later tasks), Rust `#[cfg(test)]` unit tests in `quant.rs`

**Interfaces:**
- Consumes: `synth.make_vectors` (Task 1).
- Produces: `ann_core.AnnIndex(rows: np.ndarray[float32], ids: list[str])`, `.search(query: np.ndarray, top_k: int) -> (ids: list[str], scores: list[float])` (exact float dot, M1 baseline); `scripts/ground_truth.py` writing `gt.jsonl` (`{query_id, top10_ids}`); `index_format.write_manifest(dir, meta)` / `verify_manifest(dir)` (sha256 per file, fail-fast).

- [ ] **Step 1: Write failing Python test (M1 contract)**

```python
# tests/test_ann.py
import numpy as np
from fixtures.synth import make_vectors

def test_m1_brute_force_finds_nearest():
    from ann_core import AnnIndex
    vecs = make_vectors(200, 256, seed=7)
    ids = [f"d{i}" for i in range(200)]
    idx = AnnIndex(vecs, ids)
    q = vecs[42].copy()
    got_ids, scores = idx.search(q, top_k=5)
    assert got_ids[0] == "d42"
    assert scores[0] > scores[-1]
    # agreement with numpy exact
    exact = np.argsort(-(vecs @ q), kind="stable")[:5]
    assert got_ids == [f"d{i}" for i in exact]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ann.py -v` (after `maturin develop` stub? No — module doesn't exist yet.)
Expected: FAIL with `ModuleNotFoundError: No module named 'ann_core'`.

- [ ] **Step 3: Implement quant.rs (float dot) + lib.rs**

```rust
// crates/ann-core/src/quant.rs
/// M1: exact float dot product. Later milestones add binary paths beside this.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dot_unit_vectors() {
        assert!((dot(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(dot(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }
}
```

```rust
// crates/ann-core/src/lib.rs
mod index;
mod quant;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{prelude::*, Bound, types::PyModule};

#[pyclass]
struct AnnIndex {
    rows: Vec<f32>,
    n: usize,
    dim: usize,
    ids: Vec<String>,
}

#[pymethods]
impl AnnIndex {
    #[new]
    fn new(mat: PyReadonlyArray2<f32>, ids: Vec<String>) -> PyResult<Self> {
        let a = mat.as_array();
        Ok(Self {
            rows: a.as_slice().unwrap().to_vec(),
            n: a.nrows(),
            dim: a.ncols(),
            ids,
        })
    }

    fn search(&self, py: Python<'_>, q: PyReadonlyArray1<f32>, top_k: usize)
        -> PyResult<(Vec<String>, Vec<f32>)>
    {
        py.allow_threads(|| {
            let q = q.as_slice().unwrap();
            assert_eq!(q.len(), self.dim);
            let mut scored: Vec<(u32, f32)> = (0..self.n as u32)
                .map(|i| {
                    let row = &self.rows[i as usize * self.dim..(i as usize + 1) * self.dim];
                    (i, quant::dot(row, q))
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.truncate(top_k);
            Ok((scored.iter().map(|(i, _)| self.ids[*i as usize].clone()).collect(),
                scored.iter().map(|(_, s)| *s).collect()))
        })
    }
}

#[pymodule]
fn ann_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AnnIndex>()?;
    Ok(())
}
```

Build: `maturin develop` (creates importable `ann_core`).

- [ ] **Step 4: Write ground-truth script + index manifest helper**

```python
# scripts/ground_truth.py
"""Brute-force exact top-10 for sample queries. Usage:
python scripts/ground_truth.py --vecs vecs.npy --ids ids.json --nq 1000 --out gt.jsonl
"""
import argparse, json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs", required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--nq", type=int, default=1000); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    vecs = np.load(a["vecs"])
    ids = json.load(open(a["ids"]))
    rng = np.random.default_rng(a.seed)
    qi = rng.choice(len(vecs), size=min(a.nq, len(vecs)), replace=False)
    with open(a["out"], "w") as f:
        for k, i in enumerate(qi):
            top = np.argsort(-(vecs @ vecs[i]))[:10]
            f.write(json.dumps({"query_id": k, "top10": [ids[j] for j in top]}) + "\n")

if __name__ == "__main__":
    main()
```

```python
# exa_home/index_format.py
"""Versioned index/ dir: manifest.json with sha256 per file. verify fails fast."""
from __future__ import annotations
import hashlib, json, os

MANIFEST = "manifest.json"

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_manifest(index_dir: str, meta: dict) -> None:
    files = {}
    for name in sorted(os.listdir(index_dir)):
        if name == MANIFEST:
            continue
        files[name] = sha256(os.path.join(index_dir, name))
    with open(os.path.join(index_dir, MANIFEST), "w") as f:
        json.dump({"version": 1, **meta, "files": files}, f, indent=2)

def verify_manifest(index_dir: str) -> dict:
    with open(os.path.join(index_dir, MANIFEST)) as f:
        m = json.load(f)
    for name, want in m["files"].items():
        got = sha256(os.path.join(index_dir, name))
        if got != want:
            raise ValueError(f"corrupt index file {name}: want {want[:12]} got {got[:12]}")
    return m
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p ann-core` then `maturin develop && pytest tests/test_ann.py -v`
Expected: PASS both. Record the M1 row in `EXPERIMENTS.md` (create file with table header from spec §9 + M1 latency measured by `cargo bench` later — for now wall-time from the test).

- [ ] **Step 6: Commit**

```bash
git add crates/ann-core exa_home/index_format.py scripts/ground_truth.py tests/test_ann.py EXPERIMENTS.md
git commit -m "feat: ANN M1 float brute force + ground-truth harness + index manifest"
```

---

### Task 5: M2 — binary quantization

**Files:**
- Modify: `crates/ann-core/src/quant.rs`, `crates/ann-core/src/lib.rs`
- Test: extend `tests/test_ann.py`, Rust unit tests in `quant.rs`

**Interfaces:**
- Consumes: `AnnIndex` (Task 4).
- Produces: `quant::binarize(row: &[f32]) -> [u8; 32]` (256 dims → 32 bytes, bit=sign), `quant::binary_dot_packed(doc: &[u8;32], q: &[f32]) -> f32` (naive per-bit loop; LUT replaces it in M4); Python-visible `AnnIndex.from_binary(codes: np.ndarray[uint8], ids)` + `search_binary(query, top_k)`; `scripts/measure_recall.py` (`--gt gt.jsonl --pred pred.jsonl` → recall@10).

- [ ] **Step 1: Write failing tests**

```python
def test_m2_binary_search_agrees_on_easy_query():
    from ann_core import AnnIndex
    vecs = make_vectors(200, 256, seed=7)
    ids = [f"d{i}" for i in range(200)]
    codes = (vecs > 0).astype(np.uint8)
    pack = np.packbits(codes, axis=1)  # 200×32
    idx = AnnIndex.from_binary(pack, ids)
    got_ids, _ = idx.search_binary(vecs[42], top_k=10)
    assert "d42" in got_ids  # easy self-query must survive quantization
```

```python
# scripts/measure_recall.py (new file, tested via CLI on synth data)
"""recall@10 = |pred ∩ exact| / 10 averaged over gt.jsonl queries."""
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ann.py::test_m2_binary_search_agrees_on_easy_query -v`
Expected: FAIL with `AttributeError` (no `from_binary`).

- [ ] **Step 3: Implement Rust side**

```rust
// add to quant.rs
/// 256-dim float row -> 32 bytes, bit i = (x[i] > 0).
pub fn binarize(row: &[f32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, x) in row.iter().enumerate() {
        if *x > 0.0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// Naive packed dot: bit=1 -> +q[i], else -> -q[i]. M4 replaces inner loop with LUT.
pub fn binary_dot_packed(doc: &[u8; 32], q: &[f32]) -> f32 {
    let mut s = 0.0;
    for (i, v) in q.iter().enumerate() {
        if (doc[i / 8] >> (i % 8)) & 1 == 1 {
            s += v;
        } else {
            s -= v;
        }
    }
    s
}
```

`lib.rs`: add `codes: Option<Vec<[u8; 32]>>` field (or a second constructor storing packed codes + a `search_binary` method mirroring `search` but calling `binary_dot_packed`). Exact wiring left to implementer; contract: `from_binary(pack: np.ndarray[uint8] (n×32), ids)` + `search_binary(q: np.ndarray[float32], top_k: int)`.

- [ ] **Step 4: Implement measure_recall.py + record M2**

```python
# scripts/measure_recall.py
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True); ap.add_argument("--pred", required=True)
a = ap.parse_args()
gt = [json.loads(l) for l in open(a.gt)]
pr = {json.loads(l)["query_id"]: json.loads(l)["top10"] for l in open(a.pred)}
recs = [len(set(g["top10"]) & set(pr[g["query_id"]])) / 10 for g in gt]
print(f"recall@10 = {sum(recs)/len(recs):.4f} over {len(recs)} queries")
```

Run: build a 500-doc synth corpus, generate gt via `ground_truth.py`, predict with `search_binary`, measure. Append M2 row to `EXPERIMENTS.md` (expect: ~10–30x less memory per vector, recall@10 dip vs M1 — that dip is what M5 later recovers).

- [ ] **Step 5: Run tests**

Run: `cargo test -p ann-core && maturin develop && pytest tests/test_ann.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/ann-core tests/test_ann.py scripts/measure_recall.py EXPERIMENTS.md
git commit -m "feat: ANN M2 binary quantization + recall@10 harness"
```

---

### Task 6: M3 — IVF routing (centroid trainer + inverted lists + index format)

**Files:**
- Create: `scripts/train_centroids.py`, `crates/ann-core/src/ivf.rs`
- Modify: `crates/ann-core/src/lib.rs` (IvfIndex class), `crates/ann-core/src/index.rs` (loader for versioned dir), `exa_home/index_format.py` (schema constants)
- Test: extend `tests/test_ann.py`, new `tests/test_index_format.py`

**Interfaces:**
- Consumes: `AnnIndex` search contract, manifest helpers (Task 4).
- Produces: `train_centroids.py --vecs V --k K --sample S --out centroids.npy`; on-disk layout `centroids.f32 (K×256 LE)`, `codes.bin (n×32)`, `lists.bin (K posting lists of u32 row ids)`, `doc_ids.json`, `manifest.json`; `ann_core.IvfIndex.load(dir)`, `.search(q, nprobe, top_k, allow: list[int] | None) -> (ids, scores)`; `index_format.SCHEMA = {...}` constants shared by both sides.

Index layout decision (locked here, both languages implement to it):
```
index/
  manifest.json   {version:1, dim:256, n_docs, n_centroids, files:{sha256}}
  centroids.f32   K×256 float32 LE
  codes.bin       n×32 bytes (M2 packing)
  lists.bin       u32 K, then per list: u32 len + len×u32 row ids
  doc_ids.json    ["sha1:...", ...] row order
```

- [ ] **Step 1: Write failing tests**

```python
def test_ivf_load_search(tmp_path):
    import json
    import numpy as np
    from fixtures.synth import make_vectors
    from scripts.train_centroids import main as train  # importable: guard with if __name__ == ...
    ...
```

Simpler contract test (no sklearn locally — mark training cloud_only, test loader with synthetic centroids):

```python
def test_ivf_exact_routing_matches_bruteforce(tmp_path):
    """nprobe=K must equal M2 brute force exactly."""
    from ann_core import IvfIndex
    import numpy as np, json
    from fixtures.synth import make_vectors
    vecs = make_vectors(300, 256, seed=3)
    ...write minimal index dir with K=8 random centroids + lists built by nearest assignment...
    idx = IvfIndex.load(str(tmp_path))
    got, _ = idx.search(vecs[0], nprobe=8, top_k=10)
    ...compare vs from_binary brute force top-10...

def test_manifest_verify_rejects_corruption(tmp_path):
    from exa_home.index_format import write_manifest, verify_manifest
    (tmp_path / "a.bin").write_bytes(b"hello")
    write_manifest(str(tmp_path), {"dim": 256})
    (tmp_path / "a.bin").write_bytes(b"evil!")
    try:
        verify_manifest(str(tmp_path))
        assert False, "should have raised"
    except ValueError:
        pass
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ann.py tests/test_index_format.py -v`
Expected: FAIL (no `IvfIndex`, no `index_format.SCHEMA`).

- [ ] **Step 3: Implement train_centroids.py**

```python
# scripts/train_centroids.py
"""MiniBatchKMeans on a sample -> centroids.npy. Cloud-only (needs real vecs)."""
import argparse
import numpy as np

def main(vecs_path: str, k: int, sample: int, out: str, seed: int = 0):
    from sklearn.cluster import MiniBatchKMeans
    vecs = np.load(vecs_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    take = rng.choice(len(vecs), size=min(sample, len(vecs)), replace=False)
    km = MiniBatchKMeans(n_clusters=k, batch_size=10_000, n_init=3, random_state=seed)
    km.fit(vecs[take])
    np.save(out, km.cluster_centers_.astype(np.float32))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs", required=True); ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--sample", type=int, default=500_000); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.vecs, a.k, a.sample, a.out)
```

- [ ] **Step 4: Implement ivf.rs + index.rs loader + IvfIndex binding**

`ivf.rs`: `route(centroids, q, nprobe) -> Vec<u32>` (exact float dot over K centroids, partial select top-nprobe); `search_lists(lists, codes, cluster_ids, q, top_k, allow: Option<&[u32 sorted]>)` — linear scan of posting lists with `binary_dot_packed`, skip rows not in `allow` via binary search (allow-list sorted once per query).

`index.rs`: `Loaded { centroids: Vec<f32>, k, codes: Vec<u8> (n×32, mmap-loaded), lists: Vec<Vec<u32>>, ids: Vec<String> }` + `load(dir) -> Result<Loaded>` that verifies the manifest first using the `sha2` dep already in Cargo.toml (Task 1). No new dependencies in this task.

`lib.rs`: `#[pyclass] IvfIndex { inner: Loaded }`, `load(dir)`, `search(q, nprobe, top_k, allow: Option<Vec<u32>>)`.

- [ ] **Step 5: nprobe sweep script + M3 record**

Extend `scripts/e2e_latency.py`? No — smaller: add `scripts/nprobe_sweep.py`? Fold into this task as a 20-line loop inside the commit (not a new file): run `search` at nprobe ∈ {1,2,4,8,16,K} on synth 2k-doc index, print recall@10 + ms. Append M3 row to EXPERIMENTS.md (expect: recall climbs with nprobe, latency ~linear; knee is the operating point).

- [ ] **Step 6: Run tests**

Run: `cargo test -p ann-core && maturin develop && pytest tests/test_ann.py tests/test_index_format.py -v -m "not cloud_only"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/ann-core scripts/train_centroids.py exa_home/index_format.py tests/
git commit -m "feat: ANN M3 IVF routing + versioned index format + manifest verify"
```

---

### Task 7: M4 — LUT dot-product scoring

**Files:**
- Create: `crates/ann-core/src/lut.rs`
- Modify: `crates/ann-core/src/ivf.rs` (use LUT in scan), `lib.rs` (no API change)
- Test: Rust unit test in `lut.rs` (LUT score == naive `binary_dot_packed` bit-exact within 1e-4), Python regression already covered by `test_ann.py` (scores must not change beyond tolerance — add assertion).

**Interfaces:**
- Consumes: `binary_dot_packed` (Task 5), IVF scan loop (Task 6).
- Produces: `lut::build(q: &[f32; 256]) -> [[f32; 16]; 64]`, `lut::score(doc: &[u8; 32], t: &[[f32;16];64]) -> f32`; IVF scan calls it; `search()` output unchanged in shape, faster.

Algorithm (Exa's length-4 trick, concrete): 256 bits = 64 nibbles. For nibble `g` covering dims `4g..4g+4`, table entry `e` (0–15) = Σ bit_j(e)·(+q or −q) over the 4 dims. Doc byte `b` holds 2 nibbles → 2 lookups per byte, 64 lookups per doc total vs 256 FMAs naive (4x fewer mults, register-resident tables).

- [ ] **Step 1: Write failing Rust test**

```rust
#[test]
fn lut_matches_naive() {
    let q: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
    let doc = crate::quant::binarize(&q); // self-query: all positive bits pattern
    let t = build(&q.try_into().unwrap());
    let s_lut = score(&doc, &t);
    let s_naive = crate::quant::binary_dot_packed(&doc, &q);
    assert!((s_lut - s_naive).abs() < 1e-3);
}
```

Run: `cargo test -p ann-core lut`
Expected: FAIL (module doesn't exist).

- [ ] **Step 2: Implement lut.rs + rewire scan**

```rust
// crates/ann-core/src/lut.rs
/// Per-query tables: 64 nibbles × 16 sign-combos.
pub fn build(q: &[f32; 256]) -> [[f32; 16]; 64] {
    let mut t = [[0.0f32; 16]; 64];
    for g in 0..64 {
        for e in 0..16 {
            let mut s = 0.0;
            for j in 0..4 {
                let v = q[4 * g + j];
                if (e >> j) & 1 == 1 { s += v; } else { s -= v; }
            }
            t[g][e] = s;
        }
    }
    t
}

/// 2 lookups per byte (low nibble, high nibble), 64 total.
pub fn score(doc: &[u8; 32], t: &[[f32; 16]; 64]) -> f32 {
    let mut s = 0.0;
    for (b, chunk) in doc.iter().enumerate() {
        s += t[2 * b][(chunk & 0x0F) as usize];
        s += t[2 * b + 1][(chunk >> 4) as usize];
    }
    s
}
```

Bit-order contract (must match `binarize`): doc bit `i` lives in byte `i/8` at position `i%8`. Nibble `2b` = dims `8b..8b+4` = low 4 bits of byte `b` ✓; nibble `2b+1` = high 4 bits ✓. In `build`, entry bit `j` corresponds to dim `4g+j` sign +1 ✓. Keep this comment in the code.

- [ ] **Step 3: Add Python no-regression assertion + bench**

Extend `tests/test_ann.py`:

```python
def test_m4_scores_match_m2():
    # same index via search() before/after is implicitly covered;
    # pin one known score vector on seeded data:
    from ann_core import IvfIndex
    ...load synth index (fixture builder shared with Task 6 test)...
    _, scores = idx.search(vecs[0], nprobe=K, top_k=3)
    assert all(isinstance(s, float) for s in scores)
```

(The strong check is the Rust `lut_matches_naive` test; Python pins API stability.)

- [ ] **Step 4: Run + record M4**

Run: `cargo test -p ann-core && maturin develop && pytest tests/test_ann.py -v`
Then `cargo bench -p ann-core` (benches file from Task 8? No — write the bench now, minimal):

Create `crates/ann-core/benches/search_bench.rs` here (criterion, 2k-doc synth-equivalent random data generated in-Rust with a tiny xorshift — no external files):

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_search(c: &mut Criterion) {
    // 2000 docs × 32B codes + 64 centroids, in-Rust PRNG; search nprobe=8 top-10
    c.bench_function("ivf_search_2k", |b| {
        b.iter(|| {
            // ... build once outside via lazy_static-ish setup in real code ...
        })
    });
}
```

(Implementer fills the setup; acceptance: `cargo bench -p ann-core` runs and prints ns/iter.) Append M4 row to EXPERIMENTS.md (bench before = git stash? Simpler: M4 row compares `binary_dot_packed` bench vs LUT bench — keep both functions and bench both.)

- [ ] **Step 5: Commit**

```bash
git add crates/ann-core tests/test_ann.py EXPERIMENTS.md
git commit -m "feat: ANN M4 LUT dot-product scoring (64x16 tables)"
```

---

### Task 8: Filter index + in-scan filtering

**Files:**
- Create: `crates/ann-core/src/filter.rs`, `scripts/build_filter.py`
- Modify: `lib.rs` (IvfIndex.search allow-param already exists from Task 6 — wire real bitsets), index layout += `filter/` files
- Test: extend `tests/test_ann.py` + Rust tests

**Interfaces:**
- Consumes: `IvfIndex.search(q, nprobe, top_k, allow)` (Task 6), `doc_ids.json` row order.
- Produces: `build_filter.py --docs docs.jsonl --out filter/` writing `domains.bin` (roaring serialized per domain → concatenated with offset table `domains.json {domain: [offset, len]}`), `dates.bin` + `dates.json` (per-month buckets `YYYY-MM`), `terms.bin` + `terms.json` (top-50k keyword df terms); Rust `filter::load(dir) -> FilterIdx`, `FilterIdx.resolve(expr: {domains?, month_range?, terms?}) -> RoaringBitmap`; `search()` intersects allow-set during scan (already the `allow` param — now fed by real bitsets instead of tests).

Keep V1 terms simple: lowercase alphanumeric tokens, length ≥ 3, stop at 50k most frequent.

- [ ] **Step 1: Write failing tests**

```python
def test_filter_domains_restricts_results():
    ...build tiny index (3 docs, 2 domains) + filter files via build_filter.build(docs, out)...
    idx = IvfIndex.load(d)
    all_ids, _ = idx.search(q, nprobe=K, top_k=10)
    dom_ids, _ = idx.search(q, nprobe=K, top_k=10, domains=["example.com"])
    assert set(dom_ids) <= set(all_ids)
    assert all(i in example_ids for i in dom_ids)
```

Rust: `filter::resolve` unit test on hand-built bitmaps.

Python binding shape decision: `search(q, nprobe, top_k, allow=None, domains=None, month_range=None, terms=None)` — Rust resolves filter files internally. (Cleaner than passing bitmaps over FFI.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ann.py -v -k filter`
Expected: FAIL (no `build_filter`, no `domains` kwarg).

- [ ] **Step 3: Implement build_filter.py**

```python
# scripts/build_filter.py
"""docs.jsonl -> filter/{domains,dates,terms}.{bin,json}. Cloud-scale capable, Tier-0 testable."""
import argparse, json, re
from collections import Counter
from urllib.parse import urlparse

TOKEN = re.compile(r"[a-z0-9]{3,}")

def build(docs: list[dict], out_dir: str, max_terms: int = 50_000):
    from roaring import BitMap  # python-roaring; add to pyproject deps
    ...
```

Wait — python-roaring adds a dep for one script. Alternative: write roaring-compatible bitmaps by hand? No — real dep is honest: add `roaring>=0.4` (package `roaring`, `BitMap.serialize()` produces standard Roaring format the Rust `roaring` crate reads). Add to pyproject in this task.

- [ ] **Step 4: Implement filter.rs + rewire search kwargs**

```rust
// filter.rs: load {name}.json offset tables + mmap .bin; resolve expr -> RoaringBitmap
```

- [ ] **Step 5: Run + record M6-partial**

Run: full Rust + Python suites. Append filter row to EXPERIMENTS.md (latency delta of filtered vs unfiltered scan — expect near-zero: intersect during scan is the Exa graph-vs-IVF point).

- [ ] **Step 6: Commit**

```bash
git add crates/ann-core scripts/build_filter.py tests/test_ann.py EXPERIMENTS.md pyproject.toml
git commit -m "feat: inverted filter index with in-scan intersect"
```

---

### Task 9: Reranker + degraded fallback (M5)

**Files:**
- Create: `exa_home/rerank.py`
- Test: `tests/test_rerank.py`

**Interfaces:**
- Consumes: candidate ids + texts (from store), query string.
- Produces: `rerank.Reranker(model="cross-encoder/ms-marco-MiniLM-L6-v2", timeout_ms=100, batch_size=128)` with `.rerank(query: str, candidates: list[{id, text}]) -> list[{id, score}]` sorted desc; on timeout/exception returns input order with `degraded=True` (return tuple `(ranked, degraded)`); `rerank.MockReranker` (reverses input — proves ordering flows through). Batch 128 + pair truncation to 128–192 tokens + one warmup call + fp16: ~20–35K pairs/s on A100-class GPU → 200 pairs in ~6–10ms, inside the 40ms rerank slice.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rerank.py
from exa_home.rerank import MockReranker

def test_mock_reranker_reorders():
    r = MockReranker()
    cands = [{"id": "a", "text": "t1"}, {"id": "b", "text": "t2"}]
    ranked, degraded = r.rerank("q", cands)
    assert [c["id"] for c in ranked] == ["b", "a"] and degraded is False

def test_timeout_falls_back_degraded():
    from exa_home.rerank import Reranker
    r = Reranker.__new__(Reranker)  # no model load
    r.timeout_ms = 0  # force instant timeout path... (implementer: design for testability)
```

Better testability contract: `Reranker.rerank` takes `model_fn` injectable; test injects a sleeping fn and asserts fallback order + `degraded is True`. Write the test against that contract:

```python
def test_slow_model_falls_back_to_coarse_order():
    from exa_home.rerank import Reranker
    import time
    r = Reranker.__new__(Reranker)
    r.timeout_ms = 50
    r._score_batch = lambda q, texts: (time.sleep(0.5), [0.0] * len(texts))[1]
    cands = [{"id": "a", "text": "t1"}, {"id": "b", "text": "t2"}]
    ranked, degraded = r.rerank("q", cands)
    assert [c["id"] for c in ranked] == ["a", "b"] and degraded is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_rerank.py -v`
Expected: FAIL ImportError.

- [ ] **Step 3: Implement**

```python
# exa_home/rerank.py
from __future__ import annotations
import concurrent.futures

class MockReranker:
    def rerank(self, query, candidates):
        return list(reversed(candidates)), False

class Reranker:
    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L6-v2",
                 timeout_ms=100, batch_size=128, device=None, max_pair_tokens=160):
        from sentence_transformers import CrossEncoder
        import torch
        self.timeout_ms = timeout_ms
        self.batch_size = batch_size
        self.max_pair_tokens = max_pair_tokens  # truncate pairs to 128-192 tokens for budget
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model, device=dev)
        try:
            self.model.model.half()  # fp16: the 20-35K pairs/s figures assume it
        except Exception:
            pass
        # Warmup call: first predict pays CUDA init — never pay it on a live query.
        try:
            self.model.predict([["warmup query", "warmup passage"]])
        except Exception:
            pass
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _score_batch(self, query, texts):
        # Truncate each pair to max_pair_tokens so the 6-10ms/200-pair budget holds.
        pairs = [[query, " ".join(t.split()[: self.max_pair_tokens])] for t in texts]
        return self.model.predict(pairs, batch_size=self.batch_size,
                                  show_progress_bar=False).tolist()

    def rerank(self, query, candidates):
        if not candidates:
            return [], False
        fut = self._pool.submit(self._score_batch, query, [c["text"] for c in candidates])
        try:
            scores = fut.result(timeout=self.timeout_ms / 1000)
        except Exception:
            return list(candidates), True  # degraded: keep coarse ANN order
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [{**candidates[i], "score": float(scores[i])} for i in order], False
```

- [ ] **Step 4: Run tests + cloud shape check**

Run: `pytest tests/test_rerank.py -v -m "not cloud_only"`. Add a `cloud_only` test asserting real model loads and a 200-pair batch completes in < 40ms on GPU (expected ~6–10ms at batch 128 fp16; validates the 20–40ms budget claim on the actual box — fails loudly otherwise).

- [ ] **Step 5: Commit**

```bash
git add exa_home/rerank.py tests/test_rerank.py
git commit -m "feat: cross-encoder reranker with timeout degraded fallback"
```

---

### Task 10: Orchestrator Canon-lite + profile spans (M6)

**Files:**
- Create: `exa_home/orch.py`
- Test: `tests/test_orch.py`

**Interfaces:**
- Consumes: `Embedder`/`HashEmbedder`, `IvfIndex.search`, `Reranker.rerank`, `ContentStore.get`.
- Produces: `orch.Node(name, fn, deps)`, `orch.DAG(nodes)`, `orch.DAG.run(inputs) -> {outputs, spans: {name: ms}}`; `orch.build_search_dag(embedder, ann, reranker, store, top_coarse=200)`; `orch.run_search(dag, query, filters, top_k, profile) -> {results, profile?, degraded?}`. Cancellation: a node raising cancels dependents, keeps completed node outputs; error includes failing span.

Node graph for search: `embed → retrieve → [filter in retrieve] → rerank → snippet`. Mock-friendly: every backend behind a constructor arg.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orch.py
from exa_home.orch import DAG, Node

def test_dag_runs_in_dependency_order_and_memos():
    calls = []
    def a_fn(_): calls.append("a"); return 1
    def b_fn(x): calls.append("b"); return x["a"] + 1
    def c_fn(x): calls.append("c"); return (x["a"], x["b"])
    dag = DAG([Node("a", a_fn, []), Node("b", b_fn, ["a"]), Node("c", c_fn, ["a", "b"])])
    out = dag.run({})
    assert out["outputs"]["c"] == (1, 2) and calls == ["a", "b", "c"]

def test_failure_cancels_dependents_keeps_spans():
    def bad(_): raise RuntimeError("boom")
    dag = DAG([Node("a", lambda _: 1, []), Node("b", bad, ["a"]), Node("c", lambda _: 2, ["b"])])
    out = dag.run({})
    assert out["error"]["node"] == "b" and "a" in out["spans"] and "c" not in out["outputs"]

def test_search_waterfall_sums_to_wall():
    ...build_search_dag with HashEmbedder + tiny IvfIndex + MockReranker...
    res = run_search(dag, "topic 3", {}, top_k=5, profile=True)
    assert abs(sum(res["profile"].values()) - res["profile"]["total_ms"]) < res["profile"]["total_ms"] * 0.2 + 5
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_orch.py -v`
Expected: FAIL ImportError.

- [ ] **Step 3: Implement**

```python
# exa_home/orch.py
from __future__ import annotations
import time
from collections.abc import Callable
from dataclasses import dataclass, field

@dataclass
class Node:
    name: str
    fn: Callable[[dict], object]  # (dict of dep outputs) -> output
    deps: list[str] = field(default_factory=list)
    timeout_ms: int | None = None

class DAG:
    def __init__(self, nodes: list[Node]):
        self.nodes = {n.name: n for n in nodes}

    def run(self, inputs: dict) -> dict:
        outputs, spans, order = dict(inputs), {}, []
        pending = [n for n in self.nodes.values() if not n.deps or all(d in outputs for d in n.deps)]
        ...
```

Implementer completes topological execution (sequential V1 — parallelism is an M6 stretch recorded in EXPERIMENTS; keep V1 serial but span-instrumented so the parallel upgrade is measurable). Sequential V1 must be stated, not silently assumed — the M6 row compares serial vs threaded fan-out.

`build_search_dag` + `run_search` assemble the five nodes with budgets; snippet node = first 300 chars of stored text.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_orch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add exa_home/orch.py tests/test_orch.py
git commit -m "feat: Canon-lite DAG orchestrator with span waterfall"
```

---

### Task 11: MCP server + contract tests

**Files:**
- Create: `exa_home/mcp_server.py`
- Test: `tests/test_mcp.py`

**Interfaces:**
- Consumes: `orch.build_search_dag`, `run_search`, `ContentStore`.
- Produces: FastMCP app with `home_search(query, filters?, top_k=10, profile=False)`, `home_contents(ids, max_chars_per_doc=8000)`; pagination: `home_contents` caps at 20 ids/call, returns `next_offset` when truncated — never silent; `run_stdio()` entrypoint.

- [ ] **Step 1: Write failing contract tests (mocked backends)**

```python
# tests/test_mcp.py
from exa_home.mcp_server import app  # FastMCP app object

def test_home_search_contract():
    ...call tool fn directly with stub dag (monkeypatch build_search_dag)...
    assert set(res.keys()) >= {"results"}
    assert all(set(r) >= {"id", "url", "title", "snippet", "score"} for r in res["results"])

def test_home_search_rejects_empty_query():
    ...assert raises McpError / ValueError...

def test_home_contents_paginates():
    res = home_contents(list-of-25-ids)
    assert len(res["items"]) <= 20 and "next_offset" in res
```

(FastMCP tools are plain functions under the decorator — test the functions, not the transport.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_mcp.py -v`
Expected: FAIL ImportError.

- [ ] **Step 3: Implement**

```python
# exa_home/mcp_server.py
from __future__ import annotations
from fastmcp import FastMCP

app = FastMCP("exa-home")

_deps = {}  # wired by wire() at startup; tests inject stubs

def wire(embedder=None, ann=None, reranker=None, store=None):
    _deps.update(embedder=embedder, ann=ann, reranker=reranker, store=store)

@app.tool
def home_search(query: str, filters: dict | None = None, top_k: int = 10,
                profile: bool = False) -> dict:
    if not query or not query.strip():
        raise ValueError("query must be non-empty")
    from .orch import build_search_dag, run_search
    dag = build_search_dag(_deps["embedder"], _deps["ann"], _deps["reranker"], _deps["store"])
    return run_search(dag, query, filters or {}, top_k=top_k, profile=profile)

@app.tool
def home_contents(ids: list[str], max_chars_per_doc: int = 8000) -> dict:
    if not ids:
        raise ValueError("ids must be non-empty")
    page, rest = ids[:20], ids[20:]
    items = []
    for i in page:
        d = _deps["store"].get(i)
        if d is None:
            items.append({"id": i, "error": "not_found"})
        else:
            items.append({"id": i, "url": d["url"], "title": d.get("title", ""),
                          "text": d["text"][:max_chars_per_doc]})
    out = {"items": items}
    if rest:
        out["next_offset"] = len(page)
        out["remaining"] = len(rest)
    return out

def run_stdio():
    app.run()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_mcp.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add exa_home/mcp_server.py tests/test_mcp.py
git commit -m "feat: MCP home_search + home_contents with pagination"
```

---

### Task 12: Benches + EXPERIMENTS.md + e2e latency script

**Files:**
- Create: `scripts/e2e_latency.py`
- Modify: `EXPERIMENTS.md` (fill all measured rows), `crates/ann-core/benches/search_bench.rs` (if thin, expand to concurrency 1/8/64 here)

**Interfaces:**
- Consumes: everything.
- Produces: `scripts/e2e_latency.py --index index/ --queries q.jsonl --k 10` printing p50/p99 per stage + total, asserting profile-sum ≈ wall; EXPERIMENTS.md M1–M6 table fully filled from cloud runs (local runs fill Tier-0 rows only, marked as such).

- [ ] **Step 1: Write e2e_latency.py**

```python
# scripts/e2e_latency.py
"""End-to-end p50/p99 with profile waterfall check. Cloud-only for real numbers."""
import argparse, json, statistics, time
...
def pct(xs, p): return sorted(xs)[min(len(xs)-1, int(p * len(xs)))]
...
# per query: t0; res = run_search(..., profile=True); t1
# assert abs(sum(stages) - wall) < 0.2*wall + 5ms else FAIL (waterfall integrity)
# print table: stage p50/p99 + total p50/p99; exit 1 if total p50 > budget (default 100ms)
```

Test: Tier-0 run against synth index passes trivially (budget overridden `--budget-ms 5000`); assert exit 0. `cloud_only` test asserts real budget on cloud box (fails loudly if >100ms — that's the point).

- [ ] **Step 2: Run Tier-0 e2e**

Run: build synth 500-doc index via existing pieces + `pytest tests/test_e2e_tier0.py -v` (new small test file allowed in this task).
Expected: PASS.

- [ ] **Step 3: Fill EXPERIMENTS.md + commit**

```bash
git add scripts/e2e_latency.py tests/test_e2e_tier0.py EXPERIMENTS.md crates/ann-core/benches
git commit -m "feat: e2e latency harness with waterfall integrity check"
```

---

### Task 13: Cloud runbook — slice → toy → full build → validation

**Files:**
- Create: `scripts/select_slice.py`, `scripts/download_wet.py`, `docs/RUNBOOK.md`
- Test: `cloud_only` smoke tests only (no local tests — document why in RUNBOOK)

**Interfaces:**
- Consumes: all scripts + components.
- Produces: `select_slice.py --crawl CC-MAIN-2026-34 --lang eng --limit-wet 4 --out wet_urls.txt` (DuckDB over `https://data.commoncrawl.org/cc-index/table/cc-main/warc/crawl=.../subset=warc/*.parquet`, hive_partitioning=1); `download_wet.py --urls wet_urls.txt --out data/wet/ --workers 16` (HTTPS resume via Range, retries); RUNBOOK with exact commands for: 10k toy (4 WET files → ingest → embed → train K=1024 → build index → gt 200 queries → e2e), 1M build (~130 WET, K=10k), 5M build (~620 WET, K=30k, centroid sample 500k), validation gate (p50 < 100ms AND recall@10 ≥ threshold set from M5 numbers).

- [ ] **Step 1: Write select_slice.py**

```python
# scripts/select_slice.py
"""DuckDB over the CC columnar index -> top-N English-dense WET URLs.
Cloud-only (network + data scale). Defaults locked to CC-MAIN-2026-34."""
import argparse
SQL = """
SELECT warc_filename, COUNT(*) AS n_eng FROM read_parquet(
  'https://data.commoncrawl.org/cc-index/table/cc-main/warc/crawl={crawl}/subset=warc/*.parquet',
  hive_partitioning=1)
WHERE subset = 'warc'
  AND fetch_status = 200
  AND content_mime_detected IN ('text/html', 'application/xhtml+xml')
  AND (content_languages = 'eng'
    OR content_languages LIKE 'eng,%'
    OR content_languages LIKE '%,eng'
    OR content_languages LIKE '%,eng,%')
GROUP BY warc_filename
ORDER BY n_eng DESC
LIMIT {lim}
"""
# Rank WET files by English density so 1M needs only ~130 files (~8GB),
# 5M ~620 files (~38GB). Map each warc_filename segment path /warc/ -> /wet/
# (sibling path; full list also at crawl-data/<crawl>/wet.paths.gz),
# prefix https://data.commoncrawl.org/, write wet_urls.txt.
```

Reference: columnar index layout and SQL patterns per [Common Crawl URL index docs](https://commoncrawl.org/url-index).

- [ ] **Step 2: Write download_wet.py (resume, workers, sha check optional)**

Standard `urllib` + `Range` resume + `ThreadPoolExecutor(16)` with retries (expect transient 403s on CloudFront); skip existing complete files; print total GB. HTTPS only — unsigned `s3://` has been 403 since 2022.

- [ ] **Step 3: Write docs/RUNBOOK.md with copy-paste commands**

Toy run (≈10k docs, ~30 min on the box) then full run, then validation gate. Include: index build command sequence (ingest → embed bulk → train centroids → assemble index/ via a `scripts/build_index.py`...).

Wait — `build_index.py` (assemble codes.bin/lists.bin/doc_ids.json/manifest from vecs + centroids) was never assigned! It belongs here as the assembler's first-class script (used by toy + full):

Create `scripts/build_index.py --vecs V.npy --ids ids.json --centroids C.npy --docs docs.jsonl --out index/` → writes `centroids.f32, codes.bin, lists.bin, doc_ids.json`, copies content shards, calls `write_manifest`. Tested cloud_only on toy; Tier-0 unit test with 200 synth docs (fast, no model) asserting `IvfIndex.load` round-trips.

- [ ] **Step 4: Commit**

```bash
git add scripts/select_slice.py scripts/download_wet.py scripts/build_index.py docs/RUNBOOK.md
git commit -m "feat: cloud runbook + slice selection + index assembler"
```

---

## Execution order

Tasks 1→13 in order; each task's Interfaces.Produce is the next task's Consumes. Tasks 1–3 are Python-only (no Rust toolchain needed beyond the stub). First Rust behavior lands in Task 4. GPU/model touched only in `cloud_only` paths from Task 3 onward; full cloud run in Task 13.
