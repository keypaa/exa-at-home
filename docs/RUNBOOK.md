# Cloud RUNBOOK — slice → toy → full build → validation

Task 13 of 13. Scripts + docs only: no Rust, no library code.

**The entire flow below needs the cloud box.** The sandbox has no network
(DuckDB-over-HTTPS, WET downloads, `~90MB` HF model pulls, sklearn/warcio-heavy
paths cannot run there), and local CI never runs it: slice/download/train are
`cloud_only` by nature, and the gates need a CUDA GPU + the real index.
What the sandbox *does* verify offline is listed at the bottom.

Conventions: repo root is the working directory for every command. `index/`
and `q.jsonl` at root are the Task 12 hardcoded gate artifacts — the gate
commands in §6 produce exactly those (no overrides, no flags to relocate them).

---

## 0. Box setup — Rust, venv, binding (do these first, every fresh box)

```bash
git clone https://github.com/keypaa/exa-at-home.git && cd exa-at-home
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env" && cargo --version   # expect 1.9x
python -m venv .venv && source .venv/bin/activate
pip install -e .                               # NOT .[dev]: maturin below covers it
pip install "maturin>=1.0"
cd crates/ann-core && maturin develop && cd ../..
```

Three workarounds baked in from the local-toolchain investigation
(skip 2 entirely on Python ≤3.13 — the ABI gate only bites 3.14):

1. **PEP-668 systems** (externally managed Python): never `pip install`
   system-wide. Always:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e .[dev]
   ```
2. **Python 3.14 + PyO3 0.22**: the PyO3 0.22 ABI gate rejects 3.14 unless
   forward-compat is opted in. Prefix every cargo/maturin invocation:
   ```bash
   export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
   ```
3. **maturin runs from `crates/ann-core`**, not the workspace root (the root
   `Cargo.toml` is workspace-only, no `[package]`):
   ```bash
   cd crates/ann-core && maturin develop && cd ../..
   ```

Verify the toolchain before spending GPU hours:

```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
python -m pytest tests/ -m "not cloud_only" -q   # Tier-0, seconds on any CPU
cd crates/ann-core && cargo test -q && cd ../..   # Rust unit tests
```

---

## 1. Slice selection — top-N English-dense WET URLs

Locked values: crawl `CC-MAIN-2026-34`; columnar path
`https://data.commoncrawl.org/cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-34/subset=warc/*.parquet`
with `hive_partitioning=1`; SQL filters `subset=warc`, `fetch_status=200`,
`content_mime_detected IN (text/html, application/xhtml+xml)`, eng-variant
`content_languages` LIKEs; `GROUP BY warc_filename ORDER BY n_eng DESC`.

WARC→WET mapping: replace `/warc/` with `/wet/` in each segment path (sibling
path; the full list is also published at
`crawl-data/CC-MAIN-2026-34/wet.paths.gz` if you ever need to cross-check).

Slice sizes (researched, locked): **1M ≈ top-130 WET (~8GB)**;
**5M ≈ top-620 WET (~38GB)**; toy = 4 WET (~10k docs).

```bash
# toy slice (4 WET, ~10k docs)
# NOTE (verified on-box 2026-09-14/15): the columnar-index HTTPS path 404s —
# the cc-index/table prefix does not resolve over data.commoncrawl.org.
# Until the correct layout is found, take WET files straight from wet.paths.gz
# (no density ranking; fine for the toy — ingest dedups downstream):
curl -s "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-34/wet.paths.gz" \
    -o /tmp/wetpaths.gz --max-time 60 \
  && zcat /tmp/wetpaths.gz | head -4 \
  | sed 's|^|https://data.commoncrawl.org/|' > wet_urls.txt \
  && cat wet_urls.txt
# 1M slice (~130 WET, ~8GB) — BLOCKED on the columnar fix above; the toy
# workaround generalizes (head -130) but without English-density ranking
# you download ~2.5x more for the same English yield. Fix the layout first.
python scripts/select_slice.py --crawl CC-MAIN-2026-34 --lang eng \
    --limit-wet 130 --out wet_urls_1M.txt
# 5M slice (~620 WET, ~38GB) — same blocker as 1M.
python scripts/select_slice.py --crawl CC-MAIN-2026-34 --lang eng \
    --limit-wet 620 --out wet_urls_5M.txt
```

---

## 2. Download — HTTPS resume, 16 workers, retries

**HTTPS ONLY.** Unsigned `s3://` access has returned 403 since 2022 — never
use it. Expect transient CloudFront 403s; the script retries 5x with
exponential backoff. Resume is per-file HTTP `Range` from the on-disk size;
already-complete files are skipped, so re-running is safe.

```bash
python scripts/download_wet.py --urls wet_urls.txt --out data/wet/ --workers 16
# 1M / 5M: same command with --urls wet_urls_1M.txt / wet_urls_5M.txt
# (expect ~8GB / ~38GB; the script prints the total on completion)
```

---

## 3. Toy run — 4 WET files → gate (≈30 min on the box)

Dedup policy (locked): **content_digest first, URL second** — the
`sort -u` passes below implement URL-second; digest-first is inside
`write_docs_jsonl` (SHA-1 `WARC-Payload-Digest` keyed `seen` set). Excluded
throughout: `robotstxt` / `crawldiagnostics` subsets and truncated records
(`read_wet` only yields `conversion` records with decodable non-empty text).

```bash
# 3a. ingest: WET -> deduped docs.jsonl
python - <<'EOF'
import glob
from exa_home.ingest import read_wet, write_docs_jsonl
def all_docs():
    for p in sorted(glob.glob("data/wet/*.warc.wet.gz") + glob.glob("data/wet/*.wet.gz")):
        yield from read_wet(p)
stats = write_docs_jsonl(all_docs(), "docs.jsonl")
print(stats)
EOF
# URL-second dedup + subset/truncation exclusion (digest-first is in write_docs_jsonl)
grep -v -i -E 'robotstxt|crawldiagnostics' docs.jsonl > docs.clean.jsonl
python -c "import json; seen=set(); out=open('docs.dedup.jsonl','w');
[seen.add(d['url']) or out.write(l) for l in open('docs.clean.jsonl')
 for d in [json.loads(l)] if d['url'] not in seen]"
mv docs.dedup.jsonl docs.jsonl && wc -l docs.jsonl   # toy: ~10k

# 3b. embed bulk: docs.jsonl -> vecs.npy + ids.json (Arctic-m, native MRL-256)
python - <<'EOF'
import json
import numpy as np
from exa_home.embed import Embedder
docs = [json.loads(l) for l in open("docs.jsonl")]
emb = Embedder()  # Snowflake/snowflake-arctic-embed-m-v2.0, truncate_dim=256
B = 256
vecs = np.vstack([emb.encode_docs([d["text"][:12000] for d in docs[i:i+B]])
                  for i in range(0, len(docs), B)])
np.save("vecs.npy", vecs)
json.dump([d["id"] for d in docs], open("ids.json", "w"))
print(vecs.shape, vecs.dtype)
EOF

# 3c. train centroids (toy K=1024) + assemble index/
python scripts/train_centroids.py --vecs vecs.npy --k 1024 \
    --sample 500000 --out centroids.npy
python scripts/build_index.py --vecs vecs.npy --ids ids.json \
    --centroids centroids.npy --docs docs.jsonl --out index/ --filter
python -c "from exa_home.index_format import verify_manifest; print(verify_manifest('index/'))"

# 3d. ground truth (200 queries) + query file for the gate
python scripts/ground_truth.py --vecs vecs.npy --ids ids.json \
    --nq 200 --out gt.jsonl
python - <<'EOF'
import json
docs = [json.loads(l) for l in open("docs.jsonl")]
with open("q.jsonl", "w") as f:   # Task 12 gate path: q.jsonl at root
    for d in docs[:200]:
        f.write(json.dumps({"query": d["text"][:200]}) + "\n")
EOF

# 3e. validation gate (toy thresholds; full thresholds in §6)
python scripts/e2e_latency.py --ann ivf --index index/ --queries q.jsonl \
    --k 10 --budget-ms 100 --embedder arctic --reranker real
python scripts/measure_recall.py --gt gt.jsonl --pred pred.jsonl
```

One text slice per page: embed `text[:12000]` (~3k tokens, inside the card's
8192-token RoPE context with margin — verify the effective limit on-box);
full text lives in the content store. No chunking in V1.

---

## 4. Full builds — 1M (K=10k) and 5M (K=30k)

Same pipeline as §3 with locked scale parameters. Only the differences:

```bash
# 1M build (~130 WET, ~8GB): K=10k
python scripts/download_wet.py --urls wet_urls_1M.txt --out data/wet_1M/ --workers 16
# ... ingest (§3a) + embed bulk (§3b) on data/wet_1M/ ...
python scripts/train_centroids.py --vecs vecs.npy --k 10000 \
    --sample 500000 --out centroids.npy
python scripts/build_index.py --vecs vecs.npy --ids ids.json \
    --centroids centroids.npy --docs docs.jsonl --out index/ --filter

# 5M build (~620 WET, ~38GB): K=30k, centroid sample 500k
python scripts/download_wet.py --urls wet_urls_5M.txt --out data/wet_5M/ --workers 16
# ... ingest (§3a) + embed bulk (§3b) on data/wet_5M/ ...
python scripts/train_centroids.py --vecs vecs.npy --k 30000 \
    --sample 500000 --out centroids.npy
python scripts/build_index.py --vecs vecs.npy --ids ids.json \
    --centroids centroids.npy --docs docs.jsonl --out index/ --filter
```

`build_index.py` assigns centroids in 50k-row chunks, so the 5M build never
materializes the n×K score matrix. Manifest is written last and covers
`filter/` + store shards — corrupt index fails startup fast via
`verify_manifest`, never serves silent partial recall.

---

## 5. Query file + prediction file (gate inputs)

The Task 12 `cloud_only` gate hardcodes **`index/` + `q.jsonl` at root**.
Produce them exactly — no flag relocates them:

```bash
# q.jsonl: one {"query": ...} per line (200+ queries; see §3d for the toy cut,
# use held-out head queries or the eval set for the full builds)
# pred.jsonl: one {"query_id": k, "top10": [...]} per line from the serving stack
python - <<'EOF'
import json
from exa_home.orch import build_search_dag, run_search
from exa_home.embed import Embedder
from exa_home.rerank import Reranker
from exa_home.store import ContentStore
from ann_core import IvfIndex
dag = build_search_dag(Embedder(), IvfIndex.load("index/"), Reranker(),
                       ContentStore("index/store"))
queries = [json.loads(l)["query"] for l in open("q.jsonl")]
with open("pred.jsonl", "w") as f:
    for k, q in enumerate(queries):
        res = run_search(dag, q, {}, top_k=10)
        f.write(json.dumps({"query_id": k,
                            "top10": [r["id"] for r in res["results"]]}) + "\n")
print(f"{len(queries)} predictions -> pred.jsonl")
EOF
```

---

## 6. Validation gate + deferred measurements ledger

Gate (full builds): **p50 < 100ms AND recall@10 ≥ threshold set from the M5
numbers** (the rerank recovery number in (d) below fixes the exact threshold;
do not gate 5M recall on the toy 0.4460 dip — that dip is pre-rerank by
construction).

```bash
# e2e 100ms gate — Task 12's exact hardcoded paths (index/ + q.jsonl at root)
python scripts/e2e_latency.py --ann ivf --index index/ --queries q.jsonl \
    --k 10 --budget-ms 100 --embedder arctic --reranker real
# recall gate
python scripts/ground_truth.py --vecs vecs.npy --ids ids.json \
    --nq 1000 --out gt.jsonl
python scripts/measure_recall.py --gt gt.jsonl --pred pred.jsonl
# Rust absolutes for the M2/M3 EXPERIMENTS.md rows
cargo bench -p ann-core --bench search_bench   # run from repo root
```

Deferred measurements — all funnel through this runbook; none can land sooner:

- (pre) **Arctic doc-side prefix + trust_remote_code verifies**: confirm on
  box whether `encode_docs` needs a doc-side prefix (queries use
  `prompt_name="query"`) and whether the Arctic load proves out without
  `trust_remote_code` (see `exa_home/embed.py`); land the verdict in
  `EXPERIMENTS.md` before the full builds.

- (a) **Rerank <40ms** (`test_real_model_200_pairs_under_40ms`, cloud_only):
  needs CUDA + the ~90MB `cross-encoder/ms-marco-MiniLM-L6-v2` HF download.
  `pytest tests/test_rerank.py -m cloud_only -q`.
- (b) **E2E 100ms gate** (`test_real_index_under_100ms_budget`, cloud_only):
  needs `index/` + `q.jsonl` at root per Task 12's hardcoded paths — §§3–5
  produce exactly those. `pytest tests/test_e2e_tier0.py -m cloud_only -q`.
- (c) **`cargo bench` absolutes** for the M2/M3 EXPERIMENTS.md rows
  (`binary_dot_packed_scan_2k`, `lut_scan_2k`): `cargo bench -p ann-core
  --bench search_bench` on the box; record the before/after row.
- (d) **M5 recall recovery number**: `measure_recall.py` on the full build
  after rerank lands; sets the recall@10 gate threshold above.
- (e) **Waterfall tolerance**: tighten `e2e_latency.py` tol from 20%+5ms to
  10%+2ms if the on-box spans come back clean (Task 12 concern) — one-line
  change in `run_latency`, re-run the gate to confirm zero violations.

Each landed number gets a before/after row in `EXPERIMENTS.md` (spec §9);
cloud rows are marked **on-box**/**cloud**, never Tier-0.

---

## 7. Why no local tests (offline-green vs needs-cloud split)

- **Offline-green (this sandbox):** `tests/test_build_index.py` — 200-doc
  synth assemble asserting the full `IvfIndex.load` read contract via
  `exa_home.index_format` (no `ann_core` import, same pattern as Task 12);
  SQL string content + all three argparse CLIs (`--help`); download
  resume/skip logic with mocked transport. Run:
  `python -m pytest tests/test_build_index.py -m "not cloud_only" -q`.
- **Needs-cloud:** `test_select_slice_live_toy_run` (cloud_only, DuckDB-over-
  HTTPS, 4 densest WET URLs); the §6 gate commands; everything in (a)–(e).
  Documented here rather than faked — Tier-0 rows never stand in for cloud
  gates (EXPERIMENTS.md provenance rule).
