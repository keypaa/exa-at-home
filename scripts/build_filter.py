# scripts/build_filter.py
"""docs.jsonl -> filter/{domains,dates,terms}.{bin,json}. Cloud-scale capable, Tier-0 testable.

Usage: python scripts/build_filter.py --docs docs.jsonl --out filter/ [--max-terms N]

Layout (mirrors the locked index/ layout in exa_home/index_format.py):
  {domains,dates,terms}.bin    concatenated standard-Roaring serialized bitmaps
                               (pyroaring BitMap.serialize(); the Rust `roaring`
                               crate reads the same format)
  {domains,dates,terms}.json   {key: [offset, len]} slicing the bitmap for `key`
                               out of the matching .bin

Bitmap row-ids are u32 doc row indices in doc_ids.json row order (same row
space as lists.bin). Terms are lowercase alphanumeric tokens of length >= 3
taken from doc `text` (titles excluded in V1 — keep it simple), capped at the
`max_terms` most frequent by document frequency (df desc, term asc for ties).
Domains are lowercased hostnames with a leading `www.` stripped. Dates bucket
per YYYY-MM matched off the front of the doc `date` string ("" for WET v1, so
dates are sparse until the WARC upgrade path fills them in).

SCALE: the builder is a TWO-PASS streaming algorithm (see build()): the old
one held every document's token set in RAM (`doc_tokens: list[set[str]]`)
plus a df Counter plus the postings as Python lists — 0.84GB peak RSS at
50k docs, 3.26GB at 200k, ~16GB at 1M, ~80GB at 5M. Now only the df
Counter (O(vocabulary)) and the roaring bitmaps (O(postings), compressed)
are resident; the corpus itself is never materialised.
"""

import argparse
import contextlib
import json
import os
import re
import tempfile
from collections import Counter
from urllib.parse import urlparse

TOKEN = re.compile(r"[a-z0-9]{3,}")
MONTH = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])")


def _month_key(date: str) -> str:
    m = MONTH.match(date or "")
    return f"{m.group(1)}-{m.group(2)}" if m else ""


def tokenize(text: str) -> list[str]:
    return TOKEN.findall((text or "").lower())


def domain_of(url: str) -> str:
    host = urlparse(url or "").hostname or ""
    host = host.lower()
    host = host.removeprefix("www.")
    return host


def _write_postings(out_dir: str, name: str, bitmaps: dict) -> int:
    """Serialize the bitmaps into {name}.bin + the {name}.json offset table.

    The keys are written in sorted order and the offsets table is dumped
    with sort_keys=True, so the bytes are a deterministic function of the
    postings (the Rust reader slices the blob by offset+length).
    """
    blob = bytearray()
    offsets: dict[str, list[int]] = {}
    for key in sorted(bitmaps):
        data = bitmaps[key].serialize()
        offsets[key] = [len(blob), len(data)]
        blob += data
    with open(os.path.join(out_dir, name + ".bin"), "wb") as f:
        f.write(bytes(blob))
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(offsets, f, indent=2, sort_keys=True)
    return len(offsets)


@contextlib.contextmanager
def _doc_source(docs):
    """Yield a callable returning a FRESH iterator over `docs`.

    A re-iterable input (a list, a scripts.jsonl.JsonlFile) is used as is.
    A one-shot iterator is SPOOLED to a temp JSONL file first, so the two
    passes never need the corpus in RAM.
    """
    if iter(docs) is not iter(docs):
        yield lambda: iter(docs)
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "docs.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(d) + "\n" for d in docs)
        from scripts.jsonl import read_jsonl

        yield lambda: read_jsonl(path)


def build(docs, out_dir: str, max_terms: int = 50_000) -> dict:
    """docs (ANY iterable of dicts) -> the filter/ files. Returns the counts.

    TWO streaming passes:

      pass 1  document frequency per term — no per-doc token set is kept —
              then the top `max_terms` terms (df desc, term asc for ties)
      pass 2  the postings, accumulated DIRECTLY into pyroaring bitmaps
              (.add(i) per row id) — never into Python lists

    The corpus is never materialised: `docs` may be a generator (it is
    then spooled to a temp file so it can be read twice) or a lazy,
    re-iterable view. The on-disk layout is byte-identical to the old
    list-based build.
    """
    from pyroaring import BitMap

    os.makedirs(out_dir, exist_ok=True)
    with _doc_source(docs) as source:
        # Pass 1: document frequency (+ the single-valued facets, whose
        # postings are one row id per doc anyway).
        df: Counter = Counter()
        domains: dict[str, object] = {}
        dates: dict[str, object] = {}
        n_docs = 0
        for d in source():
            i = n_docs
            n_docs += 1
            dom = domain_of(d.get("url", ""))
            if dom:
                bm = domains.get(dom)
                if bm is None:
                    bm = domains[dom] = BitMap()
                bm.add(i)
            month = _month_key(d.get("date", ""))
            if month:
                bm = dates.get(month)
                if bm is None:
                    bm = dates[month] = BitMap()
                bm.add(i)
            df.update(set(tokenize(d.get("text", ""))))
        keep = {
            t for t, _ in sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[:max_terms]
        }
        del df  # the vocabulary counter is not needed any more

        # Pass 2: the term postings, straight into the bitmaps.
        terms: dict[str, object] = {}
        for i, d in enumerate(source()):
            for t in set(tokenize(d.get("text", ""))):
                if t in keep:
                    bm = terms.get(t)
                    if bm is None:
                        bm = terms[t] = BitMap()
                    bm.add(i)

    counts = {
        "domains": _write_postings(out_dir, "domains", domains),
        "dates": _write_postings(out_dir, "dates", dates),
        "terms": _write_postings(out_dir, "terms", terms),
        "n_docs": n_docs,
    }
    with open(os.path.join(out_dir, "filter_stats.json"), "w") as f:
        json.dump(counts, f, indent=2, sort_keys=True)
    return counts


def main(docs_path: str, out: str, max_terms: int):
    """docs.jsonl -> filter/. The corpus is streamed (never a list)."""
    from scripts.jsonl import JsonlFile

    build(JsonlFile(docs_path), out, max_terms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-terms", type=int, default=50_000)
    a = ap.parse_args()
    main(a.docs, a.out, a.max_terms)
