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
"""
import argparse
import json
import os
import re
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
    if host.startswith("www."):
        host = host[4:]
    return host


def month_of(date: str) -> str:
    return _month_key(date)


def _write_postings(out_dir: str, name: str, postings: dict[str, list[int]]) -> int:
    from pyroaring import BitMap
    blob = bytearray()
    offsets: dict[str, list[int]] = {}
    for key in sorted(postings):
        # Row lists are appended in doc order, so each is already ascending.
        data = BitMap(postings[key]).serialize()
        offsets[key] = [len(blob), len(data)]
        blob += data
    with open(os.path.join(out_dir, name + ".bin"), "wb") as f:
        f.write(bytes(blob))
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(offsets, f, indent=2, sort_keys=True)
    return len(offsets)


def build(docs: list[dict], out_dir: str, max_terms: int = 50_000) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    domains: dict[str, list[int]] = {}
    dates: dict[str, list[int]] = {}
    df: Counter = Counter()
    doc_tokens: list[set[str]] = []
    for i, d in enumerate(docs):
        dom = domain_of(d.get("url", ""))
        if dom:
            domains.setdefault(dom, []).append(i)
        month = month_of(d.get("date", ""))
        if month:
            dates.setdefault(month, []).append(i)
        toks = set(tokenize(d.get("text", "")))
        doc_tokens.append(toks)
        df.update(toks)
    keep = {t for t, _ in sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[:max_terms]}
    terms: dict[str, list[int]] = {}
    for i, toks in enumerate(doc_tokens):
        for t in toks:
            if t in keep:
                terms.setdefault(t, []).append(i)
    counts = {
        "domains": _write_postings(out_dir, "domains", domains),
        "dates": _write_postings(out_dir, "dates", dates),
        "terms": _write_postings(out_dir, "terms", terms),
        "n_docs": len(docs),
    }
    with open(os.path.join(out_dir, "filter_stats.json"), "w") as f:
        json.dump(counts, f, indent=2, sort_keys=True)
    return counts


def main(docs_path: str, out: str, max_terms: int, seed: int = 0):  # noqa: ARG001
    docs = []
    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    build(docs, out, max_terms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-terms", type=int, default=50_000)
    a = ap.parse_args()
    main(a.docs, a.out, a.max_terms)
