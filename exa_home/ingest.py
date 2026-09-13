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
