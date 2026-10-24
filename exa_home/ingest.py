# exa_home/ingest.py
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import urlsplit

from warcio.archiveiterator import ArchiveIterator


@dataclass
class RawDoc:
    id: str  # content-digest based stable id
    url: str
    title: str  # "" for WET (title comes from WARC upgrade path later)
    text: str
    date: str  # "" for WET v1
    digest: str
    # Non-empty when the record could NOT be turned into a document
    # (undecodable payload). The reader yields it instead of dropping it
    # silently, so write_docs_jsonl can COUNT the loss.
    error: str = ""


@dataclass
class IngestStats:
    kept: int = 0
    skipped: int = 0
    reasons: dict | None = None


def _title_from_url(url: str) -> str:
    """Derive a usable heading from a URL slug (WET has no <title>).

    Slugs like ``/how-to-build-an-ann.html`` become ``How To Build An Ann``.
    Returns "" for URLs without a last path segment (the caller falls
    back to the first short line of the body if needed).
    """
    if not url:
        return ""
    try:
        tail = urlsplit(url).path.rsplit("/", 1)[-1]
    except ValueError:
        return ""
    if not tail:
        return ""
    base = tail.rsplit(".", 1)[0]
    parts = [p for p in base.replace("_", " ").replace("-", " ").split() if p]
    if not parts or parts == [base] and len(base) > 60:
        return ""
    return " ".join(part[:1].upper() + part[1:] for part in parts)



def read_wet(path: str) -> Iterator[RawDoc]:
    """Yield RawDoc records from a WET file.

    Payloads that cannot be decoded (or that are empty) are yielded with
    ``error`` set instead of being dropped: dropping them inside the
    generator hid the loss from the stats (the reported kept/skipped
    counts understated it).
    """
    with open(path, "rb") as f:
        for rec in ArchiveIterator(f):
            if rec.rec_type != "conversion":
                continue
            url = rec.rec_headers.get_header("WARC-Target-URI")
            payload = rec.content_stream().read()
            digest = hashlib.sha1(payload).hexdigest()
            try:
                text = payload.decode("utf-8", errors="strict").strip()
            except UnicodeDecodeError:
                yield RawDoc(
                    id="",
                    url=url,
                    title="",
                    text="",
                    date="",
                    digest=digest,
                    error="undecodable",
                )
                continue
            if not text:
                yield RawDoc(
                    id="",
                    url=url,
                    title="",
                    text="",
                    date="",
                    digest=digest,
                    error="empty",
                )
                continue
            yield RawDoc(
                id=f"sha1:{digest}",
                url=url,
                title=_title_from_url(url),
                text=text,
                date="",
                digest=digest,
            )


def write_docs_jsonl(docs: Iterator[RawDoc], out_path: str) -> IngestStats:
    from collections import Counter

    stats, reasons, seen = IngestStats(), Counter(), set()
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            if d.error:
                # The reader could not turn this record into a document:
                # count it, or the loss never shows up in the stats.
                stats.skipped += 1
                reasons[d.error] += 1
                continue
            if d.digest in seen:
                stats.skipped += 1
                reasons["duplicate"] += 1
                continue
            seen.add(d.digest)
            if len(d.text) < 50:
                stats.skipped += 1
                reasons["too_short"] += 1
                continue
            f.write(
                json.dumps(
                    {
                        "id": d.id,
                        "url": d.url,
                        "title": d.title,
                        "text": d.text,
                        "date": d.date,
                    }
                )
                + "\n"
            )
            stats.kept += 1
    stats.reasons = dict(reasons)
    return stats
