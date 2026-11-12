# exa_home/ingest.py
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

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
        tail = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    except ValueError:
        return ""
    if not tail:
        return ""
    base = "".join(ch for ch in tail if ch.isprintable()).rsplit(".", 1)[0]
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


def write_docs_jsonl(
    docs: Iterator[RawDoc], out_path: str, progress_every: int = 50_000
) -> IngestStats:
    from collections import Counter

    stats, reasons = IngestStats(), Counter()
    n_processed = 0
    # Dedup used to be an in-memory `set` of every digest: at 1M+ WET
    # records that set is hundreds of MB and grows with the corpus
    # (finding P2-05). A temp SQLite table keeps RAM bounded; the DB lives
    # in a TemporaryDirectory and is deleted when the ingest finishes.
    with tempfile.TemporaryDirectory(prefix="exa-ingest-dedup-") as tmpdir:
        conn = sqlite3.connect(os.path.join(tmpdir, "seen.sqlite"))
        try:
            # This DB only dedups within one run; crash-safety is irrelevant.
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY) WITHOUT ROWID")
            with open(out_path, "w", encoding="utf-8") as f:
                for d in docs:
                    n_processed += 1
                    if progress_every and n_processed % progress_every == 0:
                        print(
                            f"[ingest] records={n_processed} kept={stats.kept} "
                            f"skipped={stats.skipped}",
                            flush=True,
                        )
                    if d.error:
                        # The reader could not turn this record into a
                        # document: count it, or the loss never shows up in
                        # the stats.
                        stats.skipped += 1
                        reasons[d.error] += 1
                        continue
                    # INSERT OR IGNORE reports rowcount=0 for a duplicate;
                    # the first occurrence wins, as the set did.
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO seen(digest) VALUES (?)",
                        (d.digest,),
                    )
                    if cur.rowcount == 0:
                        stats.skipped += 1
                        reasons["duplicate"] += 1
                        continue
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
        finally:
            conn.close()
    stats.reasons = dict(reasons)
    if n_processed:
        print(
            f"[ingest] done: {n_processed} records -> {stats.kept} kept, "
            f"{stats.skipped} skipped",
            flush=True,
        )
    return stats
