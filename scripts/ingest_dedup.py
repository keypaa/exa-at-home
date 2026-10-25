"""Disk-bounded URL dedup for stage_ingest.

Keeps memory at one chunk of URLs (default 50k) while still detecting duplicate
URLs across the whole corpus: chunk-sort, spill each chunk to a temp file, then
k-way merge on URL. Duplicate row indices (excluding the first occurrence in
sort order, which matches the original in-memory ``seen`` logic) come back as a
``bytearray`` indexed by source-row number.
"""
from __future__ import annotations

import heapq
import json
import os
import tempfile
from pathlib import Path


def mark_duplicate_rows(path: str, chunk_size: int = 50_000) -> bytearray:
    src = Path(path)
    chunks: list[tuple] = []
    # (url, row_index) tuples accumulate one chunk at a time, then spill.
    group: list[tuple[str, int]] = []
    n_rows = 0

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="ingest-dedup-")
        with open(src, encoding="utf-8") as f:
            for row_index, line in enumerate(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                url = d.get("url", "")
                group.append((url, row_index))
                n_rows = max(n_rows, row_index + 1)
                if len(group) >= chunk_size:
                    group.sort(key=lambda x: x[0])
                    out = os.path.join(tmpdir, f"chunk-{len(chunks):05d}.tsv")
                    with open(out, "w", encoding="utf-8") as g:
                        for url, ri in group:
                            g.write(f"{url}\t{ri}\n")
                    chunks.append(out)
                    group.clear()
        if group:
            group.sort(key=lambda x: x[0])
            out = os.path.join(tmpdir, f"chunk-{len(chunks):05d}.tsv")
            with open(out, "w", encoding="utf-8") as g:
                for url, ri in group:
                    g.write(f"{url}\t{ri}\n")
            chunks.append(out)
            group.clear()

        total_n = n_rows
        # Open each chunk and prime its first line.
        files: list[object] = []
        heap: list[tuple] = []
        for ci, cp in enumerate(chunks):
            fh = open(cp, encoding="utf-8")  # noqa: SIM115
            first = fh.readline()
            if first:
                url, ri = first.rstrip("\n").split("\t", 1)
                heap.append((url, int(ri), ci))
                files.append(fh)
            else:
                fh.close()

        heapq.heapify(heap)

        dup = bytearray(total_n)
        last_url: str | None = None
        while heap:
            url, ri, ci = heapq.heappop(heap)
            if url == last_url:
                dup[ri] = 1
            else:
                last_url = url
            # advance
            nxt = files[ci].readline()
            if nxt:
                u, r = nxt.rstrip("\n").split("\t", 1)
                heapq.heappush(heap, (u, int(r), ci))
            else:
                files[ci].close()
                files[ci] = None
    finally:
        if chunks:
            for cp in chunks:
                _safe_remove(cp)
        # remove tempdir shallowly
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
    return dup


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass