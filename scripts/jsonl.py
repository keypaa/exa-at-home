# scripts/jsonl.py
"""One JSONL reader for every script (blank-line tolerant).

The pipeline used to open + strip + json.loads the same JSONL files eight
different ways, and the blank-line handling differed between them (some
skipped empty lines, some raised). This module is the single home:

    read_jsonl(path)  -> Iterator[dict]   lazy, one dict per non-blank line
    JsonlFile(path)   -> re-iterable lazy view (len() + iteration)

`read_jsonl` streams: the corpus is never materialised as a list of dicts
(GBs at 5M docs). `JsonlFile` adds `len()` (one cached scan) and can be
iterated more than once, which the two-pass filter builder needs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator


def read_jsonl(path: str) -> Iterator[dict]:
    """Yield one dict per non-blank line of a JSONL file.

    Blank lines are skipped (the corpus writers leave them behind). A
    malformed line raises ValueError naming the file and the line number
    (json.JSONDecodeError is a ValueError, so old `except ValueError`
    handlers keep working).
    """
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON line: {e}") from e


class JsonlFile:
    """Lazy, re-iterable view over a JSONL file.

    `len()` counts the non-blank lines (one cheap scan, cached) and
    iteration yields the parsed dicts. Re-iterable, so a two-pass
    consumer (scripts/build_filter.build) can stream the same file twice
    without materialising it.
    """

    def __init__(self, path: str):
        self.path = path
        self._n = -1

    def __len__(self) -> int:
        if self._n < 0:
            n = 0
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n += 1
            self._n = n
        return self._n

    def __iter__(self) -> Iterator[dict]:
        return read_jsonl(self.path)

    def __repr__(self) -> str:
        return f"JsonlFile({self.path!r})"
