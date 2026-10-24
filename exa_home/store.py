# exa_home/store.py
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict

OFFSETS_FILENAME = "offsets.json"


class ContentStore:
    """Disk shards + small RAM LRU. Serves the home_contents path."""

    def __init__(self, root: str, lru_size: int = 10_000):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._lru: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.RLock()
        self._lru_size = lru_size
        self.stats = {"hits": 0, "misses": 0}
        # id -> (shard_path, byte_offset, record_length) index. Built
        # lazily on first miss and invalidated never (shards are
        # append-only per build; a fresh ContentStore per index build
        # starts empty anyway).
        # Without this, every cold get() scans full JSONL shards line by
        # line: O(shard) per lookup, ~28ms each at 81k docs — the 1400ms
        # "rerank" wall found on-box 2026-09-16 was really this.
        # V2: build_index.py writes store/offsets.json at build time
        # (offsets tracked during put, no rescan); _ensure_index loads
        # the file when present, falling back to the one-time scan for
        # old indexes without it.
        # V3: the index also records the shard size it was built from, so
        # a shard that GREW (a doc appended after save_index) or was
        # REBUILT is detected instead of silently serving a stale offset
        # (which resolved to a DIFFERENT document under the requested id).
        self._index: dict[str, tuple[str, int, int | None]] | None = None
        # shard_path -> byte size the index was built from (None = unknown,
        # e.g. a legacy offsets.json without record lengths).
        self._shard_sizes: dict[str, int | None] = {}

    def _shard(self, doc_id: str) -> str:
        return os.path.join(
            self.root,
            f"shard-{int(hashlib.sha1(doc_id.encode()).hexdigest()[:8], 16) % 64}.jsonl",
        )

    def begin_bulk(self) -> None:
        """Enable offset tracking for a bulk load (build_index path)."""
        with self._lock:
            if self._index is None:
                self._index = {}
                self._shard_sizes = {
                    path: os.path.getsize(path) for path in self._shard_paths()
                }

    def _shard_paths(self) -> list[str]:
        if not os.path.isdir(self.root):
            return []
        return [
            os.path.join(self.root, name)
            for name in sorted(os.listdir(self.root))
            if name.startswith("shard-") and name.endswith(".jsonl")
        ]

    def put(self, doc: dict) -> None:
        path = self._shard(doc["id"])
        payload = (json.dumps(doc) + "\n").encode("utf-8")
        with self._lock:
            if self._index is None:
                with open(path, "ab") as f:
                    f.write(payload)
                return
            with open(path, "a+b") as f:
                f.seek(0, os.SEEK_END)
                off = f.tell()
                f.write(payload)
            self._index[doc["id"]] = (path, off, len(payload))
            self._shard_sizes[path] = off + len(payload)

    def save_index(self) -> str:
        with self._lock:
            self._ensure_index()
            assert self._index is not None
            out_path = os.path.join(self.root, OFFSETS_FILENAME)
            serial = {}
            for doc_id, (path, off, length) in self._index.items():
                entry = [os.path.basename(path), off]
                if length is not None:
                    entry.append(length)
                serial[doc_id] = entry
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(serial, f)
            return out_path

    def _load_persisted_index(self) -> bool:
        """Load store/offsets.json if present. Returns True on success."""
        ipath = os.path.join(self.root, OFFSETS_FILENAME)
        if not os.path.exists(ipath):
            return False
        try:
            with open(ipath, encoding="utf-8") as f:
                serial = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return False
        if not isinstance(serial, dict):
            return False
        idx: dict[str, tuple[str, int, int | None]] = {}
        for doc_id, loc in serial.items():
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue
            try:
                shard = str(loc[0])
                off = int(loc[1])
                length = int(loc[2]) if len(loc) > 2 else None
            except (ValueError, TypeError):
                continue
            idx[str(doc_id)] = (os.path.join(self.root, shard), off, length)
        self._index = idx
        self._shard_sizes = self._sizes_from_index(idx)
        return True

    @staticmethod
    def _sizes_from_index(
        idx: dict[str, tuple[str, int, int | None]],
    ) -> dict[str, int | None]:
        """Shard size the index was built from: the highest record end.

        None means "unknown" (a legacy offsets.json without record
        lengths) — such a shard is re-scanned on the first miss.
        """
        sizes: dict[str, int | None] = {path: 0 for path, _off, _len in idx.values()}
        for path, off, length in idx.values():
            if length is None:
                sizes[path] = None
            elif sizes.get(path) is not None:
                sizes[path] = max(sizes[path], off + length)
        return sizes

    def _ensure_index(self) -> None:
        with self._lock:
            if self._index is not None:
                return
            if self._load_persisted_index():
                return
            self._index = {}
            for path in self._shard_paths():
                self._scan_shard(path)

    def _scan_shard(self, path: str) -> None:
        """(Re-)index one shard, replacing its previous entries."""
        assert self._index is not None
        for doc_id, loc in list(self._index.items()):
            if loc[0] == path:
                del self._index[doc_id]
        size = 0
        if not os.path.exists(path):
            self._shard_sizes[path] = 0
            return
        with open(path, "rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                size = off + len(line)
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(d, dict) and "id" in d:
                    self._index[d["id"]] = (path, off, len(line))
        self._shard_sizes[path] = size

    def _shard_changed(self, path: str) -> bool:
        """True when the shard no longer matches the indexed state.

        A shard that grew (doc appended after save_index) or was rebuilt
        (same name, different content) invalidates the recorded offsets.
        """
        recorded = self._shard_sizes.get(path)
        try:
            current = os.path.getsize(path)
        except OSError:
            current = 0
        if recorded is None:
            return True  # unknown baseline (legacy index) -> re-scan
        return current != recorded

    def _read_record(
        self, path: str, off: int, length: int | None, doc_id: str
    ) -> dict | None:
        """Read the record at off and VERIFY it is the requested doc.

        The id check is what turns a stale offset into a miss instead of
        a different document served under the requested id.
        """
        try:
            with open(path, "rb") as f:
                f.seek(off)
                line = f.readline()
        except OSError:
            return None
        if not line:
            return None
        if length is not None and len(line) != length:
            return None
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(d, dict) or d.get("id") != doc_id:
            return None
        return d

    def _cache(self, doc_id: str, d: dict) -> dict:
        self.stats["hits"] += 1
        self._lru[doc_id] = d
        self._lru.move_to_end(doc_id)
        if len(self._lru) > self._lru_size:
            self._lru.popitem(last=False)
        return d

    def get(self, doc_id: str) -> dict | None:
        with self._lock:
            if doc_id in self._lru:
                self.stats["hits"] += 1
                self._lru.move_to_end(doc_id)
                return self._lru[doc_id]
            self._ensure_index()
            assert self._index is not None
            loc = self._index.get(doc_id)
            if loc is None:
                shard = self._shard(doc_id)
                if self._shard_changed(shard):
                    self._scan_shard(shard)
                    loc = self._index.get(doc_id)
                if loc is None:
                    self.stats["misses"] += 1
                    return None
            path, off, length = loc
            d = self._read_record(path, off, length, doc_id)
            if d is None:
                self._scan_shard(path)
                loc = self._index.get(doc_id)
                if loc is None:
                    self.stats["misses"] += 1
                    return None
                path, off, length = loc
                d = self._read_record(path, off, length, doc_id)
                if d is None:
                    self.stats["misses"] += 1
                    return None
            return self._cache(doc_id, d)
