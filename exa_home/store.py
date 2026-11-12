# exa_home/store.py
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict

from .config import SHARD_COUNT

OFFSETS_FILENAME = "offsets.json"
# Byte budget for the RAM LRU. A count-only bound let 10k large documents
# occupy hundreds of MB; the cache now evicts by total bytes as well.
DEFAULT_LRU_BYTES = 64 * 1024 * 1024

# id -> (shard_path, byte_offset, record_length|None, content_sha256|None).
# The digest is the build-time trust anchor: a record mutated in place (same
# id, same length) is detected and served as a miss instead of substituted
# content (finding P1-04). Legacy offsets.json entries have None (no digest).
Entry = tuple[str, int, int | None, str | None]


class ContentStore:
    """Disk shards + small RAM LRU. Serves the home_contents path."""

    def __init__(
        self,
        root: str,
        lru_size: int = 10_000,
        lru_bytes: int = DEFAULT_LRU_BYTES,
    ):
        self.root = root
        os.makedirs(root, exist_ok=True)
        # doc_id -> (doc, nbytes); OrderedDict is the LRU order.
        self._lru: OrderedDict[str, tuple[dict, int]] = OrderedDict()
        self._lock = threading.RLock()
        self._lru_size = lru_size
        self._lru_bytes = max(0, lru_bytes)
        self._lru_used = 0
        self.stats = {"hits": 0, "misses": 0, "scan_bytes": 0}
        # id -> (shard_path, byte_offset, record_length, digest) index. Built
        # lazily on first miss (or loaded from store/offsets.json).
        self._index: dict[str, Entry] | None = None
        # shard_path -> byte size the index was built from (None = unknown,
        # e.g. a legacy offsets.json without record lengths).
        self._shard_sizes: dict[str, int | None] = {}
        # Per-shard locks: a rescan of one shard must not block reads of
        # another shard (finding P1-01/P1-02).
        self._shard_locks: dict[str, threading.Lock] = {}
        # Bulk mode: keep one append handle per shard open for the whole
        # build instead of open/close per put (finding P2-07).
        self._bulk_handles: dict[str, object] | None = None

    def _shard(self, doc_id: str) -> str:
        return os.path.join(
            self.root,
            f"shard-{int(hashlib.sha1(doc_id.encode()).hexdigest()[:8], 16) % SHARD_COUNT}.jsonl",
        )

    def _shard_lock(self, path: str) -> threading.Lock:
        with self._lock:
            lock = self._shard_locks.get(path)
            if lock is None:
                lock = threading.Lock()
                self._shard_locks[path] = lock
            return lock

    def begin_bulk(self) -> None:
        """Enable offset tracking + cached append handles for a bulk load."""
        with self._lock:
            if self._index is None:
                self._index = {}
                self._shard_sizes = {
                    path: os.path.getsize(path) for path in self._shard_paths()
                }
            if self._bulk_handles is None:
                self._bulk_handles = {}

    def _close_bulk(self) -> None:
        """Flush + close cached bulk handles (idempotent)."""
        if self._bulk_handles is None:
            return
        for f in self._bulk_handles.values():
            try:
                f.flush()  # type: ignore[union-attr]
                f.close()  # type: ignore[union-attr]
            except OSError:
                pass
        self._bulk_handles = None

    def close(self) -> None:
        """Flush any cached bulk handles. Safe to call repeatedly."""
        with self._lock:
            self._close_bulk()

    def _shard_paths(self) -> list[str]:
        if not os.path.isdir(self.root):
            return []
        return [
            os.path.join(self.root, name)
            for name in sorted(os.listdir(self.root))
            if name.startswith("shard-") and name.endswith(".jsonl")
        ]

    def reset(self) -> int:
        """Drop this store's own records so it can be rebuilt from scratch.

        The store holds DERIVED corpus content: a rebuild of the index
        rewrites every record from the current docs, so keeping the previous
        corpus's shards only grows the directory forever (disk hygiene) —
        and the store is deliberately NOT covered by the manifest, so
        nothing else detects that drift. Returns the number of files
        removed.

        Safety: only files this store created are deleted — `shard-*.jsonl`
        and `offsets.json` sitting directly under `root`. Anything else in
        the directory (user files, nested dirs) is left untouched, and the
        in-memory state is invalidated so no read can serve a record whose
        bytes are about to disappear.
        """
        removed = 0
        with self._lock:
            self._close_bulk()
            self._index = None
            self._shard_sizes = {}
            self._shard_locks = {}
            self._lru.clear()
            self._lru_used = 0
            if not os.path.isdir(self.root):
                return 0
            root = os.path.normpath(self.root)
            for name in sorted(os.listdir(self.root)):
                path = os.path.join(self.root, name)
                mine = os.path.normpath(os.path.dirname(path)) == root and (
                    (name.startswith("shard-") and name.endswith(".jsonl"))
                    or name == OFFSETS_FILENAME
                )
                if mine and os.path.isfile(path):
                    os.remove(path)
                    removed += 1
        return removed

    def put(self, doc: dict) -> None:
        path = self._shard(doc["id"])
        payload = (json.dumps(doc) + "\n").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        with self._lock:
            if self._bulk_handles is not None:
                f = self._bulk_handles.get(path)
                if f is None:
                    # Long-lived handle owned by the store; closed in
                    # save_index()/close() (not a leak).
                    f = open(path, "ab")  # noqa: SIM115
                    self._bulk_handles[path] = f
                off = f.tell()
                f.write(payload)
            else:
                with open(path, "ab") as f:
                    off = f.tell()
                    f.write(payload)
            if self._index is None:
                return
            self._index[doc["id"]] = (path, off, len(payload), digest)
            self._shard_sizes[path] = off + len(payload)

    def save_index(self) -> str:
        with self._lock:
            self._close_bulk()
            self._ensure_index()
            assert self._index is not None
            out_path = os.path.join(self.root, OFFSETS_FILENAME)
            serial = {}
            for doc_id, (path, off, length, digest) in self._index.items():
                entry = [os.path.basename(path), off]
                if length is not None:
                    entry.append(length)
                if digest is not None:
                    entry.append(digest)
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
        idx: dict[str, Entry] = {}
        for doc_id, loc in serial.items():
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue
            try:
                shard = str(loc[0])
                off = int(loc[1])
                length = int(loc[2]) if len(loc) > 2 else None
                digest = (
                    str(loc[3])
                    if len(loc) > 3 and isinstance(loc[3], str) and len(loc[3]) == 64
                    else None
                )
            except (ValueError, TypeError):
                continue
            # Contain the shard path under self.root: an attacker-controlled
            # offsets.json must not be able to read files outside the store
            # (absolute path or ".." traversal). See finding S2.
            resolved = os.path.normpath(
                os.path.join(os.path.normpath(self.root), shard)
            )
            root_norm = os.path.normpath(self.root)
            if os.path.isabs(shard) or not (
                resolved == root_norm or resolved.startswith(root_norm + os.sep)
            ):
                continue
            idx[str(doc_id)] = (resolved, off, length, digest)
        self._index = idx
        self._shard_sizes = self._sizes_from_index(idx)
        return True

    @staticmethod
    def _sizes_from_index(idx: dict[str, Entry]) -> dict[str, int | None]:
        """Shard size the index was built from: the highest record end.

        None means "unknown" (a legacy offsets.json without record
        lengths) — such a shard is re-scanned on the first miss.
        """
        sizes: dict[str, int | None] = {
            path: 0 for path, _off, _len, _digest in idx.values()
        }
        for path, off, length, _digest in idx.values():
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
            paths = self._shard_paths()
        # Scan outside the global lock: each shard has its own lock and the
        # index is only mutated in short critical sections.
        for path in paths:
            self._scan_shard(path, preserve_known=False, force_full=True)

    def _scan_shard(
        self,
        path: str,
        preserve_known: bool = True,
        force_full: bool = False,
    ) -> None:
        """(Re-)index one shard, appending new entries.

        `preserve_known` keeps build-time entries (and their digests) when a
        record is re-read; this is what stops an in-place same-id mutation
        from being re-trusted (finding P1-04). A miss for a newly appended
        doc scans only the appended tail when the recorded size is still a
        valid record boundary (finding P1-02).
        """
        lock = self._shard_lock(path)
        with lock:
            with self._lock:
                recorded = self._shard_sizes.get(path)
                index = self._index
            assert index is not None
            try:
                current = os.path.getsize(path)
            except OSError:
                current = 0
            start = 0
            if (
                not force_full
                and preserve_known
                and recorded is not None
                and current > recorded
            ):
                # Append-only fast path: only accept the boundary when it
                # really starts a JSON record; otherwise fall back to a full
                # scan (the shard may have been rebuilt).
                try:
                    with open(path, "rb") as f:
                        f.seek(recorded)
                        if f.read(1) == b"{":
                            start = recorded
                except OSError:
                    start = 0
            new_entries: list[tuple[str, Entry]] = []
            scanned = 0
            if os.path.exists(path):
                with open(path, "rb") as f:
                    f.seek(start)
                    while True:
                        off = f.tell()
                        line = f.readline()
                        if not line:
                            break
                        scanned += len(line)
                        try:
                            d = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                        if isinstance(d, dict) and "id" in d:
                            doc_id = d["id"]
                            digest = hashlib.sha256(line).hexdigest()
                            new_entries.append((doc_id, (path, off, len(line), digest)))
            with self._lock:
                for doc_id, entry in new_entries:
                    existing = self._index.get(doc_id)
                    if (
                        preserve_known
                        and existing is not None
                        and existing[3] is not None
                    ):
                        # Never replace a build-time digest with one computed
                        # from possibly-substituted bytes.
                        continue
                    self._index[doc_id] = entry
                self._shard_sizes[path] = current
                self.stats["scan_bytes"] += scanned

    def _shard_changed(self, path: str) -> bool:
        """True when the shard no longer matches the indexed state."""
        recorded = self._shard_sizes.get(path)
        try:
            current = os.path.getsize(path)
        except OSError:
            current = 0
        if recorded is None:
            return True  # unknown baseline (legacy index) -> re-scan
        return current != recorded

    def _read_record(
        self,
        path: str,
        off: int,
        length: int | None,
        doc_id: str,
        digest: str | None = None,
    ) -> tuple[dict | None, str]:
        """Read the record at off and verify it is the requested doc.

        Returns (doc, reason) where reason is one of ok/io/short/parse/id/
        digest. The digest check is what turns an in-place mutation into a
        miss instead of substituted content (finding P1-04); callers must
        NOT rescan on a digest mismatch (that would re-trust the new bytes).
        """
        try:
            with open(path, "rb") as f:
                f.seek(off)
                line = f.readline()
        except OSError:
            return None, "io"
        if not line:
            return None, "short"
        if length is not None and len(line) != length:
            return None, "short"
        if digest is not None:
            got = hashlib.sha256(line).hexdigest()
            if got != digest:
                return None, "digest"
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, "parse"
        if not isinstance(d, dict) or d.get("id") != doc_id:
            return None, "id"
        return d, "ok"

    def _cache(self, doc_id: str, d: dict, nbytes: int | None = None) -> dict:
        self.stats["hits"] += 1
        size = nbytes if nbytes is not None else len(json.dumps(d).encode("utf-8"))
        with self._lock:
            old = self._lru.pop(doc_id, None)
            if old is not None:
                self._lru_used -= old[1]
            # A single record larger than the whole budget is served but not
            # cached: keeping it would exceed the bound on every later hit.
            if size > self._lru_bytes:
                return d
            self._lru[doc_id] = (d, size)
            self._lru_used += size
            while self._lru and (
                len(self._lru) > self._lru_size
                or (self._lru_used > self._lru_bytes and len(self._lru) > 1)
            ):
                _evicted_id, (_evicted_doc, evicted_size) = self._lru.popitem(
                    last=False
                )
                self._lru_used -= evicted_size
            return d

    def _get_locked_entry(self, doc_id: str) -> Entry | None:
        with self._lock:
            assert self._index is not None
            return self._index.get(doc_id)

    def _lookup_and_read(self, doc_id: str) -> dict | None:
        """Resolve + read without holding the global lock during disk I/O."""
        # 1) LRU hit (fast path).
        with self._lock:
            hit = self._lru.get(doc_id)
            if hit is not None:
                self.stats["hits"] += 1
                self._lru.move_to_end(doc_id)
                return hit[0]
        # 2) Index lookup. A miss may be a newly appended doc: scan that
        # shard's tail once (per-shard lock prevents duplicate scans).
        self._ensure_index()
        loc = self._get_locked_entry(doc_id)
        if loc is None:
            self._scan_shard(self._shard(doc_id), preserve_known=True)
            loc = self._get_locked_entry(doc_id)
            if loc is None:
                with self._lock:
                    self.stats["misses"] += 1
                return None
        path, off, length, digest = loc
        d, reason = self._read_record(path, off, length, doc_id, digest)
        if d is None and reason != "digest":
            # Stale offset (shard rebuilt / truncated): rescan fully once,
            # then retry. A digest mismatch is NOT rescanned: that is the
            # fail-closed path for same-id substitution.
            self._scan_shard(path, preserve_known=True, force_full=True)
            loc = self._get_locked_entry(doc_id)
            if loc is None:
                with self._lock:
                    self.stats["misses"] += 1
                return None
            path, off, length, digest = loc
            d, _reason = self._read_record(path, off, length, doc_id, digest)
            if d is None:
                with self._lock:
                    self.stats["misses"] += 1
                return None
        if d is None:
            with self._lock:
                self.stats["misses"] += 1
            return None
        return self._cache(doc_id, d, length)

    def get(self, doc_id: str) -> dict | None:
        return self._lookup_and_read(doc_id)

    def get_many(self, ids: list[str]) -> dict[str, dict | None]:
        """Batch get: load the index once, then read each id.

        Returns a dict id -> doc (or None). Kept as one API so the MCP
        surface can enrich a whole result page without a per-id index load.
        """
        if not ids:
            return {}
        self._ensure_index()
        return {doc_id: self._lookup_and_read(doc_id) for doc_id in ids}
