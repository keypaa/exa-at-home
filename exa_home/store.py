# exa_home/store.py
from __future__ import annotations
import hashlib, json, os
from collections import OrderedDict

OFFSETS_FILENAME = "offsets.json"

class ContentStore:
    """Disk shards + small RAM LRU. Serves the home_contents path."""
    def __init__(self, root: str, lru_size: int = 10_000):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._lru: OrderedDict[str, dict] = OrderedDict()
        self._lru_size = lru_size
        self.stats = {"hits": 0, "misses": 0}
        # id -> (shard_path, byte_offset) index. Built lazily on first
        # miss and invalidated never (shards are append-only per build;
        # a fresh ContentStore per index build starts empty anyway).
        # Without this, every cold get() scans full JSONL shards line by
        # line: O(shard) per lookup, ~28ms each at 81k docs — the 1400ms
        # "rerank" wall found on-box 2026-09-16 was really this.
        # V2: build_index.py writes store/offsets.json at build time
        # (offsets tracked during put, no rescan); _ensure_index loads
        # the file when present, falling back to the one-time scan for
        # old indexes without it.
        self._index: dict[str, tuple[str, int]] | None = None

    def _shard(self, doc_id: str) -> str:
        return os.path.join(self.root, f"shard-{int(hashlib.sha1(doc_id.encode()).hexdigest()[:8], 16) % 64}.jsonl")

    def begin_bulk(self) -> None:
        """Enable offset tracking for a bulk load (build_index path).

        Sets an empty index so each put() records its byte offset.
        Call save_index() after the loop — no rescan needed.
        """
        if self._index is None:
            self._index = {}

    def put(self, doc: dict) -> None:
        path = self._shard(doc["id"])
        if self._index is not None:
            try:
                off = os.path.getsize(path)
            except OSError:
                off = 0
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc) + "\n")
            # Last write wins on duplicate ids (append-only shards).
            self._index[doc["id"]] = (path, off)
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc) + "\n")

    def save_index(self) -> str:
        """Persist the in-memory offset index to store/offsets.json.

        Stored as {id: [shard_basename, offset]} (basenames, not absolute
        paths, so the index dir stays relocatable). Returns the file path.
        """
        self._ensure_index()
        assert self._index is not None
        out_path = os.path.join(self.root, OFFSETS_FILENAME)
        serial = {doc_id: [os.path.basename(p), off]
                  for doc_id, (p, off) in self._index.items()}
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
        idx: dict[str, tuple[str, int]] = {}
        for doc_id, loc in serial.items():
            try:
                shard, off = loc
                idx[doc_id] = (os.path.join(self.root, shard), int(off))
            except (ValueError, TypeError):
                continue
        self._index = idx
        return True

    def _ensure_index(self) -> None:
        """Load persisted offsets.json when present, else scan once."""
        if self._index is not None:
            return
        if self._load_persisted_index():
            return
        idx: dict[str, tuple[str, int]] = {}
        for n in range(64):
            path = os.path.join(self.root, f"shard-{n}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                while True:
                    off = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    try:
                        d = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if isinstance(d, dict) and "id" in d:
                        idx[d["id"]] = (path, off)
        self._index = idx

    def _cache(self, doc_id: str, d: dict) -> dict:
        self.stats["hits"] += 1
        self._lru[doc_id] = d
        self._lru.move_to_end(doc_id)
        if len(self._lru) > self._lru_size:
            self._lru.popitem(last=False)
        return d

    def get(self, doc_id: str) -> dict | None:
        if doc_id in self._lru:
            self.stats["hits"] += 1
            self._lru.move_to_end(doc_id)
            return self._lru[doc_id]
        self._ensure_index()
        assert self._index is not None
        loc = self._index.get(doc_id)
        if loc is None:
            self.stats["misses"] += 1
            return None
        path, off = loc
        with open(path, "rb") as f:
            f.seek(off)
            d = json.loads(f.readline())
        return self._cache(doc_id, d)
