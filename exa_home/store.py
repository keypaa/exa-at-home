# exa_home/store.py
from __future__ import annotations
import hashlib, json, os
from collections import OrderedDict

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
        self._index: dict[str, tuple[str, int]] | None = None

    def _shard(self, doc_id: str) -> str:
        return os.path.join(self.root, f"shard-{int(hashlib.sha1(doc_id.encode()).hexdigest()[:8], 16) % 64}.jsonl")

    def put(self, doc: dict) -> None:
        with open(self._shard(doc["id"]), "a", encoding="utf-8") as f:
            f.write(json.dumps(doc) + "\n")

    def _ensure_index(self) -> None:
        """Scan all shards once, recording byte offsets per doc id."""
        if self._index is not None:
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
