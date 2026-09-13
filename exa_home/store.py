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

    def _shard(self, doc_id: str) -> str:
        return os.path.join(self.root, f"shard-{int(hashlib.sha1(doc_id.encode()).hexdigest()[:8], 16) % 64}.jsonl")

    def put(self, doc: dict) -> None:
        with open(self._shard(doc["id"]), "a", encoding="utf-8") as f:
            f.write(json.dumps(doc) + "\n")

    def get(self, doc_id: str) -> dict | None:
        if doc_id in self._lru:
            self.stats["hits"] += 1
            self._lru.move_to_end(doc_id)
            return self._lru[doc_id]
        path = self._shard(doc_id)
        if not os.path.exists(path):
            self.stats["misses"] += 1
            return None
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] == doc_id:
                    self.stats["hits"] += 1
                    self._lru[doc_id] = d
                    if len(self._lru) > self._lru_size:
                        self._lru.popitem(last=False)
                    return d
        self.stats["misses"] += 1
        return None
