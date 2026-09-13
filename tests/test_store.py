# tests/test_store.py
from exa_home.store import ContentStore

def test_put_get_roundtrip(tmp_path):
    s = ContentStore(str(tmp_path))
    s.put({"id": "a", "url": "https://e.co", "title": "T", "text": "hello world"})
    got = s.get("a")
    assert got["text"] == "hello world"
    assert s.get("missing") is None
    assert s.stats["hits"] == 1 and s.stats["misses"] == 1
