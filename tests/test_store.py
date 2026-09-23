# tests/test_store.py
import json
import os

from exa_home.store import OFFSETS_FILENAME, ContentStore

def test_put_get_roundtrip(tmp_path):
    s = ContentStore(str(tmp_path))
    s.put({"id": "a", "url": "https://e.co", "title": "T", "text": "hello world"})
    got = s.get("a")
    assert got["text"] == "hello world"
    assert s.get("missing") is None
    assert s.stats["hits"] == 1 and s.stats["misses"] == 1


def test_bulk_offsets_roundtrip_no_rescan(tmp_path):
    """begin_bulk + save_index persists offsets; a fresh store loads
    them without scanning (offset-index-at-build-time)."""
    root = str(tmp_path)
    s = ContentStore(root)
    s.begin_bulk()
    docs = [{"id": f"d{i}", "url": f"https://e.co/{i}",
             "title": "T", "text": f"hello {i}"} for i in range(5)]
    for d in docs:
        s.put(d)
    ipath = s.save_index()
    assert os.path.basename(ipath) == OFFSETS_FILENAME
    serial = json.loads(open(ipath, encoding="utf-8").read())
    assert set(serial) == {f"d{i}" for i in range(5)}
    # Shard names are portable basenames, offsets are ints.
    for shard, off in serial.values():
        assert "/" not in shard and isinstance(off, int)

    fresh = ContentStore(root)
    assert fresh._index is None
    assert fresh.get("d3")["text"] == "hello 3"
    assert fresh._index is not None and len(fresh._index) == 5
    assert fresh.get("missing") is None


def test_store_without_offsets_still_scans(tmp_path):
    """Backward compat: pre-V2 indexes without offsets.json serve via
    the one-time lazy scan."""
    root = str(tmp_path)
    s = ContentStore(root)
    s.put({"id": "x", "url": "https://e.co/x", "title": "", "text": "old"})
    assert not os.path.exists(os.path.join(root, OFFSETS_FILENAME))
    fresh = ContentStore(root)
    assert fresh.get("x")["text"] == "old"
