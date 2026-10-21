import json
import os

from exa_home.store import OFFSETS_FILENAME, ContentStore


def _doc(doc_id: str, text: str = "hello world") -> dict:
    return {"id": doc_id, "url": f"https://e.co/{doc_id}", "title": "T", "text": text}


def _same_shard_pair(store: ContentStore) -> tuple[str, str]:
    """Two ids that land on the same shard (so offsets interact)."""
    by_shard: dict[str, list[str]] = {}
    for i in range(500):
        doc_id = f"doc-{i}"
        by_shard.setdefault(store._shard(doc_id), []).append(doc_id)
    pair = next(v for v in by_shard.values() if len(v) >= 2)
    return pair[0], pair[1]


def test_put_get_roundtrip(tmp_path):
    s = ContentStore(str(tmp_path))
    s.put(_doc("a"))
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
    docs = [_doc(f"d{i}", f"hello {i}") for i in range(5)]
    for d in docs:
        s.put(d)
    ipath = s.save_index()
    assert os.path.basename(ipath) == OFFSETS_FILENAME
    with open(ipath, encoding="utf-8") as f:
        serial = json.loads(f.read())
    assert set(serial) == {f"d{i}" for i in range(5)}
    # Shard names are portable basenames, offsets are ints; the record
    # length rides along so a stale offset is detectable.
    for entry in serial.values():
        shard, off = entry[0], entry[1]
        assert "/" not in shard and isinstance(off, int)
        assert len(entry) == 3 and isinstance(entry[2], int)

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
    s.put(_doc("x", "old"))
    assert not os.path.exists(os.path.join(root, OFFSETS_FILENAME))
    fresh = ContentStore(root)
    assert fresh.get("x")["text"] == "old"


def test_legacy_offsets_format_still_loads(tmp_path):
    """Backward compat: the old {id: [shard, offset]} form must still be
    readable (no record length recorded)."""
    root = str(tmp_path)
    s = ContentStore(root)
    s.begin_bulk()
    s.put(_doc("a", "legacy text"))
    ipath = s.save_index()
    with open(ipath, encoding="utf-8") as f:
        serial = json.load(f)
    legacy = {k: v[:2] for k, v in serial.items()}
    with open(ipath, "w", encoding="utf-8") as f:
        json.dump(legacy, f)

    fresh = ContentStore(root)
    assert fresh.get("a")["text"] == "legacy text"


def test_doc_appended_after_save_is_still_served(tmp_path):
    """BUG: the offset index went stale silently — a doc appended to a
    shard after save_index() was on disk but invisible to get()."""
    root = str(tmp_path)
    s = ContentStore(root)
    s.begin_bulk()
    s.put(_doc("a", "first"))
    s.save_index()
    s.put(_doc("b", "appended later"))  # after the index was written

    fresh = ContentStore(root)
    got = fresh.get("b")
    assert got is not None, "doc appended after save_index() was invisible"
    assert got["text"] == "appended later"
    assert fresh.get("a")["text"] == "first"


def test_rebuilt_shard_never_serves_another_document(tmp_path):
    """BUG: a REBUILT shard left the stale offset resolving to a
    DIFFERENT document under the requested id — silently wrong content."""
    root = str(tmp_path)
    s = ContentStore(root)
    s.begin_bulk()
    s.put(_doc("a", "original"))
    s.put(_doc("b", "original two"))
    s.save_index()

    # Rebuild the shard holding "a": same file, different content.
    shard = s._shard("a")
    with open(shard, "w", encoding="utf-8") as f:
        f.write(json.dumps(_doc("rebuilt", "rebuilt content")) + "\n")

    fresh = ContentStore(root)
    got = fresh.get("a")
    assert got is None or got["id"] == "a", "stale offset served another document"
    # The rebuilt content is served under its OWN id.
    assert fresh.get("rebuilt")["text"] == "rebuilt content"


def test_put_records_the_offset_it_wrote_to(tmp_path, monkeypatch):
    """BUG: the offset was taken with os.path.getsize() BEFORE the file
    was opened, so a concurrent append in between recorded an offset
    pointing into the middle of a record."""
    root = str(tmp_path)
    s = ContentStore(root)
    s.begin_bulk()
    a, b = _same_shard_pair(s)
    s.put(_doc(a, "aaa"))

    # A stale size lookup is exactly what a concurrent append looks like.
    monkeypatch.setattr(os.path, "getsize", lambda p: 0)
    s.put(_doc(b, "bbb"))
    monkeypatch.undo()
    s.save_index()

    off_a = s._index[a][1]
    off_b = s._index[b][1]
    len_a = len(json.dumps(_doc(a, "aaa"))) + 1
    assert off_b > off_a, "offset was not taken from the file handle"
    assert off_b == off_a + len_a

    fresh = ContentStore(root)
    assert fresh.get(b)["text"] == "bbb"
    assert fresh.get(a)["text"] == "aaa"
