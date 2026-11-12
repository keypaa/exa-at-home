import json
import os
import threading
import time

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
    # length + content digest ride along so a stale/substituted record is
    # detectable.
    for entry in serial.values():
        shard, off = entry[0], entry[1]
        assert "/" not in shard and isinstance(off, int)
        assert len(entry) == 4 and isinstance(entry[2], int)
        assert isinstance(entry[3], str) and len(entry[3]) == 64

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


def test_store_serves_original_content_after_in_place_mutation(tmp_path):
    """Same id, same length, different bytes: the build-time digest wins.

    Without the digest the stale offset happily returns substituted
    content under the requested id (finding P1-04).
    """
    root = str(tmp_path)
    s = ContentStore(root)
    s.begin_bulk()
    s.put(_doc("a", "original content"))
    s.save_index()

    shard = s._shard("a")
    with open(shard, "rb") as f:
        raw = f.read()
    original = json.dumps(_doc("a", "original content"))
    replaced = json.dumps(_doc("a", "tampered content"))
    assert len(original) == len(replaced)
    with open(shard, "wb") as f:
        f.write(raw.replace(original.encode(), replaced.encode()))

    fresh = ContentStore(root)
    assert fresh.get("a") is None, "substituted content was served"
    # A full rescan must not re-trust the tampered bytes either.
    assert fresh.get("a") is None


def test_store_concurrent_gets_do_not_serialize_disk_io(tmp_path, monkeypatch):
    """8 concurrent cold gets must overlap their reads, not queue behind a
    global lock (finding P1-01: 0.28x at 8 threads)."""
    root = str(tmp_path)
    s = ContentStore(root)
    s.begin_bulk()
    ids = [f"d{i}" for i in range(8)]
    for i, doc_id in enumerate(ids):
        s.put(_doc(doc_id, f"text {i}"))
    s.save_index()

    fresh = ContentStore(root)
    real_read = fresh._read_record

    def slow_read(*args, **kwargs):
        time.sleep(0.05)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(fresh, "_read_record", slow_read)
    errors: list[BaseException] = []

    def worker(doc_id):
        try:
            assert fresh.get(doc_id)["id"] == doc_id
        except BaseException as e:  # noqa: BLE001 — surface in the main thread
            errors.append(e)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    assert not errors
    # Serial would be 8 * 50ms = 400ms; overlapping reads stay far below.
    assert elapsed < 0.25, f"reads serialized: {elapsed:.3f}s"


def test_store_rescan_is_bounded_to_appended_tail(tmp_path):
    """A newly appended doc rescans only the appended bytes, not the whole
    shard (finding P1-02: 13-20ms per new-doc get)."""
    root = str(tmp_path)
    s = ContentStore(root)
    s.begin_bulk()
    target_shard = s._shard("appended")
    # Fill ONE shard with many records so a full rescan would be expensive.
    same_shard = [
        f"doc-{i}" for i in range(20_000) if s._shard(f"doc-{i}") == target_shard
    ][:200]
    assert len(same_shard) == 200
    for i, doc_id in enumerate(same_shard):
        s.put(_doc(doc_id, f"body {i} " + "x" * 100))
    s.save_index()

    fresh = ContentStore(root)
    fresh._ensure_index()
    shard_size = os.path.getsize(target_shard)
    # Append one new record directly to the shard (a later ingest batch).
    appended = (json.dumps(_doc("appended", "new body")) + "\n").encode()
    with open(target_shard, "ab") as f:
        f.write(appended)

    fresh.stats["scan_bytes"] = 0
    assert fresh.get("appended")["text"] == "new body"
    assert fresh.stats["scan_bytes"] <= len(appended) + 1
    assert fresh.stats["scan_bytes"] < shard_size / 10


def test_store_lru_byte_budget_evicts(tmp_path):
    """The LRU is bounded by bytes, not only by entry count (P2-12)."""
    s = ContentStore(str(tmp_path), lru_size=10_000, lru_bytes=1_000)
    for i in range(6):
        s.put(_doc(f"d{i}", "y" * 400))
    for i in range(6):
        assert s.get(f"d{i}") is not None
    assert s._lru_used <= 1_000
    assert len(s._lru) <= 2


def test_store_get_many_matches_individual_gets(tmp_path):
    s = ContentStore(str(tmp_path))
    for i in range(5):
        s.put(_doc(f"d{i}", f"body {i}"))
    got = s.get_many(["d0", "d3", "missing"])
    assert got["d0"]["text"] == "body 0"
    assert got["d3"]["text"] == "body 3"
    assert got["missing"] is None
    assert s.get("d0")["text"] == "body 0"


def test_store_bulk_keeps_handles_open_until_save(tmp_path):
    """Bulk puts reuse one append handle per shard (P2-07)."""
    s = ContentStore(str(tmp_path))
    s.begin_bulk()
    s.put(_doc("a"))
    s.put(_doc("b"))
    assert s._bulk_handles, "bulk mode did not open a cached handle"
    handles = dict(s._bulk_handles)
    s.put(_doc("c"))
    # Same handle object reused for later puts.
    for path, handle in handles.items():
        assert s._bulk_handles[path] is handle
    s.save_index()
    assert s._bulk_handles is None
    fresh = ContentStore(str(tmp_path))
    assert fresh.get("c")["id"] == "c"


def test_store_reset_prunes_only_its_own_files(tmp_path):
    """reset() drops THIS store's shards + offsets so a rebuild cannot grow
    the dir by a full corpus per rebuild — and touches nothing else."""
    root = str(tmp_path)
    s = ContentStore(root)
    s.begin_bulk()
    s.put(_doc("a", "first corpus body"))
    s.save_index()
    assert s._shard_paths()

    # Files this store did not write must survive a reset.
    (tmp_path / "notes.txt").write_text("keep me")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "shard-0.jsonl").write_text("not mine")

    fresh = ContentStore(root)
    removed = fresh.reset()
    assert removed == 2, f"expected shard + offsets removed, got {removed}"
    assert (tmp_path / "notes.txt").is_file()
    assert (tmp_path / "nested" / "shard-0.jsonl").is_file()
    assert not fresh._shard_paths()

    # A fresh bulk load starts from a clean slate: old ids are gone.
    fresh.begin_bulk()
    fresh.put(_doc("b", "second corpus body"))
    fresh.save_index()
    assert fresh.get("b")["text"] == "second corpus body"
    assert fresh.get("a") is None
    # reset() on an empty/absent store is a no-op, not an error.
    assert ContentStore(str(tmp_path / "missing")).reset() == 0
