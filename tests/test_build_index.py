# tests/test_build_index.py (Task 13)
"""Tier-0 for scripts/build_index.py + offline coverage for the net scripts.

select_slice.py / download_wet.py are network-bound (DuckDB-over-HTTPS,
CloudFront fetches) and cannot run in the offline sandbox — see docs/RUNBOOK.md
"why no local tests". What IS asserted offline here: the locked SQL string
content, all three argparse CLIs (--help), download resume/skip logic via
mocked transport, and the full build_index.py assemble path.

The build_index round-trip asserts the IvfIndex.load contract WITHOUT
importing ann_core (same pattern as Task 12): write via build_index.main,
verify via exa_home.index_format + independent numpy/struct/json readback.
"""
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fixtures.synth import make_docs, make_vectors

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _write_inputs(tmp_path, n=200, k=8, seed=13):
    vecs = make_vectors(n, 256, seed=seed)
    docs = make_docs(n, seed=seed)
    ids = [d["id"] for d in docs]
    rng = np.random.default_rng(seed)
    centroids = vecs[rng.choice(n, size=k, replace=False)]
    vpath, ipath, cpath, dpath = (tmp_path / "vecs.npy",
                                  tmp_path / "ids.json",
                                  tmp_path / "centroids.npy",
                                  tmp_path / "docs.jsonl")
    np.save(vpath, vecs)
    ipath.write_text(json.dumps(ids))
    np.save(cpath, centroids)
    dpath.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    return vecs, ids, docs, centroids, vpath, ipath, cpath, dpath


def test_build_index_roundtrip_200_docs(tmp_path):
    """200 synth docs: assemble -> manifest verifies -> bytes parse back
    exactly per the locked layout (the IvfIndex.load read contract)."""
    from scripts.build_index import main as build
    from exa_home.index_format import (
        CENTROIDS_FILE, CODES_FILE, DOC_IDS_FILE, LISTS_FILE,
        verify_manifest,
    )

    n, k, dim = 200, 8, 256
    vecs, ids, docs, centroids, vpath, ipath, cpath, dpath = \
        _write_inputs(tmp_path, n, k)
    out = tmp_path / "index"

    stats = build(str(vpath), str(ipath), str(cpath), str(dpath), str(out))
    assert stats == {"n_docs": n, "n_centroids": k, "n_stored": n}

    m = verify_manifest(str(out))
    assert m["n_docs"] == n and m["n_centroids"] == k and m["dim"] == dim

    # Independent readback, one assertion per locked file.
    got_c = np.fromfile(out / CENTROIDS_FILE, dtype="<f4").reshape(k, dim)
    assert np.array_equal(got_c, centroids.astype(np.float32))
    got_codes = np.fromfile(out / CODES_FILE, dtype=np.uint8).reshape(n, 32)
    want_codes = np.packbits((vecs > 0).astype(np.uint8), axis=1,
                             bitorder="little")
    assert np.array_equal(got_codes, want_codes)
    raw = (out / LISTS_FILE).read_bytes()
    (k2,) = struct.unpack_from("<I", raw, 0)
    assert k2 == k
    off, total = 4, 0
    for _ in range(k):
        (ln,) = struct.unpack_from("<I", raw, off)
        off += 4 + 4 * ln
        total += ln
    assert total == n and off == len(raw)
    # Lists partition rows by nearest centroid (argmax assignment, chunked).
    assign = np.argmax(vecs @ centroids.T, axis=1)
    off = 4
    for c in range(k):
        (ln,) = struct.unpack_from("<I", raw, off)
        off += 4
        rows = struct.unpack_from(f"<{ln}I", raw, off) if ln else ()
        off += 4 * ln
        assert sorted(rows) == sorted(np.where(assign == c)[0].tolist())
    assert json.loads((out / DOC_IDS_FILE).read_text()) == ids

    # The default <out>/store serves back doc text (home_contents path).
    from exa_home.store import OFFSETS_FILENAME, ContentStore
    # Offset index written at build time and covered by the manifest.
    assert (out / "store" / OFFSETS_FILENAME).exists()
    assert any(k.endswith(OFFSETS_FILENAME) for k in m["files"])
    cs = ContentStore(str(out / "store"))
    assert cs.get(ids[0])["text"] == docs[0]["text"]
    assert cs.get(ids[-1])["text"] == docs[-1]["text"]
    assert cs._index is not None and len(cs._index) == n


def test_build_index_rejects_shape_mismatch(tmp_path):
    from scripts.build_index import main as build
    vecs, ids, _docs, _c, vpath, ipath, cpath, dpath = _write_inputs(tmp_path)
    ipath.write_text(json.dumps(ids[:-1]))  # 199 ids vs 200 vecs
    with pytest.raises(ValueError, match="ids"):
        build(str(vpath), str(ipath), str(cpath), str(dpath),
              str(tmp_path / "index"))
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((10, 128), dtype=np.float32))
    with pytest.raises(ValueError, match="256"):
        build(str(bad), str(ipath), str(cpath), str(dpath),
              str(tmp_path / "index2"))


def test_select_slice_sql_locked_content():
    """The researched SQL, verbatim: columnar path, hive partitioning,
    all WHERE gates, GROUP-BY density rank."""
    from scripts.select_slice import CRAWL, SQL, render_sql, warc_to_wet
    assert CRAWL == "CC-MAIN-2026-34"
    q = render_sql()
    assert ("https://data.commoncrawl.org/cc-index/table/cc-main/warc/"
            "crawl=CC-MAIN-2026-34/subset=warc/*.parquet") in q
    assert "hive_partitioning=1" in q
    assert "subset = 'warc'" in q
    assert "fetch_status = 200" in q
    assert ("content_mime_detected IN "
            "('text/html', 'application/xhtml+xml')") in q
    for like in ("content_languages = 'eng'",
                 "LIKE 'eng,%'", "LIKE '%,eng'", "LIKE '%,eng,%'"):
        assert like in q, like
    assert "GROUP BY warc_filename" in q
    assert "ORDER BY n_eng DESC" in q
    assert render_sql(limit_wet=130).strip().endswith("LIMIT 130")
    # /warc/ -> /wet/ sibling mapping; HTTPS prefix added by to_urls.
    assert warc_to_wet("crawl-data/CC-MAIN-2026-34/segments/123/warc/f.warc.gz") == \
        "crawl-data/CC-MAIN-2026-34/segments/123/wet/f.warc.gz"
    from scripts.select_slice import to_urls
    assert to_urls([("crawl-data/C/segments/1/warc/f.warc.gz", 42)]) == \
        ["https://data.commoncrawl.org/crawl-data/C/segments/1/wet/f.warc.gz"]
    with pytest.raises(ValueError):
        warc_to_wet("crawl-data/CC-MAIN-2026-34/wet.paths.gz")


def test_download_skips_complete_files(tmp_path, monkeypatch):
    """Skip-existing path needs no network once the size check is mocked."""
    import scripts.download_wet as dw
    f = tmp_path / "x.wet.gz"
    f.write_bytes(b"data" * 100)
    monkeypatch.setattr(dw, "_remote_size", lambda url: 400)
    path, size = dw.fetch_one("https://example.com/x.wet.gz", str(tmp_path))
    assert (path, size) == (str(f), 400)
    assert f.read_bytes() == b"data" * 100  # untouched


def test_download_resumes_partial_file(tmp_path, monkeypatch):
    """Range-resume append logic with a stubbed transport (offline)."""
    import urllib.request

    import scripts.download_wet as dw
    full = b"0123456789abcdef" * 64  # 1024 bytes
    f = tmp_path / "y.wet.gz"
    f.write_bytes(full[:400])
    seen = {}

    class Resp:
        status = 206

        def __init__(self, payload):
            self._p = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            out, self._p = self._p[:n], self._p[n:]
            return out

    def fake_urlopen(req, timeout=None):
        seen["range"] = req.get_header("Range")
        assert seen["range"] == "bytes=400-"
        return Resp(full[400:])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    path, size = dw.fetch_one("https://example.com/y.wet.gz", str(tmp_path))
    assert size == 1024
    assert Path(path).read_bytes() == full


@pytest.mark.parametrize("script,flags", [
    ("select_slice.py", ("--crawl", "--lang", "--limit-wet", "--out")),
    ("download_wet.py", ("--urls", "--out", "--workers")),
    ("build_index.py", ("--vecs", "--ids", "--centroids", "--docs", "--out",
                        "--store", "--no-store", "--filter", "--max-terms",
                        "--crawl")),
])
def test_script_cli_help(script, flags):
    r = subprocess.run([sys.executable, str(SCRIPTS / script), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for flag in flags:
        assert flag in r.stdout, flag


@pytest.mark.cloud_only
def test_select_slice_live_toy_run(tmp_path):
    """Cloud box: real DuckDB-over-HTTPS query, 4 densest WET URLs."""
    from scripts.select_slice import main
    out = tmp_path / "wet_urls.txt"
    urls = main("CC-MAIN-2026-34", 4, str(out))
    assert len(urls) == 4
    assert all(u.startswith("https://data.commoncrawl.org/") for u in urls)
    assert all("/wet/" in u for u in urls)
