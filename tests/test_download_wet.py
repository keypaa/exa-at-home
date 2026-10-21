# tests/test_download_wet.py
"""Regression tests for scripts/download_wet.py resume logic (offline).

The real downloader is cloud-only (HTTPS to CloudFront), so the transport is
stubbed here. The contract under test:

  A RESUMED file must end up BYTE-IDENTICAL to a fresh download.

The old `fetch_one` captured the on-disk size ONCE before the retry loop.
A failed attempt leaves its partial bytes on disk, so the next attempt
computed the `Range` header from the STALE size and opened the file in
"ab" (append) mode -> the partial prefix was DUPLICATED (corrupt gzip
stream). The function still returned normally, and the next run saw
`have >= total` -> "already complete: skip", making the corruption
permanent and silently ingested.
"""

from pathlib import Path

import pytest


class _Boom(Exception):
    """Transport failure (mid-transfer reset)."""


class _Resp:
    """Minimal urlopen response: status + a payload that can fail mid-read."""

    def __init__(self, status: int, payload: bytes, fail_after: int | None = None):
        self.status = status
        self._p = payload
        self._fail_after = fail_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        if self._fail_after is not None:
            if self._fail_after <= 0:
                raise _Boom("connection reset")
            out = self._p[: self._fail_after]
            self._p = self._p[self._fail_after :]
            self._fail_after = 0
            return out
        out, self._p = self._p[:n], self._p[n:]
        return out


class _NoSleep:
    """Stand-in for the `time` module: retries must not sleep in tests."""

    def sleep(self, _seconds):
        pass


def _transport(payload: bytes, ranges: list, fail_first_after: int | None = None):
    """Fake urlopen: honours Range (206) or ignores it (200)."""

    def fake_urlopen(req, timeout=None):
        rng = req.get_header("Range")
        ranges.append(rng)
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            status, body = 206, payload[start:]
        else:
            status, body = 200, payload
        fail_after = fail_first_after if len(ranges) == 1 else None
        return _Resp(status, body, fail_after=fail_after)

    return fake_urlopen


@pytest.fixture
def dw():
    import scripts.download_wet as module

    return module


def _install(monkeypatch, dw, payload, ranges, fail_first_after=None, total=None):
    import urllib.request

    monkeypatch.setattr(
        dw, "_remote_size", lambda url: len(payload) if total is None else total
    )
    monkeypatch.setattr(
        urllib.request, "urlopen", _transport(payload, ranges, fail_first_after)
    )
    monkeypatch.setattr(dw, "time", _NoSleep())


def test_retry_after_partial_failure_does_not_duplicate_prefix(
    tmp_path, monkeypatch, dw
):
    """The failed attempt's partial prefix must not be duplicated.

    A partially downloaded file (200 bytes) is on disk; the server honours
    Range (206). Attempt 1 resumes at 200 and dies 100 bytes later, so the
    file holds 300 bytes. Attempt 2 must ask for `bytes=300-` — the size
    that is REALLY on disk. The old code re-used the size captured before
    the loop (`bytes=200-`), so the server resent bytes 200.. and the
    "ab" append DUPLICATED them: 1100 bytes of corrupt gzip stream.
    """
    payload = bytes(range(256)) * 4  # 1024 bytes
    f = tmp_path / "x.wet.gz"
    f.write_bytes(payload[:200])  # partial file from an earlier run

    ranges: list = []
    _install(monkeypatch, dw, payload, ranges, fail_first_after=100)

    path, size = dw.fetch_one("https://example.com/x.wet.gz", str(tmp_path))

    assert size == len(payload)
    assert Path(path).read_bytes() == payload
    assert ranges == ["bytes=200-", "bytes=300-"]


def test_resume_after_a_failed_first_attempt_uses_the_disk_size(
    tmp_path, monkeypatch, dw
):
    """Nothing on disk yet: attempt 1 dies mid-transfer, attempt 2 resumes."""
    payload = bytes(range(256)) * 4  # 1024 bytes
    ranges: list = []
    _install(monkeypatch, dw, payload, ranges, fail_first_after=200)

    path, size = dw.fetch_one("https://example.com/x.wet.gz", str(tmp_path))

    assert size == len(payload)
    assert Path(path).read_bytes() == payload
    assert ranges == [None, "bytes=200-"]


def test_resumed_download_is_byte_identical_to_a_fresh_download(
    tmp_path, monkeypatch, dw
):
    """Same bytes, whether the file was downloaded in one go or resumed."""
    payload = bytes(range(256)) * 4  # 1024 bytes

    resumed_ranges: list = []
    _install(monkeypatch, dw, payload, resumed_ranges, fail_first_after=100)
    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    (resumed_dir / "x.wet.gz").write_bytes(payload[:200])  # partial file
    resumed_path, _size = dw.fetch_one("https://example.com/x.wet.gz", str(resumed_dir))

    fresh_ranges: list = []
    _install(monkeypatch, dw, payload, fresh_ranges)
    fresh_dir = tmp_path / "fresh"
    fresh_dir.mkdir()
    fresh_path, _size = dw.fetch_one("https://example.com/x.wet.gz", str(fresh_dir))

    assert Path(resumed_path).read_bytes() == Path(fresh_path).read_bytes() == payload
    assert fresh_ranges == [None]


def test_oversized_local_file_is_restarted_not_skipped(tmp_path, monkeypatch, dw):
    """A local file LONGER than the remote object is corrupt, not complete.

    That is exactly the state the old bug left behind (duplicated prefix):
    the old code saw `have >= total` and skipped the file forever.
    """
    payload = b"payload-bytes" * 100  # 1200 bytes
    stale = b"stale-prefix" + payload  # 1212 bytes > remote size
    f = tmp_path / "x.wet.gz"
    f.write_bytes(stale)

    ranges: list = []
    _install(monkeypatch, dw, payload, ranges, total=len(payload))

    path, size = dw.fetch_one("https://example.com/x.wet.gz", str(tmp_path))
    assert size == len(payload)
    assert Path(path).read_bytes() == payload
    assert ranges == [None]  # restarted from scratch, no bogus Range


def test_complete_file_is_skipped_without_a_request(tmp_path, monkeypatch, dw):
    """Already-complete files are still skipped (no network round-trip)."""
    payload = b"payload-bytes" * 10
    f = tmp_path / "x.wet.gz"
    f.write_bytes(payload)

    ranges: list = []
    _install(monkeypatch, dw, payload, ranges)

    path, size = dw.fetch_one("https://example.com/x.wet.gz", str(tmp_path))
    assert (Path(path), size) == (f, len(payload))
    assert f.read_bytes() == payload
    assert ranges == []  # nothing was fetched
