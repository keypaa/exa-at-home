# scripts/download_wet.py
"""Fetch N WET files with resume. Cloud-only (network).

HTTPS ONLY: unsigned `s3://` access to Common Crawl has returned 403 since
2022 — always fetch via https://data.commoncrawl.org/. Expect transient
CloudFront 403s: the retry loop below rides those out.

Resume: per-file HTTP `Range` from the current on-disk size; skip files that
already match the remote Content-Length. 16 workers by default.
"""

import argparse
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from exa_home.config import DOWNLOAD_WORKERS

RETRIES = 5
BACKOFF_S = 2.0


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            ln = r.headers.get("Content-Length")
            return int(ln) if ln else None
    except Exception:  # noqa: BLE001 — size probe is best-effort; None skips
        return None


def _truncate(path: str, size: int) -> None:
    """Cut `path` to exactly `size` bytes (no-op when it is missing)."""
    if not os.path.exists(path):
        if size:
            raise OSError(f"{path}: cannot truncate a missing file")
        return
    with open(path, "r+b") as f:
        f.truncate(size)


def fetch_one(url: str, out_dir: str) -> tuple[str, int]:
    """Download one URL with resume + retries. Returns (path, bytes_on_disk).

    Resume is OFFSET-SAFE: the on-disk size is re-read at the TOP of every
    attempt, because a failed attempt leaves its partial bytes on disk. The
    old code captured the size ONCE before the retry loop, so the next
    attempt sent a stale `Range` and appended behind the partial prefix —
    duplicating it (corrupt gzip stream) — and the next run then saw
    `have >= total` and skipped the corrupt file as "already complete".

    A resumed file therefore ends up byte-identical to a fresh download.
    """
    name = url.rsplit("/", 1)[-1]
    path = os.path.join(out_dir, name)
    total = _remote_size(url)
    if total is not None and total > 0:
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have > total:
            # Local file is LONGER than the remote object: it is corrupt
            # (e.g. a duplicated partial prefix left by an earlier crash).
            # Restart from scratch instead of "already complete: skip".
            _truncate(path, 0)
        elif have >= total:
            return path, have  # already complete: skip
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            # Re-read the size EVERY attempt: a failed attempt may have
            # left a partial prefix behind, and the Range header + the
            # append offset must both match what is really on disk.
            have = os.path.getsize(path) if os.path.exists(path) else 0
            req = urllib.request.Request(url)
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status == 206:
                    mode = "ab"  # resume accepted
                elif r.status == 200:
                    mode = "wb"  # server ignored Range: restart from scratch
                else:
                    raise OSError(f"unexpected HTTP {r.status} for {url}")
                if mode == "ab":
                    # Drop anything past the resume offset so the append can
                    # never duplicate a partial prefix.
                    _truncate(path, have)
                with open(path, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            return path, os.path.getsize(path)
        except Exception as e:  # noqa: BLE001 — transient 403s/resets; retried then raised as OSError
            # (narrowing would drop an unknown transient class and fail the batch)
            last = e
            time.sleep(BACKOFF_S * (2**attempt))
    raise OSError(f"{url}: failed after {RETRIES} attempts: {last}")


def main(urls_path: str, out: str, workers: int = 16) -> None:
    os.makedirs(out, exist_ok=True)
    with open(urls_path) as f:
        urls = [ln.strip() for ln in f if ln.strip()]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _path, _size in ex.map(lambda u: fetch_one(u, out), urls):
            done += 1
    total_gb = (
        sum(
            os.path.getsize(os.path.join(out, u.rsplit("/", 1)[-1]))
            for u in urls
            if os.path.exists(os.path.join(out, u.rsplit("/", 1)[-1]))
        )
        / 1e9
    )
    print(f"{done}/{len(urls)} WET files in {out} ({total_gb:.1f} GB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urls", required=True, help="wet_urls.txt from select_slice.py")
    ap.add_argument("--out", default="data/wet/")
    ap.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS)
    a = ap.parse_args()
    main(a.urls, a.out, a.workers)
