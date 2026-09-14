# scripts/select_slice.py
"""DuckDB over the CC columnar index -> top-N English-dense WET URLs.

Cloud-only (network + data scale): runs DuckDB-over-HTTPS against the
Common Crawl columnar index, so it cannot run in the offline sandbox and has
no local tests — the on-box smoke is the `--limit-wet 4` toy invocation in
docs/RUNBOOK.md. Defaults locked to CC-MAIN-2026-34.

Slice sizing (researched, locked): 1M usable English docs ~= top ~130 WET
files (~8GB); 5M ~= top ~620 files (~38GB); toy = 4 WET (~10k docs).

Dedup + exclusions (locked) are enforced downstream, not in this SQL:
digest-first (`content_digest`, SHA-1 `WARC-Payload-Digest`) then URL-second
dedup happens in `exa_home.ingest.write_docs_jsonl` + the RUNBOOK URL pass;
`robotstxt` / `crawldiagnostics` subsets and truncated records are dropped at
ingest (`read_wet` only yields `conversion` records with decodable text).
"""
import argparse

CRAWL = "CC-MAIN-2026-34"
BASE = "https://data.commoncrawl.org"

# Locked SQL (verbatim): rank WARC segments by English-doc density so the
# slice needs only the densest files. eng-variant LIKEs are the language gate
# (V1 is eng-only; --lang accepts nothing else).
SQL = """
SELECT warc_filename, COUNT(*) AS n_eng FROM read_parquet(
  'https://data.commoncrawl.org/cc-index/table/cc-main/warc/crawl={crawl}/subset=warc/*.parquet',
  hive_partitioning=1)
WHERE subset = 'warc'
  AND fetch_status = 200
  AND content_mime_detected IN ('text/html', 'application/xhtml+xml')
  AND (content_languages = 'eng'
    OR content_languages LIKE 'eng,%'
    OR content_languages LIKE '%,eng'
    OR content_languages LIKE '%,eng,%')
GROUP BY warc_filename
ORDER BY n_eng DESC
LIMIT {lim}
"""


def render_sql(crawl: str = CRAWL, limit_wet: int = 4) -> str:
    return SQL.format(crawl=crawl, lim=limit_wet)


def warc_to_wet(warc_filename: str) -> str:
    """Map a WARC segment path to its sibling WET path.

    The full WET file list is also published at
    crawl-data/<crawl>/wet.paths.gz; the /warc/ -> /wet/ sibling mapping is
    the documented shortcut (see RUNBOOK).
    """
    if "/warc/" not in warc_filename:
        raise ValueError(f"not a warc segment path: {warc_filename!r}")
    return warc_filename.replace("/warc/", "/wet/", 1)


def to_urls(rows: list[tuple]) -> list[str]:
    return [f"{BASE}/{warc_to_wet(warc)}" for warc, _n in rows]


def main(crawl: str, limit_wet: int, out: str) -> list[str]:
    import duckdb  # cloud-only import: needs network + DuckDB HTTPS support

    # Generic-HTTP globs need an explicit opt-in on current DuckDB versions.
    con = duckdb.connect()
    con.execute("SET allow_asterisks_in_http_paths = true")
    rows = con.execute(render_sql(crawl, limit_wet)).fetchall()
    urls = to_urls(rows)
    with open(out, "w") as f:
        for u in urls:
            f.write(u + "\n")
    print(f"{len(urls)} WET URLs -> {out} (crawl {crawl})")
    return urls


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crawl", default=CRAWL)
    ap.add_argument("--lang", default="eng", choices=["eng"],
                    help="V1 is eng-only (locked); the SQL LIKEs are the gate.")
    ap.add_argument("--limit-wet", type=int, default=4)
    ap.add_argument("--out", default="wet_urls.txt")
    a = ap.parse_args()
    main(a.crawl, a.limit_wet, a.out)
