// crates/ann-core/src/filter.rs
// Task 8 (M6-partial): roaring-bitmap filter index over doc row ids.
//
// On-disk layout under index/filter/ (written by scripts/build_filter.py):
//   {domains,dates,terms}.bin    concatenated standard-Roaring serialized bitmaps
//   {domains,dates,terms}.json   {key: [offset, len]} slicing one bitmap out
//                                of the matching .bin
//
// Row-ids are u32 doc row indices in doc_ids.json row order (the same row
// space as lists.bin). No interaction with the M5 bit-packing: those are bit
// positions inside a doc's 32-byte code, these are row numbers.
//
// Query semantics (FilterExpr): dimensions conjoin (AND). Within `domains` or
// `terms`, multiple values disjoin (OR): {domains: [a, b]} = docs on a OR b,
// and docs must also satisfy every other dimension. `month_range` is an
// inclusive YYYY-MM span over the per-month date buckets.

use roaring::RoaringBitmap;
use std::collections::BTreeMap;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

pub const FILTER_DIR: &str = "filter";
const DOMAINS: &str = "domains";
const DATES: &str = "dates";
const TERMS: &str = "terms";

/// Portable-Roaring layout constants, mirroring the `roaring` crate's
/// reader (`src/bitmap/serialization.rs`) so the fast path below accepts
/// exactly what the crate accepts — and rejects exactly what it rejects.
const SER_COOKIE_NO_RUNCONTAINER: u32 = 12346;
const SER_COOKIE: u16 = 12347;
/// `roaring`'s ARRAY_LIMIT: a container with more elements than this is a
/// bitmap container, not an array container.
const ARRAY_LIMIT: usize = 4096;
/// A bitmap container is 1024 u64 words = 8 KiB.
const BITMAP_CONTAINER_BYTES: usize = 8 * 1024;

/// One facet: the raw .bin blob plus its {key: [offset, len]} table.
#[derive(Debug)]
struct Facet {
    blob: Vec<u8>,
    offsets: BTreeMap<String, (usize, usize)>,
    /// Test-only counter: how often the row-space check had to fall back
    /// to a full bitmap deserialization (the fast path should make this
    /// zero for every bitmap the builder writes).
    #[cfg(test)]
    fallbacks: std::sync::atomic::AtomicUsize,
}

impl Facet {
    fn get(&self, key: &str) -> Result<RoaringBitmap, String> {
        let (off, len) = self
            .offsets
            .get(key)
            .copied()
            .ok_or_else(|| format!("filter key not found: {key}"))?;
        let end = off
            .checked_add(len)
            .filter(|&e| e <= self.blob.len())
            .ok_or_else(|| format!("filter entry for {key} runs past end of blob"))?;
        RoaringBitmap::deserialize_from(Cursor::new(&self.blob[off..end]))
            .map_err(|e| format!("filter bitmap for {key} is corrupt: {e}"))
    }

    /// Max row id stored under the offset-table entry (`key`, `off`, `len`),
    /// without materializing the bitmap.
    ///
    /// The row-space check needs one number per key; deserializing every
    /// bitmap to read it cost ~2.1s of start-up on a 50k-doc index
    /// (50k terms + ~100k domains). The fast path reads the same bytes
    /// without allocating the containers, and falls back to the crate's
    /// reader whenever it does not understand the bytes — so a corrupt
    /// bitmap is still rejected (by the fallback), and the crate stays
    /// the authority on validity.
    ///
    /// The (off, len) pair is passed in because the row-space check
    /// already walks the offset table: looking the key up again would
    /// cost a BTreeMap descent per bitmap.
    fn max_row_at(&self, key: &str, off: usize, len: usize) -> Result<Option<u32>, String> {
        let end = off
            .checked_add(len)
            .filter(|&e| e <= self.blob.len())
            .ok_or_else(|| format!("filter entry for {key} runs past end of blob"))?;
        let bytes = &self.blob[off..end];
        if let Some(max) = serialized_max_row(bytes) {
            return Ok(Some(max));
        }
        #[cfg(test)]
        {
            self.fallbacks
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let bm = RoaringBitmap::deserialize_from(Cursor::new(bytes))
            .map_err(|e| format!("filter bitmap for {key} is corrupt: {e}"))?;
        Ok(bm.max())
    }
}

/// Max row id in a serialized (portable-format) Roaring bitmap, computed
/// without deserializing it.
///
/// `bytes` is one complete serialized bitmap (the slice the filter's
/// `.json` offset table points at). Returns the largest row id the bitmap
/// holds, or `None` when the bitmap is empty or the bytes are not in a
/// format this fast path understands (or are malformed) — the caller must
/// then fall back to the full deserialization, which reports the same
/// "no max" for an empty bitmap.
///
/// The container rules mirror the `roaring` crate's reader exactly:
/// the descriptive header stores `cardinality - 1`; a container is an
/// array container when its cardinality is <= ARRAY_LIMIT and a bitmap
/// container otherwise; array values must be strictly ascending; a
/// bitmap container's popcount must equal its cardinality. Anything else
/// returns `None` so the crate's reader gets the final word.
fn serialized_max_row(bytes: &[u8]) -> Option<u32> {
    let cookie = u32::from_le_bytes(bytes.get(0..4)?.try_into().ok()?);
    // (container count, descriptive-header offset, offset headers present,
    //  run-container bitmap)
    let (size, desc_off, has_offsets, run_bitmap) = if cookie == SER_COOKIE_NO_RUNCONTAINER {
        let size = u32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?) as usize;
        if size > u16::MAX as usize + 1 {
            return None;
        }
        (size, 8usize, true, None)
    } else if (cookie as u16) == SER_COOKIE {
        let size = ((cookie >> 16) + 1) as usize;
        let rb_len = size.div_ceil(8);
        let rb = bytes.get(4..4 + rb_len)?;
        (size, 4 + rb_len, size >= 4, Some(rb))
    } else {
        return None; // unknown cookie: let the crate reject it
    };
    let descs_end = desc_off.checked_add(size.checked_mul(4)?)?;
    let data_off = if has_offsets {
        descs_end.checked_add(size.checked_mul(4)?)?
    } else {
        descs_end
    };
    if bytes.len() < data_off {
        return None;
    }
    let mut max: Option<u32> = None;
    // Where the next container's data starts when there are no offset
    // headers (the crate reads the containers back to back then).
    let mut walk = data_off;
    for i in 0..size {
        let d = desc_off + 4 * i;
        let key = u16::from_le_bytes(bytes.get(d..d + 2)?.try_into().ok()?) as u32;
        let card = u16::from_le_bytes(bytes.get(d + 2..d + 4)?.try_into().ok()?) as usize + 1;
        let is_run = run_bitmap.is_some_and(|rb| rb[i / 8] & (1 << (i % 8)) != 0);
        let pos = if has_offsets {
            let o = descs_end + 4 * i;
            u32::from_le_bytes(bytes.get(o..o + 4)?.try_into().ok()?) as usize
        } else {
            walk
        };
        // (max value inside the container, container data size in bytes)
        let (cmax, csize) = if is_run {
            // Run container: u16 run count, then run count x (start u16,
            // length-1 u16).
            let runs = u16::from_le_bytes(bytes.get(pos..pos + 2)?.try_into().ok()?) as usize;
            if runs == 0 {
                return None;
            }
            let end = pos.checked_add(2 + runs.checked_mul(4)?)?;
            if end > bytes.len() {
                return None;
            }
            let mut m = 0u32;
            for r in 0..runs {
                let at = pos + 2 + 4 * r;
                let start = u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?) as u32;
                let len = u16::from_le_bytes(bytes.get(at + 2..at + 4)?.try_into().ok()?) as u32;
                let stop = start.checked_add(len)?;
                if stop > u16::MAX as u32 {
                    return None;
                }
                m = m.max(stop);
            }
            (m, 2 + runs * 4)
        } else if card <= ARRAY_LIMIT {
            // Array container: `card` strictly ascending u16 values.
            let end = pos.checked_add(card.checked_mul(2)?)?;
            if end > bytes.len() {
                return None;
            }
            let mut prev: Option<u16> = None;
            let mut m = 0u32;
            for v in 0..card {
                let at = pos + 2 * v;
                let val = u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?);
                if let Some(p) = prev {
                    if val <= p {
                        return None; // out-of-order / duplicate: the crate rejects it
                    }
                }
                prev = Some(val);
                m = m.max(val as u32);
            }
            (m, card * 2)
        } else {
            // Bitmap container: 8 KiB of u64 words. The crate requires the
            // popcount to match the declared cardinality (which need not
            // be 65536 — the crate writes a bitmap container for any
            // cardinality above ARRAY_LIMIT).
            let end = pos.checked_add(BITMAP_CONTAINER_BYTES)?;
            if end > bytes.len() {
                return None;
            }
            let mut ones = 0u64;
            let mut m = 0u32;
            for w in 0..BITMAP_CONTAINER_BYTES / 8 {
                let at = pos + 8 * w;
                let word = u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?);
                ones += word.count_ones() as u64;
                if word != 0 {
                    m = (w as u32) * 64 + (63 - word.leading_zeros());
                }
            }
            if ones != card as u64 {
                return None;
            }
            (m, BITMAP_CONTAINER_BYTES)
        };
        let value = (key << 16) | cmax;
        max = Some(max.map_or(value, |m: u32| m.max(value)));
        if !has_offsets {
            walk = pos + csize;
        }
    }
    max
}

/// Bitmaps for the three facets, loaded from index/filter/.
#[derive(Debug)]
pub struct FilterIdx {
    domains: Facet,
    dates: Facet,
    terms: Facet,
}

/// A filter expression: all present dimensions must hold (AND). Within a
/// dimension: `domains` values are OR'd (a doc has one domain),
/// `month_range` spans YYYY-MM buckets (inherently an OR over months), and
/// `terms` values are AND'd (a doc has many terms; conjunction narrows).
#[derive(Debug, Default)]
pub struct FilterExpr {
    pub domains: Option<Vec<String>>,
    pub month_range: Option<(String, String)>,
    pub terms: Option<Vec<String>>,
}

fn load_facet(dir: &Path, name: &str) -> Result<Facet, String> {
    let bin = dir.join(format!("{name}.bin"));
    let js = dir.join(format!("{name}.json"));
    let blob = fs::read(&bin).map_err(|e| format!("cannot read {}: {e}", bin.display()))?;
    let raw = fs::read(&js).map_err(|e| format!("cannot read {}: {e}", js.display()))?;
    let table: BTreeMap<String, Vec<u64>> = serde_json::from_slice(&raw)
        .map_err(|e| format!("{} is not a JSON offset table: {e}", js.display()))?;
    let mut offsets = BTreeMap::new();
    for (k, v) in table {
        if v.len() != 2 {
            return Err(format!(
                "{}: entry for {k} must be [offset, len]",
                js.display()
            ));
        }
        offsets.insert(k, (v[0] as usize, v[1] as usize));
    }
    Ok(Facet {
        blob,
        offsets,
        #[cfg(test)]
        fallbacks: std::sync::atomic::AtomicUsize::new(0),
    })
}

/// Load the filter/ subdir of an index dir. Errors (fail-fast) when any of
/// the six files is missing or malformed.
pub fn load(index_dir: &Path) -> Result<FilterIdx, String> {
    let dir = index_dir.join(FILTER_DIR);
    Ok(FilterIdx {
        domains: load_facet(&dir, DOMAINS)?,
        dates: load_facet(&dir, DATES)?,
        terms: load_facet(&dir, TERMS)?,
    })
}

/// Normalize a domain query value exactly the way scripts/build_filter.py
/// does when it indexes the domains facet.
///
/// The builder (`domain_of()`) lowercases the hostname and then strips a
/// leading "www." — so the stored key is the bare lowercase host. The
/// query side has to do the same, or `domains=["Example.COM"]` matches
/// nothing and the filter silently returns an empty result set (while
/// `terms=["UPPER"]` works, because the terms facet was already
/// lowercased on the query side).
fn normalize_domain(domain: &str) -> String {
    let lower = domain.to_lowercase();
    match lower.strip_prefix("www.") {
        Some(bare) => bare.to_string(),
        None => lower,
    }
}

/// Single-valued facets (domains, months): a doc holds one value, so
/// multiple values disjoin (OR). Unknown values match nothing and are
/// skipped (empty OR = empty).
fn or_keys(facet: &Facet, keys: &[String]) -> Result<RoaringBitmap, String> {
    let mut out = RoaringBitmap::new();
    for k in keys {
        // Anything failing past the contains check is a corrupt blob and
        // must fail fast, not skip.
        if !facet.offsets.contains_key(k) {
            continue;
        }
        out |= facet.get(k)?;
    }
    Ok(out)
}

/// Multi-valued facet (terms): a doc holds many terms, so multiple values
/// conjoin (AND) — the narrowing semantic filter-in-scan exists for. An
/// unknown term matches nothing, so the conjunction is empty (short-circuit).
/// An explicitly empty list matches nothing (Some(empty)), same as or_keys.
fn and_keys(facet: &Facet, keys: &[String]) -> Result<RoaringBitmap, String> {
    let mut out: Option<RoaringBitmap> = None;
    for k in keys {
        let bm = if !facet.offsets.contains_key(k) {
            return Ok(RoaringBitmap::new());
        } else {
            facet.get(k)?
        };
        out = Some(match out {
            None => bm,
            Some(a) => a & bm,
        });
    }
    Ok(out.unwrap_or_default())
}

impl FilterIdx {
    /// Resolve an expression to a row-id bitmap (AND across dimensions).
    /// An empty expression (no dimensions set) returns None, meaning "no
    /// filtering". Present-but-empty value lists match nothing (Some(empty)).
    pub fn resolve(&self, expr: &FilterExpr) -> Result<Option<RoaringBitmap>, String> {
        let mut acc: Option<RoaringBitmap> = None;
        // Plain helper fn (not a closure): the fold moves `acc` each step,
        // which a capturing closure cannot do.
        fn and(acc: Option<RoaringBitmap>, bm: RoaringBitmap) -> Option<RoaringBitmap> {
            Some(match acc {
                None => bm,
                Some(a) => a & bm,
            })
        }
        if let Some(ds) = &expr.domains {
            // Lookup is normalized: build_filter indexes lowercase hosts
            // with a leading "www." stripped.
            let norm: Vec<String> = ds.iter().map(|d| normalize_domain(d)).collect();
            acc = and(acc, or_keys(&self.domains, &norm)?);
        }
        if let Some((lo, hi)) = &expr.month_range {
            let months: Vec<String> = self
                .dates
                .offsets
                .keys()
                .filter(|m| *m >= lo && *m <= hi)
                .cloned()
                .collect();
            acc = and(acc, or_keys(&self.dates, &months)?);
        }
        if let Some(ts) = &expr.terms {
            // Lookup is lowercase: build_filter indexes lowercase keys.
            let lower: Vec<String> = ts.iter().map(|t| t.to_lowercase()).collect();
            acc = and(acc, and_keys(&self.terms, &lower)?);
        }
        Ok(acc)
    }

    /// Sorted row ids for a resolved bitmap (None = unfiltered; the caller
    /// passes no allow-list in that case).
    pub fn rows(resolved: &Option<RoaringBitmap>) -> Option<Vec<u32>> {
        resolved.as_ref().map(|b| b.iter().collect())
    }

    /// Row-space agreement: every stored bitmap's max row must index a real
    /// doc (< n_docs). Runs once at load so a filter built against a stale
    /// doc_ids.json fails fast instead of silently restricting results.
    ///
    /// The max row id is read straight out of the serialized bitmap (no
    /// deserialization — see `Facet::max_row`), so this stays cheap even
    /// with ~150k bitmaps.
    pub fn check_row_space(&self, n_docs: usize) -> Result<(), String> {
        let n = n_docs as u32;
        for (facet_name, facet) in [
            ("domains", &self.domains),
            ("dates", &self.dates),
            ("terms", &self.terms),
        ] {
            for (key, &(off, len)) in facet.offsets.iter() {
                if let Some(max) = facet.max_row_at(key, off, len)? {
                    if max >= n {
                        return Err(format!(
                            "filter/{facet_name} key {key} has row id {max} \
                             but the index has only {n_docs} docs"
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Manifest-relative file names under index/ covered by sha256 verify.
    pub fn manifest_files() -> [&'static str; 6] {
        [
            "filter/domains.bin",
            "filter/domains.json",
            "filter/dates.bin",
            "filter/dates.json",
            "filter/terms.bin",
            "filter/terms.json",
        ]
    }
}

/// Write a FilterIdx test fixture in the build_filter.py layout: serialize
/// with the real RoaringBitmap writer so resolve() exercises the real reader.
/// `dir` is the index dir; files land in `dir/filter/` like the real builder
/// (plus the manifest-relevant layout, minus the manifest itself).
#[cfg(test)]
pub fn write_test_fixture(dir: &Path, facets: &[(&str, Vec<(&str, Vec<u32>)>)]) {
    let fdir = subdir(dir);
    fs::create_dir_all(&fdir).unwrap();
    for (name, keys) in facets {
        let mut blob = Vec::new();
        let mut table = BTreeMap::new();
        for (key, rows) in keys {
            let bm: RoaringBitmap = rows.iter().copied().collect();
            let mut bytes = Vec::new();
            bm.serialize_into(Cursor::new(&mut bytes)).unwrap();
            table.insert(key.to_string(), vec![blob.len() as u64, bytes.len() as u64]);
            blob.extend_from_slice(&bytes);
        }
        fs::write(fdir.join(format!("{name}.bin")), &blob).unwrap();
        fs::write(
            fdir.join(format!("{name}.json")),
            serde_json::to_string(&table).unwrap(),
        )
        .unwrap();
    }
}

/// Index-dir path helper (avoids duplicating the join in index.rs/lib.rs).
pub fn subdir(index_dir: &Path) -> PathBuf {
    index_dir.join(FILTER_DIR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn fixture_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ann-core-filter-test-{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        write_test_fixture(
            &dir,
            &[
                (
                    DOMAINS,
                    vec![("example.com", vec![0, 2]), ("other.org", vec![1])],
                ),
                (DATES, vec![("2026-01", vec![0, 2]), ("2026-02", vec![1])]),
                (
                    TERMS,
                    vec![("alpha", vec![0, 1]), ("beta", vec![0]), ("zeta", vec![2])],
                ),
            ],
        );
        dir
    }

    /// Hand-write one facet of the Load-shape (index_dir/filter/ + only the
    /// facet named `only`): keeps the missing-vs-malformed test honest about
    /// the real directory level instead of index_dir root.
    fn write_partial(dir: &Path, only: &str, bin: &[u8], js: &[u8]) {
        let fdir = subdir(dir);
        fs::create_dir_all(&fdir).unwrap();
        fs::write(fdir.join(format!("{only}.bin")), bin).unwrap();
        fs::write(fdir.join(format!("{only}.json")), js).unwrap();
    }

    fn sorted(bm: &RoaringBitmap) -> Vec<u32> {
        bm.iter().collect()
    }

    #[test]
    fn resolve_conjoins_facets_and_disjoins_within() {
        let dir = fixture_dir("resolve");
        let f = load(&dir).unwrap();
        // Single domain.
        let e = FilterExpr {
            domains: Some(vec!["example.com".into()]),
            ..Default::default()
        };
        assert_eq!(sorted(&f.resolve(&e).unwrap().unwrap()), vec![0, 2]);
        // Two domains OR.
        let e = FilterExpr {
            domains: Some(vec!["example.com".into(), "other.org".into()]),
            ..Default::default()
        };
        assert_eq!(sorted(&f.resolve(&e).unwrap().unwrap()), vec![0, 1, 2]);
        // Month span: single month.
        let e = FilterExpr {
            month_range: Some(("2026-01".into(), "2026-01".into())),
            ..Default::default()
        };
        assert_eq!(sorted(&f.resolve(&e).unwrap().unwrap()), vec![0, 2]);
        // Full span covers everything.
        let e = FilterExpr {
            month_range: Some(("2026-01".into(), "2026-02".into())),
            ..Default::default()
        };
        assert_eq!(sorted(&f.resolve(&e).unwrap().unwrap()), vec![0, 1, 2]);
        // Terms AND: alpha ∩ beta = {0}.
        let e = FilterExpr {
            terms: Some(vec!["alpha".into(), "beta".into()]),
            ..Default::default()
        };
        assert_eq!(sorted(&f.resolve(&e).unwrap().unwrap()), vec![0]);
        // Case-insensitive term lookup.
        let e = FilterExpr {
            terms: Some(vec!["ALPHA".into()]),
            ..Default::default()
        };
        assert_eq!(sorted(&f.resolve(&e).unwrap().unwrap()), vec![0, 1]);
        // Cross-facet AND: domain ∩ term.
        let e = FilterExpr {
            domains: Some(vec!["example.com".into()]),
            terms: Some(vec!["alpha".into()]),
            ..Default::default()
        };
        assert_eq!(sorted(&f.resolve(&e).unwrap().unwrap()), vec![0]);
        // Empty expression = no filtering.
        assert!(f.resolve(&FilterExpr::default()).unwrap().is_none());
        // Unknown keys match nothing, not an error.
        let e = FilterExpr {
            domains: Some(vec!["nope.example".into()]),
            ..Default::default()
        };
        assert!(f.resolve(&e).unwrap().unwrap().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn domain_lookup_is_case_insensitive_and_strips_www() {
        // scripts/build_filter.py indexes lowercased hostnames with a
        // leading "www." stripped (domain_of(): host.lower() then
        // removeprefix("www.")).  The query side must normalize exactly
        // the same way, or "Example.COM" silently matches nothing.
        let dir = fixture_dir("domains");
        let f = load(&dir).unwrap();
        for q in [
            "example.com",     // already normalized
            "EXAMPLE.COM",     // upper case
            "Example.Com",     // mixed case
            "www.example.com", // leading www.
            "WWW.Example.COM", // both
        ] {
            let e = FilterExpr {
                domains: Some(vec![q.into()]),
                ..Default::default()
            };
            assert_eq!(
                sorted(&f.resolve(&e).unwrap().unwrap()),
                vec![0, 2],
                "domain query {q:?} must match the indexed example.com"
            );
        }
        // A domain that is genuinely unknown still matches nothing.
        let e = FilterExpr {
            domains: Some(vec!["nope.example".into()]),
            ..Default::default()
        };
        assert!(f.resolve(&e).unwrap().unwrap().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fast_max_row_matches_the_deserializer() {
        // The fast path must agree with the `roaring` crate's reader on
        // every bitmap shape: empty, a single array container, several
        // containers, a dense bitmap container, and the largest row id.
        let cases: Vec<Vec<u32>> = vec![
            vec![],
            vec![7],
            vec![0, 2, 5],
            vec![1, 70_000],
            vec![1, 70_000, 140_000],
            vec![0, 1, 2, 65_535, 65_536, 131_072],
            (0..5_000u32).collect(), // dense: bitmap container
            (0..100u32).collect(),   // dense: run container (see below)
            vec![u32::MAX],
        ];
        for rows in cases {
            let bm: RoaringBitmap = rows.iter().copied().collect();
            let mut bytes = Vec::new();
            bm.serialize_into(Cursor::new(&mut bytes)).unwrap();
            assert_eq!(
                serialized_max_row(&bytes),
                bm.max(),
                "fast path disagrees with the reader for {} rows",
                rows.len()
            );
        }
    }

    #[test]
    fn fast_max_row_matches_the_deserializer_on_random_bitmaps() {
        // Deterministic pseudo-random bitmaps of every shape (empty, tiny,
        // dense, multi-container, high row ids): the fast path must agree
        // with the crate's reader on all of them.
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for case in 0..200 {
            let n = (next() % 40) as usize;
            let span = 1 + (next() % 200_000) as u32;
            let mut rows: Vec<u32> = Vec::with_capacity(n);
            for _ in 0..n {
                rows.push((next() % u64::from(span)) as u32);
            }
            rows.sort_unstable();
            rows.dedup();
            let bm: RoaringBitmap = rows.iter().copied().collect();
            let mut bytes = Vec::new();
            bm.serialize_into(Cursor::new(&mut bytes)).unwrap();
            assert_eq!(
                serialized_max_row(&bytes),
                bm.max(),
                "case {case}: {n} rows in span {span}"
            );
        }
    }

    #[test]
    fn fast_max_row_reads_run_container_bitmaps() {
        // scripts/build_filter.py writes its bitmaps with pyroaring, which
        // uses the run-container variant of the portable format (cookie
        // 12347): a run bitmap, then the descriptive headers, then offset
        // headers only when there are 4+ containers. These two blobs were
        // produced by pyroaring (BitMap(range(0, 100)) and
        // BitMap of four 10-row runs); the crate's reader accepts them, so
        // the fast path must too.
        let one_run: Vec<u8> = (0..15)
            .map(|i| {
                u8::from_str_radix(&"3b3000000100006300010000006300"[i * 2..i * 2 + 2], 16).unwrap()
            })
            .collect();
        assert_eq!(serialized_max_row(&one_run), Some(99));
        let four_runs: Vec<u8> = (0..61)
            .map(|i| u8::from_str_radix(
                &"3b3003000f00000900010009000200090003000900250000002b00000031000000370000000100000009000100701109000100e0220900010050340900"[i * 2..i * 2 + 2],
                16,
            ).unwrap())
            .collect();
        assert_eq!(serialized_max_row(&four_runs), Some(210_009));
        // ... and the crate agrees on both.
        assert_eq!(
            RoaringBitmap::deserialize_from(Cursor::new(&one_run))
                .unwrap()
                .max(),
            Some(99)
        );
        assert_eq!(
            RoaringBitmap::deserialize_from(Cursor::new(&four_runs))
                .unwrap()
                .max(),
            Some(210_009)
        );
    }

    #[test]
    fn check_row_space_does_not_deserialize_bitmaps() {
        // The load-time row-space check must not materialize every
        // bitmap: that was 2.1s of start-up on a 50k-doc index. The
        // per-facet counter counts fallbacks to the full deserializer.
        let dir = fixture_dir("fastpath");
        let f = load(&dir).unwrap();
        assert!(f.check_row_space(3).is_ok());
        assert_eq!(
            f.domains
                .fallbacks
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "domains facet fell back"
        );
        assert_eq!(
            f.dates.fallbacks.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "dates facet fell back"
        );
        assert_eq!(
            f.terms.fallbacks.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "terms facet fell back"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn check_row_space_still_rejects_corrupt_bitmaps() {
        // A bitmap the fast path cannot read must fall back to the
        // crate's reader, which rejects it — a corrupt filter still fails
        // the load instead of loading silently.
        let dir = std::env::temp_dir().join("ann-core-filter-test-corrupt");
        let _ = fs::remove_dir_all(&dir);
        let fdir = subdir(&dir);
        fs::create_dir_all(&fdir).unwrap();
        let garbage = b"not a roaring bitmap at all";
        fs::write(fdir.join(format!("{DOMAINS}.bin")), garbage).unwrap();
        fs::write(
            fdir.join(format!("{DOMAINS}.json")),
            r#"{"example.com": [0, 27]}"#,
        )
        .unwrap();
        // The other two facets are missing on purpose: load fails before
        // the row-space check, so write them too.
        for name in [DATES, TERMS] {
            fs::write(fdir.join(format!("{name}.bin")), b"").unwrap();
            fs::write(fdir.join(format!("{name}.json")), "{}").unwrap();
        }
        let f = load(&dir).unwrap();
        let err = f.check_row_space(3).unwrap_err();
        assert!(err.contains("corrupt"), "{err}");
        assert!(err.contains("example.com"), "{err}");
        // The corrupt bitmap went through the fallback, not the fast path.
        assert_eq!(
            f.domains
                .fallbacks
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn check_row_space_rejects_stale_row_ids() {
        let dir = fixture_dir("rowspace");
        let f = load(&dir).unwrap();
        assert!(f.check_row_space(3).is_ok());
        assert!(f.check_row_space(2).unwrap_err().contains("row id 2"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_fails_fast_on_missing_or_malformed() {
        let dir = std::env::temp_dir().join("ann-core-filter-test-bad");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // Empty dir: no filter/ subdir at all — missing .bin errors.
        assert!(load(&dir).unwrap_err().contains("cannot read"));
        // Present-but-garbage JSON offset table errors.
        write_partial(&dir, DOMAINS, b"junk", b"not json");
        assert!(load(&dir).unwrap_err().contains("not a JSON offset table"));
        // Valid table naming a slice past the end of the blob errors.
        write_partial(&dir, DOMAINS, b"junk", br#"{"x": [0, 99]}"#);
        // Missing sibling facet (dates) errors before any bitmap is parsed.
        assert!(load(&dir).unwrap_err().contains("cannot read"));
        let _ = fs::remove_dir_all(&dir);
    }
}
