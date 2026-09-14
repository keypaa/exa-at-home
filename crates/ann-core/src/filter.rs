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

/// One facet: the raw .bin blob plus its {key: [offset, len]} table.
#[derive(Debug)]
struct Facet {
    blob: Vec<u8>,
    offsets: BTreeMap<String, (usize, usize)>,
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
    Ok(Facet { blob, offsets })
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
            acc = and(acc, or_keys(&self.domains, ds)?);
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
    pub fn check_row_space(&self, n_docs: usize) -> Result<(), String> {
        let n = n_docs as u32;
        for (facet_name, facet) in [
            ("domains", &self.domains),
            ("dates", &self.dates),
            ("terms", &self.terms),
        ] {
            for key in facet.offsets.keys() {
                let bm = facet.get(key)?;
                if let Some(max) = bm.max() {
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
