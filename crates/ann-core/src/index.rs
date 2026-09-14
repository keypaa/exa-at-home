// crates/ann-core/src/index.rs
// M3: versioned index/ dir loader (manifest + sha256 verify).
// On-disk layout (locked with exa_home/index_format.py SCHEMA):
//   manifest.json   {version:1, dim, n_docs, n_centroids, files:{sha256}}
//   centroids.f32   K×256 float32 little-endian
//   codes.bin       n×32 bytes (M2 packing: bit i = vec[i] > 0,
//                   byte i/8, bit i%8, LSB-first)
//   lists.bin       u32 K, then per list: u32 len + len×u32 row ids (LE)
//   doc_ids.json    ["...", ...] in row order

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: u32 = 1;
pub const DIM: usize = 256;
pub const CODE_BYTES: usize = 32;

const MANIFEST: &str = "manifest.json";
const CENTROIDS_FILE: &str = "centroids.f32";
const CODES_FILE: &str = "codes.bin";
const LISTS_FILE: &str = "lists.bin";
const DOC_IDS_FILE: &str = "doc_ids.json";

#[derive(Debug)]
pub struct Loaded {
    pub centroids: Vec<f32>,
    pub k: usize,
    pub codes: Vec<[u8; 32]>,
    pub lists: Vec<Vec<u32>>,
    pub ids: Vec<String>,
    /// Present when index/filter/ exists (Task 8). Manifest-covered either
    /// way: filter files are checksummed when present, and their absence is
    /// pinned by the manifest `has_filter` flag (see below).
    pub filter: Option<super::filter::FilterIdx>,
}

fn err(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let bytes = fs::read(path)?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(format!("{:x}", h.finalize()))
}

fn read_u32_le(raw: &[u8], off: &mut usize, what: &str) -> Result<u32, String> {
    let end = off
        .checked_add(4)
        .filter(|&e| e <= raw.len())
        .ok_or_else(|| format!("{LISTS_FILE} truncated reading {what}"))?;
    let v = u32::from_le_bytes(raw[*off..end].try_into().unwrap());
    *off = end;
    Ok(v)
}

/// Verify the manifest, then load and shape-check every file.
/// `ids.len() == nrows` is enforced here (not call-site).
pub fn load(dir: &Path) -> Result<Loaded, String> {
    let file = |name: &str| -> PathBuf { dir.join(name) };

    let mraw = fs::read(file(MANIFEST)).map_err(|e| format!("cannot read {MANIFEST}: {e}"))?;
    let m: serde_json::Value =
        serde_json::from_slice(&mraw).map_err(|e| format!("{MANIFEST} is not JSON: {e}"))?;

    let version = m
        .get("version")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("{MANIFEST} missing integer 'version'"))?;
    if version != SCHEMA_VERSION as u64 {
        return Err(format!(
            "unsupported index version {version}: this code reads version {SCHEMA_VERSION}"
        ));
    }
    let dim = m
        .get("dim")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("{MANIFEST} missing integer 'dim'"))?;
    if dim as usize != DIM {
        return Err(format!("index dim {dim} != {DIM}"));
    }
    let files = m
        .get("files")
        .and_then(|v| v.as_object())
        .ok_or_else(|| format!("{MANIFEST} missing object 'files'"))?;
    // Deterministic check order (BTreeMap over the JSON map).
    let want: BTreeMap<&str, &str> = files
        .iter()
        .map(|(k, v)| {
            v.as_str()
                .map(|s| (k.as_str(), s))
                .ok_or_else(|| format!("{MANIFEST}: files[{k}] is not a string"))
        })
        .collect::<Result<_, _>>()?;
    for (name, hex) in &want {
        let got =
            sha256_file(&file(name)).map_err(|e| format!("cannot read index file {name}: {e}"))?;
        if got != *hex {
            return Err(format!(
                "corrupt index file {name}: want {} got {}",
                &hex[..12.min(hex.len())],
                &got[..12]
            ));
        }
    }

    // centroids.f32: K×256 LE float32.
    let craw =
        fs::read(file(CENTROIDS_FILE)).map_err(|e| format!("cannot read {CENTROIDS_FILE}: {e}"))?;
    if craw.len() % (DIM * 4) != 0 {
        return Err(format!(
            "{CENTROIDS_FILE} size {} is not a multiple of {DIM} float32s",
            craw.len()
        ));
    }
    let k = craw.len() / (DIM * 4);
    let centroids: Vec<f32> = craw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    // codes.bin: n×32 bytes.
    let brow = fs::read(file(CODES_FILE)).map_err(|e| format!("cannot read {CODES_FILE}: {e}"))?;
    if brow.len() % CODE_BYTES != 0 {
        return Err(format!(
            "{CODES_FILE} size {} is not a multiple of {CODE_BYTES}",
            brow.len()
        ));
    }
    let n = brow.len() / CODE_BYTES;
    let mut codes = Vec::with_capacity(n);
    for row in brow.chunks_exact(CODE_BYTES) {
        let mut c = [0u8; 32];
        c.copy_from_slice(row);
        codes.push(c);
    }

    // lists.bin: u32 K, then per list u32 len + len×u32 row ids.
    let lraw = fs::read(file(LISTS_FILE)).map_err(|e| format!("cannot read {LISTS_FILE}: {e}"))?;
    let mut off = 0usize;
    let lk = read_u32_le(&lraw, &mut off, "K")? as usize;
    if lk != k {
        return Err(format!(
            "{LISTS_FILE} has K={lk} lists but {CENTROIDS_FILE} has K={k}"
        ));
    }
    let mut lists = Vec::with_capacity(k);
    let mut total: usize = 0;
    for c in 0..k {
        let len = read_u32_le(&lraw, &mut off, &format!("len(list {c})"))? as usize;
        let end = off
            .checked_add(
                len.checked_mul(4)
                    .ok_or_else(|| format!("len(list {c}) overflows"))?,
            )
            .filter(|&e| e <= lraw.len())
            .ok_or_else(|| format!("{LISTS_FILE} truncated in list {c} (len {len})"))?;
        let rows: Vec<u32> = lraw[off..end]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for &r in &rows {
            if (r as usize) >= n {
                return Err(format!(
                    "{LISTS_FILE} list {c} has out-of-range row id {r} (n={n})"
                ));
            }
        }
        off = end;
        total += len;
        lists.push(rows);
    }
    if off != lraw.len() {
        return Err(format!(
            "{LISTS_FILE} has {} trailing bytes after {k} lists",
            lraw.len() - off
        ));
    }
    if total != n {
        return Err(format!(
            "{LISTS_FILE} covers {total} rows but {CODES_FILE} has n={n}"
        ));
    }

    // doc_ids.json: row-order ids, length must equal nrows.
    let draws =
        fs::read(file(DOC_IDS_FILE)).map_err(|e| format!("cannot read {DOC_IDS_FILE}: {e}"))?;
    let ids: Vec<String> = serde_json::from_slice(&draws)
        .map_err(|e| format!("{DOC_IDS_FILE} is not a JSON string list: {e}"))?;
    if ids.len() != n {
        return Err(format!(
            "{DOC_IDS_FILE} has {} ids but {CODES_FILE} has n={n} rows",
            ids.len()
        ));
    }
    let n_docs = ids.len();

    // Task 8: filter/ subdir (optional). When present, all six files must
    // verify: presence/absence is pinned by the manifest `has_filter` flag so
    // a dropped filter/ dir fails fast instead of serving unfiltered queries
    // as if no filter had been requested.
    let filter_dir = super::filter::subdir(dir);
    let filter_present = filter_dir.is_dir();
    let want_filter = m.get("has_filter").and_then(|v| v.as_bool());
    match (filter_present, want_filter) {
        (true, Some(false)) => {
            return Err(format!(
                "index/filter/ present but {MANIFEST} says has_filter=false \
                 (stale filter dir or stale manifest)"
            ));
        }
        (false, Some(true)) => {
            return Err(format!(
                "index/filter/ missing but {MANIFEST} says has_filter=true \
                 (filter files were deleted)"
            ));
        }
        _ => {}
    }
    let filter = if filter_present {
        // Hashes were already verified by the manifest loop above (filter
        // files are entries in the same `files` map); here we only require
        // that all six are pinned, so a legacy manifest that predates the
        // filter build fails here instead of loading unverified bitmaps.
        for name in super::filter::FilterIdx::manifest_files() {
            if !want.contains_key(name) {
                return Err(format!(
                    "index/filter/ present but {MANIFEST} has no checksum \
                     for {name} (manifest predates the filter build)"
                ));
            }
        }
        Some(super::filter::load(dir).map_err(|e| format!("filter load: {e}"))?)
    } else {
        None
    };
    if let Some(f) = &filter {
        f.check_row_space(n_docs)?;
    }

    Ok(Loaded {
        centroids,
        k,
        codes,
        lists,
        ids,
        filter,
    })
}

pub fn load_py(dir: &str) -> PyResult<Loaded> {
    load(Path::new(dir)).map_err(err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Write a minimal valid index/ dir by hand (mirrors the Python test
    /// helper `_write_ivf_index`): K=2 centroids on the x-axis, 4 docs with
    /// 1-hot sign codes, nearest-centroid lists, verifiable manifest.
    fn write_fixture(dir: &Path) {
        // centroids: c0=(+1,0,...), c1=(-1,0,...).
        let mut craw = Vec::new();
        for c in 0..2usize {
            for d in 0..DIM {
                let v: f32 = if d == 0 {
                    if c == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    0.0
                };
                craw.extend_from_slice(&v.to_le_bytes());
            }
        }
        fs::write(dir.join(CENTROIDS_FILE), &craw).unwrap();
        // codes: doc0/doc1 bit0=1, doc2/doc3 bit0=0, rest 0.
        let mut brow = vec![0u8; 4 * CODE_BYTES];
        brow[0 * CODE_BYTES] = 0x01;
        brow[1 * CODE_BYTES] = 0x01;
        fs::write(dir.join(CODES_FILE), &brow).unwrap();
        // lists: K=2, list0=[0,1], list1=[2,3].
        let mut lraw = Vec::new();
        lraw.extend_from_slice(&2u32.to_le_bytes());
        for rows in [&[0u32, 1][..], &[2u32, 3][..]] {
            lraw.extend_from_slice(&(rows.len() as u32).to_le_bytes());
            for r in rows {
                lraw.extend_from_slice(&r.to_le_bytes());
            }
        }
        fs::write(dir.join(LISTS_FILE), &lraw).unwrap();
        fs::write(dir.join(DOC_IDS_FILE), r#"["d0","d1","d2","d3"]"#).unwrap();
        // manifest over the four files (never over itself).
        let mut files = HashMap::new();
        for name in [CENTROIDS_FILE, CODES_FILE, LISTS_FILE, DOC_IDS_FILE] {
            files.insert(name.to_string(), sha256_file(&dir.join(name)).unwrap());
        }
        let m = serde_json::json!({"version": 1, "dim": 256, "n_docs": 4,
                                   "n_centroids": 2, "files": files});
        fs::write(
            dir.join(MANIFEST),
            serde_json::to_string_pretty(&m).unwrap(),
        )
        .unwrap();
    }

    fn fixture_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ann-core-ivf-test-{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir);
        dir
    }

    #[test]
    fn load_parses_locked_layout() {
        let dir = fixture_dir("ok");
        let l = load(&dir).unwrap();
        assert_eq!(l.k, 2);
        assert_eq!(l.centroids.len(), 2 * DIM);
        assert!((l.centroids[0] - 1.0).abs() < 1e-9);
        assert!((l.centroids[DIM] + 1.0).abs() < 1e-9);
        assert_eq!(l.codes.len(), 4);
        assert_eq!(l.codes[0][0], 0x01);
        assert_eq!(l.codes[3][0], 0x00);
        assert_eq!(l.lists, vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(l.ids, vec!["d0", "d1", "d2", "d3"]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_corruption_and_version_and_id_mismatch() {
        // Bit-flip one code byte -> manifest mismatch.
        let dir = fixture_dir("corrupt");
        let mut brow = fs::read(dir.join(CODES_FILE)).unwrap();
        brow[0] ^= 0xFF;
        fs::write(dir.join(CODES_FILE), &brow).unwrap();
        assert!(load(&dir).unwrap_err().contains("corrupt index file"));
        let _ = fs::remove_dir_all(&dir);

        // Bump version -> version error (before any hashing).
        let dir = fixture_dir("version");
        let mraw = fs::read(dir.join(MANIFEST)).unwrap();
        let mut m: serde_json::Value = serde_json::from_slice(&mraw).unwrap();
        m["version"] = serde_json::json!(999);
        fs::write(dir.join(MANIFEST), serde_json::to_string(&m).unwrap()).unwrap();
        assert!(load(&dir)
            .unwrap_err()
            .contains("unsupported index version"));
        let _ = fs::remove_dir_all(&dir);

        // Drop one id -> ids-length error.
        let dir = fixture_dir("ids");
        fs::write(dir.join(DOC_IDS_FILE), r#"["d0","d1","d2"]"#).unwrap();
        let mut files = HashMap::new();
        for name in [CENTROIDS_FILE, CODES_FILE, LISTS_FILE, DOC_IDS_FILE] {
            files.insert(name.to_string(), sha256_file(&dir.join(name)).unwrap());
        }
        let m = serde_json::json!({"version": 1, "dim": 256,
                                   "files": files});
        fs::write(dir.join(MANIFEST), serde_json::to_string(&m).unwrap()).unwrap();
        assert!(load(&dir).unwrap_err().contains("3 ids but"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn route_then_scan_equals_full_scan() {
        // End-to-end logic check inside Rust: routing at nprobe==K visits
        // every list, so route+search_lists == scan of all lists.
        let dir = fixture_dir("logic");
        let l = load(&dir).unwrap();
        let q = vec![1.0f32; DIM];
        let all: Vec<u32> = vec![0, 1];
        let full = super::super::ivf::search_lists(&l.lists, &l.codes, &all, &q, 4, None);
        let routed = super::super::ivf::route(&l.centroids, l.k, DIM, &q, 2);
        assert_eq!(routed, vec![0, 1]); // +x query routes to +x centroid first
        let got = super::super::ivf::search_lists(&l.lists, &l.codes, &routed, &q, 4, None);
        assert_eq!(got, full);
        let _ = fs::remove_dir_all(&dir);
    }
}
