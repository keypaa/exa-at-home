// crates/ann-core/src/lib.rs
mod filter;
mod index;
mod ivf;
pub mod lut;
pub mod quant;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyModule, Bound};
use std::cmp::Ordering;

/// Descending score comparison (best score first) that is a *total* order.
///
/// `f32::partial_cmp` returns `None` when either side is NaN, so the old
/// `partial_cmp(..).unwrap()` ranking sorts panicked inside Rust and
/// surfaced to Python as a `pyo3_runtime.PanicException`. NaN now compares
/// equal to NaN and ranks last (worst) against every real score.
pub(crate) fn cmp_desc(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater, // a is NaN -> a sorts after b
        (false, true) => Ordering::Less,
        (false, false) => b.total_cmp(&a),
    }
}

/// Reject a non-finite (NaN/Inf) query vector.
///
/// A NaN query poisons every score it takes part in; that is a caller bug,
/// not a valid search, so it is refused with a clean ValueError instead of
/// being silently ranked.
fn check_finite_query(q: &[f32], what: &str) -> Result<(), String> {
    if let Some(i) = q.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "{what}: query value at index {i} is not finite ({:?}) - \
             queries must be finite float32 values",
            q[i]
        ));
    }
    Ok(())
}

/// Does this filter expression constrain anything at all?
///
/// An explicitly empty value list is "no constraint" (the same as not
/// passing the kwarg), so only a non-empty list or a month span asks for a
/// filter index.
fn expr_is_empty(expr: &filter::FilterExpr) -> bool {
    expr.domains.as_ref().is_none_or(|v| v.is_empty())
        && expr.month_range.is_none()
        && expr.terms.as_ref().is_none_or(|v| v.is_empty())
}

/// Resolve the coarse-candidate count of an `IvfIndex.search` call.
///
/// The argument used to be called `top_k`, but in this binding it never
/// meant "top k results": it is the number of COARSE candidates the ANN
/// returns (e.g. 50), which the caller reranks down to its own top-k
/// (e.g. 10). `top_k` is still accepted as a keyword only to reject it
/// with a message that names the new argument — a silent alias would
/// keep the trap alive (`top_k=10` would quietly return ten coarse
/// candidates, not ten final results).
///
/// Returns the message on error (the caller turns it into a
/// ValueError); kept free of PyO3 types so it stays unit-testable.
fn resolve_n_coarse(n_coarse: Option<usize>, top_k: Option<usize>) -> Result<usize, String> {
    match (n_coarse, top_k) {
        (Some(n), None) => Ok(n),
        (None, None) => Err(
            "IvfIndex.search: n_coarse is required (the number of coarse \
             candidates to retrieve, e.g. 50) - it used to be called top_k"
                .into(),
        ),
        (None, Some(_)) => Err(
            "IvfIndex.search: the 'top_k' argument was renamed to 'n_coarse' \
             - in this binding it is the number of coarse candidates to \
             retrieve (e.g. 50), not the number of final results; pass \
             n_coarse=<value> instead"
                .into(),
        ),
        (Some(_), Some(_)) => Err(
            "IvfIndex.search: pass either n_coarse or its old name top_k, not both".into(),
        ),
    }
}

#[pyclass]
struct AnnIndex {
    rows: Vec<f32>,
    n: usize,
    dim: usize,
    ids: Vec<String>,
    codes: Vec<[u8; 32]>,
}

#[pymethods]
impl AnnIndex {
    #[new]
    fn new(mat: PyReadonlyArray2<f32>, ids: Vec<String>) -> PyResult<Self> {
        let a = mat.as_array();
        if a.nrows() != ids.len() {
            return Err(PyValueError::new_err(format!(
                "AnnIndex::new: {} matrix rows but {} ids",
                a.nrows(),
                ids.len()
            )));
        }
        // Element-wise copy (not a flat as_slice().unwrap()): tolerates
        // non-contiguous / Fortran-order matrices instead of panicking.
        let mut rows = Vec::with_capacity(a.nrows() * a.ncols());
        for i in 0..a.nrows() {
            let row = a.row(i);
            rows.extend(row.iter().copied());
        }
        Ok(Self {
            rows,
            n: a.nrows(),
            dim: a.ncols(),
            ids,
            codes: Vec::new(),
        })
    }

    #[staticmethod]
    fn from_binary(pack: PyReadonlyArray2<u8>, ids: Vec<String>) -> PyResult<Self> {
        let a = pack.as_array();
        if a.ncols() != 32 {
            return Err(PyValueError::new_err(format!(
                "from_binary expects n×32 uint8 codes, got ncols={}",
                a.ncols()
            )));
        }
        if a.nrows() != ids.len() {
            return Err(PyValueError::new_err(format!(
                "from_binary: {} code rows but {} ids",
                a.nrows(),
                ids.len()
            )));
        }
        // Element-wise copy (not a flat as_slice().unwrap()): tolerates
        // non-contiguous / Fortran-order pack arrays instead of panicking.
        let mut codes = Vec::with_capacity(a.nrows());
        for i in 0..a.nrows() {
            let row = a.row(i);
            let mut c = [0u8; 32];
            for (j, v) in row.iter().enumerate() {
                c[j] = *v;
            }
            codes.push(c);
        }
        Ok(Self {
            rows: Vec::new(),
            n: a.nrows(),
            dim: 256,
            ids,
            codes,
        })
    }

    fn search(
        &self,
        py: Python<'_>,
        q: PyReadonlyArray1<f32>,
        top_k: usize,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        // Copy the query out BEFORE releasing the GIL: the array wrapper
        // holds the GIL guard and cannot cross the allow_threads boundary
        // (E0277). 256 floats — the copy is negligible next to the scan.
        // Element-wise copy (not as_slice().unwrap()): tolerates
        // non-contiguous / strided query arrays instead of panicking.
        let q: Vec<f32> = match q.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => q.as_array().iter().copied().collect(),
        };
        if q.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "AnnIndex.search: query dim {} != index dim {}",
                q.len(),
                self.dim
            )));
        }
        // A non-finite query is a caller error: refuse it before the scan
        // (the ranking sort is total now, but NaN scores are meaningless).
        check_finite_query(&q, "AnnIndex.search").map_err(PyValueError::new_err)?;
        py.allow_threads(|| {
            let q = q.as_slice();
            let mut scored: Vec<(u32, f32)> = (0..self.n as u32)
                .map(|i| {
                    let row = &self.rows[i as usize * self.dim..(i as usize + 1) * self.dim];
                    (i, quant::dot(row, q))
                })
                .collect();
            scored.sort_by(|a, b| cmp_desc(a.1, b.1));
            scored.truncate(top_k);
            Ok((
                scored
                    .iter()
                    .map(|(i, _)| self.ids[*i as usize].clone())
                    .collect(),
                scored.iter().map(|(_, s)| *s).collect(),
            ))
        })
    }

    fn search_binary(
        &self,
        py: Python<'_>,
        q: PyReadonlyArray1<f32>,
        top_k: usize,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        if self.codes.is_empty() {
            return Err(PyValueError::new_err(
                "search_binary needs an index built with from_binary",
            ));
        }
        // Copy out BEFORE allow_threads (E0277, same as search) and handle
        // strided/non-contiguous input: fall back to a gathered Vec<float>.
        let q: Vec<f32> = match q.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => q.as_array().iter().copied().collect(),
        };
        if q.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "search_binary: query dim {} != index dim {}",
                q.len(),
                self.dim
            )));
        }
        // A non-finite query is a caller error: refuse it before the scan.
        check_finite_query(&q, "AnnIndex.search_binary").map_err(PyValueError::new_err)?;
        py.allow_threads(|| {
            let q = q.as_slice();
            let mut scored: Vec<(u32, f32)> = self
                .codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i as u32, quant::binary_dot_packed(c, q)))
                .collect();
            scored.sort_by(|a, b| cmp_desc(a.1, b.1));
            scored.truncate(top_k);
            Ok((
                scored
                    .iter()
                    .map(|(i, _)| self.ids[*i as usize].clone())
                    .collect(),
                scored.iter().map(|(_, s)| *s).collect(),
            ))
        })
    }
}

/// M3 IVF index over a versioned index/ dir. Routing is exact float dot
/// over the centroids; per-list scan is M4 LUT-scored (`lut::build` once per
/// query + `lut::score` per doc).
/// `nprobe == K` scans every list, so it must equal M2 brute force
/// (within fp reassociation).
///
/// `search` returns the `n_coarse` best ANN candidates (e.g. 50) — the
/// caller reranks them down to its own top-k (e.g. 10). The argument used
/// to be called `top_k`, which read like a result count and was not one.
#[pyclass]
struct IvfIndex {
    inner: index::Loaded,
}

#[pymethods]
impl IvfIndex {
    #[staticmethod]
    fn load(dir: String) -> PyResult<Self> {
        Ok(Self {
            inner: index::load_py(&dir)?,
        })
    }

    #[pyo3(signature = (q, nprobe, n_coarse=None, allow=None, domains=None, month_range=None, terms=None, top_k=None))]
    fn search(
        &self,
        py: Python<'_>,
        q: PyReadonlyArray1<f32>,
        nprobe: usize,
        n_coarse: Option<usize>,
        allow: Option<Vec<u32>>,
        domains: Option<Vec<String>>,
        month_range: Option<(String, String)>,
        terms: Option<Vec<String>>,
        top_k: Option<usize>,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        let inner = &self.inner;
        // The coarse-candidate count (was misnamed `top_k`): the number of
        // ANN candidates to retrieve, not the number of final results.
        let n_coarse = resolve_n_coarse(n_coarse, top_k).map_err(PyValueError::new_err)?;
        // Copy the query out BEFORE releasing the GIL (E0277, same as
        // AnnIndex::search); also gather strided input instead of panicking.
        let qv: Vec<f32> = match q.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => q.as_array().iter().copied().collect(),
        };
        if qv.len() != index::DIM {
            return Err(PyValueError::new_err(format!(
                "IvfIndex.search: query dim {} != index dim {}",
                qv.len(),
                index::DIM
            )));
        }
        if nprobe == 0 {
            return Err(PyValueError::new_err(
                "IvfIndex.search: nprobe must be >= 1",
            ));
        }
        // A non-finite query is a caller error: refuse it before the scan
        // (the ranking sort is total now, but NaN scores are meaningless).
        check_finite_query(&qv, "IvfIndex.search").map_err(PyValueError::new_err)?;
        // Resolve the filter facets to a row-id bitmap BEFORE releasing the
        // GIL (error paths raise instead of serving partial results). Rust
        // resolves filter files internally — bitmaps never cross the FFI.
        let expr = filter::FilterExpr {
            domains,
            month_range,
            terms,
        };
        let filter_rows: Option<Vec<u32>> = match &inner.filter {
            Some(f) => filter::FilterIdx::rows(&f.resolve(&expr).map_err(|e| {
                PyValueError::new_err(format!("IvfIndex.search: filter resolve: {e}"))
            })?)
            .map(|mut rows| {
                rows.sort_unstable();
                rows
            }),
            None => {
                // An explicitly empty filter list is "no constraint" (the
                // same as not passing the kwarg): only a real filter request
                // needs index/filter/.
                if !expr_is_empty(&expr) {
                    return Err(PyValueError::new_err(
                        "IvfIndex.search: filter kwargs need index/filter/ \
                         (build with scripts/build_filter.py)",
                    ));
                }
                None
            }
        };
        // Intersect the explicit allow-list with the filter bitmap (both
        // sorted ascending, so the ivf scan can binary-search).
        let mut allow_sorted = match (allow, filter_rows) {
            (Some(mut a), Some(b)) => {
                a.sort_unstable();
                let inter: Vec<u32> = {
                    let (mut i, mut j) = (0usize, 0usize);
                    let mut out = Vec::new();
                    while i < a.len() && j < b.len() {
                        if a[i] == b[j] {
                            out.push(a[i]);
                            i += 1;
                            j += 1;
                        } else if a[i] < b[j] {
                            i += 1;
                        } else {
                            j += 1;
                        }
                    }
                    out
                };
                Some(inter)
            }
            (Some(mut a), None) => {
                a.sort_unstable();
                Some(a)
            }
            (None, b) => b,
        };
        // Deduplicate: only needed after the merge above (each input is a set
        // on its own — allow may repeat, bitmap rows never do — but a single
        // dedup covers both paths).
        if let Some(a) = allow_sorted.as_mut() {
            a.dedup();
        }
        let allow_slice = allow_sorted.as_deref();
        py.allow_threads(|| {
            let clusters = ivf::route(&inner.centroids, inner.k, index::DIM, &qv, nprobe);
            let scored = ivf::search_lists(
                &inner.lists,
                &inner.codes,
                &clusters,
                &qv,
                n_coarse,
                allow_slice,
            );
            Ok((
                scored
                    .iter()
                    .map(|(i, _)| inner.ids[*i as usize].clone())
                    .collect(),
                scored.iter().map(|(_, s)| *s).collect(),
            ))
        })
    }
}

#[pymodule]
fn ann_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AnnIndex>()?;
    m.add_class::<IvfIndex>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmp_desc_is_total_and_ranks_nan_last() {
        // Descending score order (best first) that never panics on NaN and
        // never lets a NaN outrank a real score.
        assert_eq!(cmp_desc(2.0, 1.0), Ordering::Less);
        assert_eq!(cmp_desc(1.0, 2.0), Ordering::Greater);
        assert_eq!(cmp_desc(1.0, 1.0), Ordering::Equal);
        assert_eq!(cmp_desc(f32::NAN, 1.0), Ordering::Greater);
        assert_eq!(cmp_desc(1.0, f32::NAN), Ordering::Less);
        assert_eq!(cmp_desc(f32::NAN, f32::NAN), Ordering::Equal);
        // Sorting a slice that contains a NaN must not panic.
        let mut v = [1.0f32, f32::NAN, -1.0, 0.0];
        v.sort_by(|a, b| cmp_desc(*a, *b));
        assert_eq!(v[0], 1.0);
        assert!(v[3].is_nan()); // NaN ranks last (worst)
    }

    #[test]
    fn check_finite_query_rejects_nan_and_inf() {
        assert!(check_finite_query(&[0.0, 1.0], "AnnIndex.search").is_ok());
        let mut q = [0.0f32; 4];
        q[2] = f32::NAN;
        let err = check_finite_query(&q, "AnnIndex.search").unwrap_err();
        assert!(err.contains("not finite"), "{err}");
        assert!(err.contains("index 2"), "{err}");
        q[2] = f32::INFINITY;
        assert!(check_finite_query(&q, "AnnIndex.search")
            .unwrap_err()
            .contains("not finite"));
        q[2] = f32::NEG_INFINITY;
        assert!(check_finite_query(&q, "AnnIndex.search")
            .unwrap_err()
            .contains("not finite"));
    }

    #[test]
    fn n_coarse_is_the_coarse_candidate_count_and_top_k_is_rejected() {
        // The renamed argument: n_coarse is the number of COARSE
        // candidates to retrieve (e.g. 50), not the number of final
        // results. The old keyword is rejected with a message that names
        // the new one — a silent alias would keep the trap alive.
        assert_eq!(resolve_n_coarse(Some(50), None).unwrap(), 50);
        let err = resolve_n_coarse(None, None).unwrap_err();
        assert!(err.contains("n_coarse is required"), "{err}");
        let err = resolve_n_coarse(None, Some(50)).unwrap_err();
        assert!(err.contains("renamed"), "{err}");
        assert!(err.contains("n_coarse"), "{err}");
        assert!(err.contains("top_k"), "{err}");
        let err = resolve_n_coarse(Some(50), Some(10)).unwrap_err();
        assert!(err.contains("not both"), "{err}");
    }

    #[test]
    fn empty_filter_lists_mean_no_constraint() {
        // An explicitly empty value list is "no constraint" (same as not
        // passing the kwarg): it must not demand index/filter/.
        assert!(expr_is_empty(&filter::FilterExpr::default()));
        assert!(expr_is_empty(&filter::FilterExpr {
            domains: Some(vec![]),
            terms: Some(vec![]),
            ..Default::default()
        }));
        // A non-empty list, or any month span, is a real filter request.
        assert!(!expr_is_empty(&filter::FilterExpr {
            domains: Some(vec!["example.com".into()]),
            ..Default::default()
        }));
        assert!(!expr_is_empty(&filter::FilterExpr {
            terms: Some(vec!["alpha".into()]),
            ..Default::default()
        }));
        assert!(!expr_is_empty(&filter::FilterExpr {
            month_range: Some(("2026-01".into(), "2026-02".into())),
            ..Default::default()
        }));
    }
}
