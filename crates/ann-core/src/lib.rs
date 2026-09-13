// crates/ann-core/src/lib.rs
mod index;
mod ivf;
mod quant;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyModule, Bound};

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
        Ok(Self {
            rows: a.as_slice().unwrap().to_vec(),
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
        let q: Vec<f32> = q.as_slice().unwrap().to_vec();
        assert_eq!(q.len(), self.dim);
        py.allow_threads(|| {
            let q = q.as_slice();
            let mut scored: Vec<(u32, f32)> = (0..self.n as u32)
                .map(|i| {
                    let row = &self.rows[i as usize * self.dim..(i as usize + 1) * self.dim];
                    (i, quant::dot(row, q))
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
        py.allow_threads(|| {
            let q = q.as_slice();
            let mut scored: Vec<(u32, f32)> = self
                .codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i as u32, quant::binary_dot_packed(c, q)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
/// over the centroids; per-list scan reuses the M2 `binary_dot_packed`.
/// `nprobe == K` scans every list, so it must equal M2 brute force exactly.
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

    #[pyo3(signature = (q, nprobe, top_k, allow=None))]
    fn search(
        &self,
        py: Python<'_>,
        q: PyReadonlyArray1<f32>,
        nprobe: usize,
        top_k: usize,
        allow: Option<Vec<u32>>,
    ) -> PyResult<(Vec<String>, Vec<f32>)> {
        let inner = &self.inner;
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
        // Sorted once per query so the ivf scan can binary-search.
        let mut allow_sorted = allow;
        if let Some(a) = allow_sorted.as_mut() {
            a.sort_unstable();
        }
        let allow_slice = allow_sorted.as_deref();
        py.allow_threads(|| {
            let clusters = ivf::route(&inner.centroids, inner.k, index::DIM, &qv, nprobe);
            let scored = ivf::search_lists(
                &inner.lists,
                &inner.codes,
                &clusters,
                &qv,
                top_k,
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
