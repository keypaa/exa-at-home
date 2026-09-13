// crates/ann-core/src/lib.rs
mod index;
mod quant;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{prelude::*, Bound, types::PyModule};

#[pyclass]
struct AnnIndex {
    rows: Vec<f32>,
    n: usize,
    dim: usize,
    ids: Vec<String>,
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
        })
    }

    fn search(&self, py: Python<'_>, q: PyReadonlyArray1<f32>, top_k: usize)
        -> PyResult<(Vec<String>, Vec<f32>)>
    {
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
            Ok((scored.iter().map(|(i, _)| self.ids[*i as usize].clone()).collect(),
                scored.iter().map(|(_, s)| *s).collect()))
        })
    }
}

#[pymodule]
fn ann_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AnnIndex>()?;
    Ok(())
}
