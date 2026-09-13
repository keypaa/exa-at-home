// crates/ann-core/src/lib.rs (Task 1 stub; replaced by the M1 module in Task 4)
use pyo3::{prelude::*, Bound, types::PyModule};

#[pymodule]
fn ann_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
