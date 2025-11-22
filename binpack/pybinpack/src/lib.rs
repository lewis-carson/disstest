#![allow(non_local_definitions)]

mod batch;
mod error;
mod skip;
mod stream;

use pyo3::prelude::*;
use stream::PySparseBatchStream;

#[pymodule]
fn binpack_loader(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySparseBatchStream>()?;
    Ok(())
}
