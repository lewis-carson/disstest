use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Binpack reader error: {0}")]
    Reader(#[from] sfbinpack::CompressedReaderError),
    #[error("no binpack files provided")]
    NoFiles,
    #[error("unsupported feature set '{0}'")]
    UnsupportedFeatureSet(String),
}

impl From<LoaderError> for PyErr {
    fn from(err: LoaderError) -> Self {
        match err {
            LoaderError::Io(e) => PyIOError::new_err(e.to_string()),
            LoaderError::Reader(e) => PyRuntimeError::new_err(e.to_string()),
            LoaderError::NoFiles | LoaderError::UnsupportedFeatureSet(_) => {
                PyValueError::new_err(err.to_string())
            }
        }
    }
}
