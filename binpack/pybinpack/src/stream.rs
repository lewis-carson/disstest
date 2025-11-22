use std::{
    fs::File,
    path::{Path, PathBuf},
};

use pyo3::{prelude::*, types::PyDict};
use sfbinpack::{CompressedReaderError, CompressedTrainingDataEntryReader, TrainingDataEntry};

use crate::{
    batch::{FeatureSet, SparseBatchData},
    error::LoaderError,
    skip::{SkipConfig, SkipState},
};

#[pyclass(name = "SparseBatchStream", unsendable)]
pub struct PySparseBatchStream {
    feature_set: FeatureSet,
    batch_size: usize,
    source: EntrySource,
    skip_state: Option<SkipState>,
}

#[pymethods]
impl PySparseBatchStream {
    #[new]
    #[pyo3(signature = (feature_set, files, batch_size, skip_config=None, cyclic=false, num_workers=1))]
    fn new(
        feature_set: &str,
        files: Vec<String>,
        batch_size: usize,
        skip_config: Option<&PyDict>,
        cyclic: bool,
        num_workers: usize,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "batch_size must be greater than zero",
            ));
        }

        let feature_set = FeatureSet::try_from_name(feature_set)?;
        let paths = files.into_iter().map(PathBuf::from).collect::<Vec<_>>();
        let source = EntrySource::new(paths, cyclic)?;
        let skip_cfg = parse_skip_config(skip_config)?;
        let skip_state = SkipState::maybe_new(skip_cfg);

        // currently single-threaded but we keep the parameter for API parity
        let _ = num_workers;

        Ok(Self {
            feature_set,
            batch_size,
            source,
            skip_state,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PySparseBatchStream>> {
        Ok(slf.into())
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match self.next_batch_data() {
            Ok(Some(batch)) => batch.into_py(py).map(Some),
            Ok(None) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    pub fn next_batch(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match self.next_batch_data() {
            Ok(Some(batch)) => batch.into_py(py).map(Some),
            Ok(None) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }
}

impl PySparseBatchStream {
    fn next_batch_data(&mut self) -> Result<Option<SparseBatchData>, LoaderError> {
        let mut buffer = Vec::with_capacity(self.batch_size);
        while buffer.len() < self.batch_size {
            match self.source.next_entry()? {
                Some(entry) => {
                    if let Some(skip) = &mut self.skip_state {
                        if !skip.should_keep(&entry) {
                            continue;
                        }
                    }
                    buffer.push(entry);
                }
                None => break,
            }
        }

        if buffer.is_empty() {
            Ok(None)
        } else {
            Ok(Some(SparseBatchData::from_entries(
                buffer,
                self.feature_set,
            )))
        }
    }
}

struct EntrySource {
    files: Vec<PathBuf>,
    reader: Option<CompressedTrainingDataEntryReader<File>>,
    file_idx: usize,
    cyclic: bool,
}

impl EntrySource {
    fn new(files: Vec<PathBuf>, cyclic: bool) -> Result<Self, LoaderError> {
        if files.is_empty() {
            return Err(LoaderError::NoFiles);
        }

        Ok(Self {
            files,
            reader: None,
            file_idx: 0,
            cyclic,
        })
    }

    fn next_entry(&mut self) -> Result<Option<TrainingDataEntry>, LoaderError> {
        loop {
            if self.reader.is_none() && !self.advance_reader()? {
                return Ok(None);
            }

            if let Some(reader) = self.reader.as_mut() {
                if reader.has_next() {
                    let entry = reader.next();
                    return Ok(Some(entry));
                } else {
                    self.reader = None;
                }
            }
        }
    }

    fn advance_reader(&mut self) -> Result<bool, LoaderError> {
        let total_files = self.files.len();
        let mut attempts = 0;

        while attempts < total_files {
            if self.file_idx >= self.files.len() {
                if self.cyclic {
                    self.file_idx = 0;
                } else {
                    break;
                }
            }

            let path = self.files[self.file_idx].clone();
            self.file_idx += 1;
            attempts += 1;

            match open_reader(&path) {
                Ok(Some(reader)) => {
                    self.reader = Some(reader);
                    return Ok(true);
                }
                Ok(None) => continue,
                Err(err) => return Err(err),
            }
        }

        Ok(false)
    }
}

fn open_reader(
    path: &Path,
) -> Result<Option<CompressedTrainingDataEntryReader<File>>, LoaderError> {
    let file = File::open(path).map_err(|err| {
        LoaderError::Io(std::io::Error::new(
            err.kind(),
            format!("{}: {}", path.display(), err),
        ))
    })?;

    match CompressedTrainingDataEntryReader::new(file) {
        Ok(reader) => Ok(Some(reader)),
        Err(CompressedReaderError::EndOfFile) => Ok(None),
        Err(err) => Err(LoaderError::from(err)),
    }
}

fn parse_skip_config(dict: Option<&PyDict>) -> PyResult<SkipConfig> {
    let mut cfg = SkipConfig::default();
    if let Some(d) = dict {
        if let Some(value) = d.get_item("filtered")? {
            cfg.filtered = value.extract::<bool>()?;
        }
        if let Some(value) = d.get_item("random_fen_skipping")? {
            cfg.random_fen_skipping = value.extract::<i32>()?;
        }
        if let Some(value) = d.get_item("wld_filtered")? {
            cfg.wld_filtered = value.extract::<bool>()?;
        }
        if let Some(value) = d.get_item("early_fen_skipping")? {
            cfg.early_fen_skipping = value.extract::<i32>()?;
        }
        if let Some(value) = d.get_item("simple_eval_skipping")? {
            cfg.simple_eval_skipping = value.extract::<i32>()?;
        }
        if let Some(value) = d.get_item("param_index")? {
            cfg.param_index = value.extract::<i32>()?;
        }
    }

    Ok(cfg)
}
