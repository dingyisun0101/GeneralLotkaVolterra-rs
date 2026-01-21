/// ==============================================================================================
/// ================================= Time Series Container ======================================
/// ==============================================================================================

use serde::Serialize;
use ndarray::{Array1, ArrayD};
use std::fs::{create_dir_all, File};
use std::io::{Error, ErrorKind, Result, Write};
use std::path::Path;

use super::{Mode, SystemState};

/// Snapshot payload without repeated `mode` storage (owned data).
#[derive(Clone, Serialize)]
pub struct SystemStateRecord<T> {
    pub time: usize,
    pub state: Array1<T>,
    pub space: Option<ArrayD<T>>,
    pub mass: T,
}

/// Time series with one shared `mode` and owned sample data.
#[derive(Clone, Serialize)]
pub struct SystemStateTimeSeries<T> {
    pub epoch: usize,
    pub mode: Mode<T>,
    pub samples: Vec<SystemStateRecord<T>>,
}

impl<T> SystemStateTimeSeries<T>
where
    T: Clone,
{
    /// Empty time series (no samples yet).
    #[inline]
    pub fn empty(epoch: usize, mode: Mode<T>) -> Self {
        Self {
            epoch,
            mode,
            samples: Vec::new(),
        }
    }

    /// Add a borrowed snapshot (no copies of state/space/mass data).
    #[inline]
    pub fn add(&mut self, gs: &SystemState<T>) {
        self.samples.push(SystemStateRecord {
            time: gs.time,
            state: gs.state.clone(),
            space: gs.space.clone(),
            mass: gs.mass.clone(),
        });
    }
}

impl<T> SystemStateTimeSeries<T>
where
    T: Serialize,
{
    /// Write the list of samples into `{output_path}/{epoch}.json` (pretty-printed).
    pub fn save(&self, output_path: &Path) -> Result<()> {
        create_dir_all(output_path).map_err(|e| {
            Error::new(
                e.kind(),
                format!("GSTS::save: create dir {}: {e}", output_path.display()),
            )
        })?;

        let file_path = output_path.join(format!("{}.json", self.epoch));
        let json = serde_json::to_string_pretty(&self).map_err(|e| {
            Error::new(
                ErrorKind::InvalidData,
                format!("GSTS::save: serialize {}: {e}", file_path.display()),
            )
        })?;

        let mut file = File::create(&file_path).map_err(|e| {
            Error::new(
                e.kind(),
                format!("GSTS::save: create {}: {e}", file_path.display()),
            )
        })?;

        file.write_all(json.as_bytes()).map_err(|e| {
            Error::new(
                e.kind(),
                format!("GSTS::save: write {}: {e}", file_path.display()),
            )
        })?;

        Ok(())
    }
}
