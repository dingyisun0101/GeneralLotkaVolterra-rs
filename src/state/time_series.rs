/// ==============================================================================================
/// ================================= Time Series Container ======================================
/// ==============================================================================================

use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;
use ndarray::{Array1, ArrayD};
use std::fs::{create_dir_all, read_dir, read_to_string, File};
use std::io::{Error, ErrorKind, Result, Write};
use std::path::{Path, PathBuf};

use super::{Mode, SystemState};

/// Snapshot payload without repeated `mode` storage (owned data).
#[derive(Clone, Serialize, Deserialize)]
pub struct SystemStateRecord<T> {
    pub time: usize,
    pub state: Array1<T>,
    pub space: Option<ArrayD<T>>,
    pub mass: T,
}

/// Time series with one shared `mode` and owned sample data.
#[derive(Clone, Serialize, Deserialize)]
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

impl<T> SystemStateTimeSeries<T>
where
    T: DeserializeOwned,
{
    /// Load from:
    /// - `{output_path}/{latest_epoch}.json` if `output_path` is a directory, or
    /// - `output_path` directly if it points to a `.json` file.
    pub fn from(output_path: &Path) -> Result<Self> {
        let file_path = if output_path.extension().and_then(|s| s.to_str()) == Some("json") {
            output_path.to_path_buf()
        } else {
            Self::latest_epoch_file(output_path)?
        };

        let raw = read_to_string(&file_path).map_err(|e| {
            Error::new(
                e.kind(),
                format!("GSTS::from: read {}: {e}", file_path.display()),
            )
        })?;

        serde_json::from_str(&raw).map_err(|e| {
            Error::new(
                ErrorKind::InvalidData,
                format!("GSTS::from: deserialize {}: {e}", file_path.display()),
            )
        })
    }

    fn latest_epoch_file(output_path: &Path) -> Result<PathBuf> {
        let entries = read_dir(output_path).map_err(|e| {
            Error::new(
                e.kind(),
                format!("GSTS::from: read dir {}: {e}", output_path.display()),
            )
        })?;

        let mut latest: Option<(usize, PathBuf)> = None;

        for entry in entries {
            let entry = entry.map_err(|e| {
                Error::new(
                    e.kind(),
                    format!("GSTS::from: iterate dir {}: {e}", output_path.display()),
                )
            })?;

            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }

            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Ok(epoch) = stem.parse::<usize>() else {
                continue;
            };

            if latest.as_ref().is_none_or(|(best, _)| epoch > *best) {
                latest = Some((epoch, path));
            }
        }

        latest.map(|(_, path)| path).ok_or_else(|| {
            Error::new(
                ErrorKind::NotFound,
                format!(
                    "GSTS::from: no epoch json files found in {}",
                    output_path.display()
                ),
            )
        })
    }
}
