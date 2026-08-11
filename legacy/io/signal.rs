/*!
Aggregate signal output.

Purpose:
    `SignalWriter` persists compact aggregate state samples under
    `{output_path}/signal/{n}.json`. The samples-per-chunk count is fixed at
    construction from the configured byte budget and state length.
*/

use std::fs::{File, create_dir_all, read_to_string};
use std::io::{BufWriter, Error, ErrorKind, Result};
use std::path::{Path, PathBuf};

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use super::WriterStats;
use crate::{Mode, SystemState};

const ESTIMATED_JSON_FLOAT_BYTES: usize = 24;
const ESTIMATED_SAMPLE_OVERHEAD_BYTES: usize = 192;
const ESTIMATED_FILE_OVERHEAD_BYTES: usize = 512;

#[derive(Clone, Serialize, Deserialize)]
pub struct SignalRecord<T> {
    pub time: usize,
    pub state: Array1<T>,
    pub mass: T,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SignalSeries<T> {
    pub file: usize,
    pub mode: Mode<T>,
    pub samples: Vec<SignalRecord<T>>,
}

pub struct SignalWriter {
    dir: PathBuf,
    mode: Mode<f64>,
    sample_bytes: usize,
    samples_per_chunk: usize,
    file_index: usize,
    stats: WriterStats,
    samples: Vec<SignalRecord<f64>>,
}

impl SignalWriter {
    pub fn new(
        output_path: &Path,
        mode: Mode<f64>,
        max_bytes: usize,
        state_len: usize,
    ) -> Result<Self> {
        let dir = output_path.join("signal");
        create_dir_all(&dir).map_err(|e| {
            Error::new(
                e.kind(),
                format!("SignalWriter::new: create dir {}: {e}", dir.display()),
            )
        })?;
        let sample_bytes = estimate_signal_sample(state_len);
        let chunk_samples = samples_per_chunk(max_bytes, sample_bytes);

        Ok(Self {
            dir,
            mode,
            sample_bytes,
            samples_per_chunk: chunk_samples,
            file_index: 1,
            stats: WriterStats::default(),
            samples: Vec::with_capacity(chunk_samples),
        })
    }

    pub fn push(&mut self, gs: &SystemState<f64>) -> Result<()> {
        if self.samples.len() >= self.samples_per_chunk {
            self.flush()?;
        }

        self.samples.push(SignalRecord {
            time: gs.time,
            state: gs.state.clone(),
            mass: gs.mass,
        });
        self.stats.samples += 1;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<WriterStats> {
        self.flush()?;
        Ok(self.stats)
    }

    fn flush(&mut self) -> Result<()> {
        if self.samples.is_empty() {
            return Ok(());
        }

        let file_path = self.dir.join(format!("{}.json", self.file_index));
        let file = File::create(&file_path).map_err(|e| {
            Error::new(
                e.kind(),
                format!("SignalWriter::flush: create {}: {e}", file_path.display()),
            )
        })?;
        let writer = BufWriter::new(file);
        let series_sample_len = self.samples.len();
        let series = SignalSeries {
            file: self.file_index,
            mode: self.mode.clone(),
            samples: std::mem::take(&mut self.samples),
        };

        serde_json::to_writer(writer, &series).map_err(|e| {
            Error::new(
                ErrorKind::InvalidData,
                format!(
                    "SignalWriter::flush: serialize {}: {e}",
                    file_path.display()
                ),
            )
        })?;

        self.file_index += 1;
        self.stats.files += 1;
        self.stats.estimated_bytes = self
            .stats
            .estimated_bytes
            .saturating_add(estimate_file_bytes(self.sample_bytes, series_sample_len));
        Ok(())
    }
}

pub fn load_signal_series(path: &Path) -> Result<SignalSeries<f64>> {
    let raw = read_to_string(path).map_err(|e| {
        Error::new(
            e.kind(),
            format!("load_signal_series: read {}: {e}", path.display()),
        )
    })?;

    serde_json::from_str(&raw).map_err(|e| {
        Error::new(
            ErrorKind::InvalidData,
            format!("load_signal_series: deserialize {}: {e}", path.display()),
        )
    })
}

fn samples_per_chunk(max_bytes: usize, sample_bytes: usize) -> usize {
    max_bytes
        .saturating_sub(ESTIMATED_FILE_OVERHEAD_BYTES)
        .checked_div(sample_bytes.max(1))
        .unwrap_or(0)
        .max(1)
}

fn estimate_file_bytes(sample_bytes: usize, samples: usize) -> usize {
    ESTIMATED_FILE_OVERHEAD_BYTES.saturating_add(sample_bytes.saturating_mul(samples))
}

#[inline]
fn estimate_signal_sample(state_len: usize) -> usize {
    state_len
        .saturating_add(1)
        .saturating_mul(ESTIMATED_JSON_FLOAT_BYTES)
        .saturating_add(ESTIMATED_SAMPLE_OVERHEAD_BYTES)
}
