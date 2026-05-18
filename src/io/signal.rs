/*!
Aggregate signal output.

Purpose:
    `SignalWriter` persists compact aggregate state samples under
    `{output_path}/signal/{n}.json`, flushing chunks according to an estimated
    JSON byte budget.
*/

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Error, ErrorKind, Result};
use std::path::{Path, PathBuf};

use ndarray::Array1;
use serde::{Deserialize, Serialize};

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
    max_bytes: usize,
    file_index: usize,
    estimated_bytes: usize,
    samples: Vec<SignalRecord<f64>>,
}

impl SignalWriter {
    pub fn new(output_path: &Path, mode: Mode<f64>, max_bytes: usize) -> Result<Self> {
        let dir = output_path.join("signal");
        create_dir_all(&dir).map_err(|e| {
            Error::new(
                e.kind(),
                format!("SignalWriter::new: create dir {}: {e}", dir.display()),
            )
        })?;

        Ok(Self {
            dir,
            mode,
            max_bytes: max_bytes.max(ESTIMATED_FILE_OVERHEAD_BYTES),
            file_index: 1,
            estimated_bytes: ESTIMATED_FILE_OVERHEAD_BYTES,
            samples: Vec::new(),
        })
    }

    pub fn push(&mut self, gs: &SystemState<f64>) -> Result<()> {
        let sample_bytes = estimate_signal_sample(gs.state.len());
        if !self.samples.is_empty()
            && self.estimated_bytes.saturating_add(sample_bytes) > self.max_bytes
        {
            self.flush()?;
        }

        self.samples.push(SignalRecord {
            time: gs.time,
            state: gs.state.clone(),
            mass: gs.mass,
        });
        self.estimated_bytes = self.estimated_bytes.saturating_add(sample_bytes);
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.flush()
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
        self.estimated_bytes = ESTIMATED_FILE_OVERHEAD_BYTES;
        Ok(())
    }
}

#[inline]
fn estimate_signal_sample(state_len: usize) -> usize {
    state_len
        .saturating_add(1)
        .saturating_mul(ESTIMATED_JSON_FLOAT_BYTES)
        .saturating_add(ESTIMATED_SAMPLE_OVERHEAD_BYTES)
}
