/*!
Persisted output formats.

Purpose:
    `io` owns JSON-facing signal and space streams. Solvers keep using
    `SystemState` internally, while output is written as dedicated aggregate
    signal and spatial snapshot series.
*/

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct WriterStats {
    pub files: usize,
    pub samples: usize,
    pub estimated_bytes: usize,
}

pub mod signal;
pub mod space;
