/*!
General Lotka-Volterra crate root.

Purpose:
    This crate provides state containers, non-spatial solver machinery, task
    runners, and Cargo examples for ecological dynamical-system experiments.

Current implementation boundary:
    The ready solver path is replicator-form. GLV-named task modules are
    placeholders until a dedicated GLV right-hand side and integrator are added.
*/

/// Target maximum JSON chunk size used by aggregate signal output writers.
pub const SIGNAL_OUTPUT_FILE_SIZE: usize = 128 * 1024 * 1024;

/// Target maximum JSON chunk size used by full spatial snapshot output writers.
pub const SPACE_OUTPUT_FILE_SIZE: usize = 128 * 1024 * 1024;

pub mod io;
pub mod solvers;
pub mod system_state;
pub mod tasks;
pub mod utils;

pub use system_state::{Mode, Scalar, SystemState};
