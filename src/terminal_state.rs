//! Shared terminal ecological products in GLV recordings.

use std::path::Path;

use ecological_model_core::terminal_state::TerminalClassification;
use scientific_workflow::prelude::basics::{StateError, StorageError};
use thiserror::Error;

use crate::recording::{COMPLETED_ITERATION_METADATA_KEY, TERMINAL_STATE_METADATA_KEY};

pub use ecological_model_core::terminal_state::{
    AbsorptionDiagnostics, EquilibriumDiagnostics, PeriodicOrbitDiagnostics, StopReason,
    TERMINAL_STATE_FORMAT, TerminalState, TerminationSignal,
};

/// Opens and cross-checks the core terminal product in a completed GLV recording.
pub fn open_terminal_state(
    recording: impl AsRef<Path>,
) -> Result<TerminalState, TerminalStateOpenError> {
    let reader = crate::reading::open_completed_glv_recording(recording)?;
    let value = reader
        .terminal_metadata()
        .get(TERMINAL_STATE_METADATA_KEY)
        .cloned()
        .ok_or(TerminalStateOpenError::MissingProduct)?;
    let state = TerminalState::from_json_bytes(&serde_json::to_vec(&value)?)?;
    let iteration = reader
        .terminal_metadata()
        .get(COMPLETED_ITERATION_METADATA_KEY)
        .and_then(serde_json::Value::as_u64);
    if iteration != Some(state.iteration()) {
        return Err(TerminalStateOpenError::MetadataMismatch);
    }
    if state.classification() == TerminalClassification::Equilibrium {
        let final_state = reader.read_latest_state_from_stream(crate::CHECKPOINT_STREAM)?;
        let abundance = final_state.payload::<crate::AggregateAbundance>(crate::ABUNDANCE_FIELD)?;
        let total = abundance.sum_serial();
        let composition = abundance
            .as_slice()
            .iter()
            .map(|value| value / total)
            .collect::<Vec<_>>();
        if composition != state.composition() {
            return Err(TerminalStateOpenError::CheckpointMismatch);
        }
    }
    Ok(state)
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TerminalStateOpenError {
    #[error("completed GLV recording has no terminal-state product")]
    MissingProduct,
    #[error("terminal product and recording completion metadata disagree")]
    MetadataMismatch,
    #[error("equilibrium terminal composition differs from the final checkpoint")]
    CheckpointMismatch,
    #[error(transparent)]
    Terminal(#[from] ecological_model_core::terminal_state::TerminalStateError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error(transparent)]
    State(#[from] StateError),
}
