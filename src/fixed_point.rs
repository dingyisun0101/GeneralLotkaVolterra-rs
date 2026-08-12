//! Verified fixed-point extraction owned by GLV's termination contract.

use std::path::Path;

use scientific_workflow::prelude::{StateError, StorageError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::recording::{COMPLETED_ITERATION_METADATA_KEY, TERMINATION_DIAGNOSTICS_METADATA_KEY};
use crate::terminal_state::{TerminalStateError, open_terminal_state};
use crate::termination::FixedPointDiagnostics;

/// Versioned representation emitted for accepted GLV fixed points.
pub const ACCEPTED_FIXED_POINT_FORMAT: &str = "general-lotka-volterra.accepted-fixed-point.v1";

/// The final GLV state accepted by the configured fixed-point monitor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedFixedPoint {
    format: String,
    iteration: u64,
    physical_time: Option<f64>,
    composition: Vec<f64>,
    diagnostics: FixedPointDiagnostics,
}

impl AcceptedFixedPoint {
    /// Returns the stable document format identifier.
    pub fn format(&self) -> &str {
        &self.format
    }

    /// Returns the exact accepted simulation iteration.
    pub const fn iteration(&self) -> u64 {
        self.iteration
    }

    /// Returns the optional accepted physical time.
    pub const fn physical_time(&self) -> Option<f64> {
        self.physical_time
    }

    /// Returns the normalized final accepted global composition.
    pub fn composition(&self) -> &[f64] {
        &self.composition
    }

    /// Returns the monitor evidence committed with the recording.
    pub const fn diagnostics(&self) -> &FixedPointDiagnostics {
        &self.diagnostics
    }

    /// Serializes the versioned product as compact JSON bytes.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, AcceptedFixedPointError> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Parses and semantically validates one versioned product.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, AcceptedFixedPointError> {
        let fixed_point: Self = serde_json::from_slice(bytes)?;
        fixed_point.validate()?;
        Ok(fixed_point)
    }

    fn validate(&self) -> Result<(), AcceptedFixedPointError> {
        if self.format != ACCEPTED_FIXED_POINT_FORMAT {
            return Err(AcceptedFixedPointError::InvalidProduct(
                "unsupported fixed-point format".to_owned(),
            ));
        }
        if self.diagnostics.iteration != self.iteration {
            return Err(AcceptedFixedPointError::InvalidProduct(
                "diagnostic and product iterations differ".to_owned(),
            ));
        }
        if self.physical_time.is_some_and(|value| !value.is_finite())
            || self.diagnostics.completed_windows == 0
            || self.diagnostics.final_window_samples == 0
            || [
                self.diagnostics.maximum_composition_distance,
                self.diagnostics.relative_mass_range,
                self.diagnostics.maximum_scaled_residual,
            ]
            .into_iter()
            .any(|value| !value.is_finite() || value < 0.0)
        {
            return Err(AcceptedFixedPointError::InvalidProduct(
                "fixed-point coordinates or diagnostics are invalid".to_owned(),
            ));
        }
        validate_composition(&self.composition)
    }
}

/// Opens a completed recording only when GLV itself accepted a fixed point.
pub fn open_accepted_fixed_point(
    recording: impl AsRef<Path>,
) -> Result<AcceptedFixedPoint, AcceptedFixedPointError> {
    let recording = recording.as_ref();
    let terminal_state = open_terminal_state(recording)?;
    if !terminal_state.is_accepted_fixed_point() {
        return Err(AcceptedFixedPointError::NotAcceptedFixedPoint {
            reason: terminal_state.termination_reason().to_owned(),
        });
    }
    let reader = crate::reading::open_completed_glv_recording(recording)?;
    let terminal = reader.terminal_metadata();
    let diagnostics: FixedPointDiagnostics = serde_json::from_value(
        terminal
            .get(TERMINATION_DIAGNOSTICS_METADATA_KEY)
            .cloned()
            .ok_or_else(|| invalid_terminal("missing fixed-point diagnostics"))?,
    )
    .map_err(|error| invalid_terminal(format!("invalid fixed-point diagnostics: {error}")))?;
    let completed_iteration = terminal
        .get(COMPLETED_ITERATION_METADATA_KEY)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| invalid_terminal("missing completed iteration"))?;
    if diagnostics.iteration != completed_iteration
        || terminal_state.iteration() != completed_iteration
    {
        return Err(invalid_terminal(
            "diagnostic, completed, and checkpoint iterations differ",
        ));
    }
    let fixed_point = AcceptedFixedPoint {
        format: ACCEPTED_FIXED_POINT_FORMAT.to_owned(),
        iteration: completed_iteration,
        physical_time: terminal_state.physical_time(),
        composition: terminal_state.composition().to_vec(),
        diagnostics,
    };
    fixed_point.validate()?;
    Ok(fixed_point)
}

fn validate_composition(values: &[f64]) -> Result<(), AcceptedFixedPointError> {
    validate_values(values)?;
    let total = values.iter().sum::<f64>();
    if !total.is_finite() || (total - 1.0).abs() > 1.0e-12 {
        return Err(AcceptedFixedPointError::InvalidProduct(
            "fixed-point composition must sum to one".to_owned(),
        ));
    }
    Ok(())
}

fn validate_values(values: &[f64]) -> Result<(), AcceptedFixedPointError> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(AcceptedFixedPointError::InvalidProduct(
            "fixed-point values must be nonempty, finite, and nonnegative".to_owned(),
        ));
    }
    Ok(())
}

fn invalid_terminal(message: impl Into<String>) -> AcceptedFixedPointError {
    AcceptedFixedPointError::InvalidTerminalMetadata(message.into())
}

/// Failure to reconstruct a fixed point accepted by GLV itself.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AcceptedFixedPointError {
    /// The canonical terminal-state product was unavailable or invalid.
    #[error(transparent)]
    TerminalState(#[from] TerminalStateError),
    /// Workflow rejected recording integrity or decoding.
    #[error(transparent)]
    Storage(#[from] StorageError),
    /// The canonical checkpoint lacked its typed abundance payload.
    #[error(transparent)]
    State(#[from] StateError),
    /// The recording completed for a different scientific reason.
    #[error("recording did not terminate at a fixed point: {reason}")]
    NotAcceptedFixedPoint { reason: String },
    /// GLV terminal evidence was missing or internally inconsistent.
    #[error("invalid GLV terminal metadata: {0}")]
    InvalidTerminalMetadata(String),
    /// A serialized accepted-fixed-point product was invalid.
    #[error("invalid accepted fixed-point product: {0}")]
    InvalidProduct(String),
    /// Versioned JSON could not be encoded or decoded.
    #[error("invalid accepted fixed-point JSON")]
    Json(#[from] serde_json::Error),
}
