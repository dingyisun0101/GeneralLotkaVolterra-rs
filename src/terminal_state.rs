//! Canonical terminal composition shared by ecological simulations.

use std::collections::VecDeque;
use std::path::Path;

use scientific_workflow::prelude::basics::{StateError, StorageError, SystemState};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::recording::{
    COMPLETED_ITERATION_METADATA_KEY, TERMINAL_STATE_METADATA_KEY, TERMINATION_REASON_METADATA_KEY,
    TerminationReason,
};
use crate::{ABUNDANCE_FIELD, AggregateAbundance};

/// Versioned terminal-state document embedded in completed recording metadata.
pub const TERMINAL_STATE_FORMAT: &str = "ecological.terminal-state.v1";
const LEGACY_TERMINAL_STATE_FORMAT: &str = "general-lotka-volterra.terminal-state.v1";

/// Configuration for the bounded trailing estimate used without fixed-point acceptance.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalStatePolicy {
    /// Completed-iteration cadence for terminal-state sampling.
    pub sample_interval_iterations: u64,
    /// Maximum number of normalized compositions retained for the trailing mean.
    pub trailing_window_samples: usize,
}

impl Default for TerminalStatePolicy {
    fn default() -> Self {
        Self {
            sample_interval_iterations: 10,
            trailing_window_samples: 128,
        }
    }
}

impl TerminalStatePolicy {
    /// Validates nonzero sampling controls.
    pub fn validate(self) -> Result<(), TerminalStateError> {
        if self.sample_interval_iterations == 0 || self.trailing_window_samples == 0 {
            return Err(TerminalStateError::InvalidPolicy);
        }
        Ok(())
    }
}

/// Scientific interpretation of the common terminal composition vector.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalStateClassification {
    /// The final state passed GLV's configured fixed-point monitor.
    AcceptedFixedPoint,
    /// The final state is absorbing under the simulator's dynamics.
    AbsorbedState,
    /// The run ended otherwise and this vector is a trailing sample mean.
    TrailingAverage,
}

/// One normalized terminal composition with an explicit scientific classification.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalState {
    format: String,
    classification: TerminalStateClassification,
    termination_reason: String,
    iteration: u64,
    physical_time: Option<f64>,
    composition: Vec<f64>,
    sample_count: usize,
    first_sample_iteration: u64,
    last_sample_iteration: u64,
}

impl TerminalState {
    /// Returns the stable document format.
    pub fn format(&self) -> &str {
        &self.format
    }

    /// Returns whether GLV's monitor accepted the final state as a fixed point.
    pub fn is_accepted_fixed_point(&self) -> bool {
        self.classification == TerminalStateClassification::AcceptedFixedPoint
    }

    /// Returns the terminal vector's scientific classification.
    pub const fn classification(&self) -> TerminalStateClassification {
        self.classification
    }

    /// Returns the successful GLV termination reason.
    pub fn termination_reason(&self) -> &str {
        &self.termination_reason
    }

    /// Returns the final completed simulation iteration.
    pub const fn iteration(&self) -> u64 {
        self.iteration
    }

    /// Returns the optional final physical time.
    pub const fn physical_time(&self) -> Option<f64> {
        self.physical_time
    }

    /// Returns the normalized terminal composition shared by both classifications.
    pub fn composition(&self) -> &[f64] {
        &self.composition
    }

    /// Returns the number of states represented in the composition.
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the first represented sample iteration.
    pub const fn first_sample_iteration(&self) -> u64 {
        self.first_sample_iteration
    }

    /// Returns the last represented sample iteration.
    pub const fn last_sample_iteration(&self) -> u64 {
        self.last_sample_iteration
    }

    /// Serializes the stable JSON product.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, TerminalStateError> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Parses and validates the stable JSON product.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, TerminalStateError> {
        let state: Self = serde_json::from_slice(bytes)?;
        state.validate()?;
        Ok(state)
    }

    fn validate(&self) -> Result<(), TerminalStateError> {
        if self.format != TERMINAL_STATE_FORMAT && self.format != LEGACY_TERMINAL_STATE_FORMAT
            || self.termination_reason.is_empty()
            || self.sample_count == 0
            || self.first_sample_iteration > self.last_sample_iteration
            || self.last_sample_iteration > self.iteration
            || self.physical_time.is_some_and(|value| !value.is_finite())
        {
            return Err(TerminalStateError::InvalidProduct);
        }
        validate_composition(&self.composition)?;
        match self.classification {
            TerminalStateClassification::AcceptedFixedPoint
                if self.termination_reason != "fixed_point"
                    || self.sample_count != 1
                    || self.first_sample_iteration != self.iteration
                    || self.last_sample_iteration != self.iteration =>
            {
                Err(TerminalStateError::InvalidProduct)
            }
            TerminalStateClassification::TrailingAverage
                if matches!(self.termination_reason.as_str(), "fixed_point" | "absorbed") =>
            {
                Err(TerminalStateError::InvalidProduct)
            }
            TerminalStateClassification::AbsorbedState
                if self.termination_reason != "absorbed"
                    || self.sample_count != 1
                    || self.first_sample_iteration != self.iteration
                    || self.last_sample_iteration != self.iteration =>
            {
                Err(TerminalStateError::InvalidProduct)
            }
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Debug)]
struct CompositionSample {
    iteration: u64,
    composition: Vec<f64>,
}

/// Bounded sampler used synchronously by a GLV runner.
pub struct TerminalStateMonitor {
    policy: TerminalStatePolicy,
    samples: VecDeque<CompositionSample>,
}

impl TerminalStateMonitor {
    /// Creates an empty bounded sampler.
    pub fn new(policy: TerminalStatePolicy) -> Result<Self, TerminalStateError> {
        policy.validate()?;
        Ok(Self {
            samples: VecDeque::with_capacity(policy.trailing_window_samples),
            policy,
        })
    }

    /// Returns whether the completed iteration lies on this monitor's cadence.
    pub fn samples_iteration(&self, iteration: u64) -> bool {
        iteration.is_multiple_of(self.policy.sample_interval_iterations)
    }

    /// Samples one state when its completed iteration lies on the configured cadence.
    pub fn observe(&mut self, state: &SystemState) -> Result<(), TerminalStateError> {
        let iteration = state.simulation_time().iteration();
        if self.samples_iteration(iteration) {
            self.push_composition(iteration, &normalized_composition(state)?)?;
        }
        Ok(())
    }

    /// Samples one aggregate composition without depending on a model's spatial state.
    pub fn observe_composition(
        &mut self,
        iteration: u64,
        abundance: &[f64],
    ) -> Result<(), TerminalStateError> {
        if self.samples_iteration(iteration) {
            self.push_composition(iteration, abundance)?;
        }
        Ok(())
    }

    /// Produces the common terminal product, forcing the final state into the tail.
    pub fn finish(
        self,
        final_state: &SystemState,
        reason: &TerminationReason,
    ) -> Result<TerminalState, TerminalStateError> {
        let classification = match reason {
            TerminationReason::FixedPoint(_) => TerminalStateClassification::AcceptedFixedPoint,
            _ => TerminalStateClassification::TrailingAverage,
        };
        self.finish_composition(
            final_state.simulation_time().iteration(),
            final_state.simulation_time().physical_time(),
            &normalized_composition(final_state)?,
            reason.as_str(),
            classification,
        )
    }

    /// Produces a terminal product from aggregate abundance only.
    pub fn finish_composition(
        mut self,
        iteration: u64,
        physical_time: Option<f64>,
        final_abundance: &[f64],
        termination_reason: &str,
        classification: TerminalStateClassification,
    ) -> Result<TerminalState, TerminalStateError> {
        if self
            .samples
            .back()
            .is_none_or(|sample| sample.iteration != iteration)
        {
            self.push_composition(iteration, final_abundance)?;
        }
        let (composition, sample_count, first, last) = match classification {
            TerminalStateClassification::AcceptedFixedPoint
            | TerminalStateClassification::AbsorbedState => {
                (normalize_copy(final_abundance)?, 1, iteration, iteration)
            }
            TerminalStateClassification::TrailingAverage => {
                let first = self
                    .samples
                    .front()
                    .ok_or(TerminalStateError::EmptyWindow)?;
                let last = self.samples.back().ok_or(TerminalStateError::EmptyWindow)?;
                let mut mean = vec![0.0; first.composition.len()];
                for sample in &self.samples {
                    if sample.composition.len() != mean.len() {
                        return Err(TerminalStateError::ShapeChanged);
                    }
                    for (value, sample_value) in mean.iter_mut().zip(&sample.composition) {
                        *value += sample_value;
                    }
                }
                for value in &mut mean {
                    *value /= self.samples.len() as f64;
                }
                normalize_values(&mut mean)?;
                (mean, self.samples.len(), first.iteration, last.iteration)
            }
        };
        let product = TerminalState {
            format: TERMINAL_STATE_FORMAT.to_owned(),
            classification,
            termination_reason: termination_reason.to_owned(),
            iteration,
            physical_time,
            composition,
            sample_count,
            first_sample_iteration: first,
            last_sample_iteration: last,
        };
        product.validate()?;
        Ok(product)
    }

    fn push_composition(
        &mut self,
        iteration: u64,
        abundance: &[f64],
    ) -> Result<(), TerminalStateError> {
        while self.samples.len() >= self.policy.trailing_window_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(CompositionSample {
            iteration,
            composition: normalize_copy(abundance)?,
        });
        Ok(())
    }
}

/// Opens and validates the terminal-state product from a completed GLV recording.
pub fn open_terminal_state(
    recording: impl AsRef<Path>,
) -> Result<TerminalState, TerminalStateError> {
    let reader = crate::reading::open_completed_glv_recording(recording)?;
    let terminal = reader.terminal_metadata();
    let state: TerminalState = serde_json::from_value(
        terminal
            .get(TERMINAL_STATE_METADATA_KEY)
            .cloned()
            .ok_or(TerminalStateError::MissingProduct)?,
    )?;
    state.validate()?;
    let reason = terminal
        .get(TERMINATION_REASON_METADATA_KEY)
        .and_then(serde_json::Value::as_str);
    let iteration = terminal
        .get(COMPLETED_ITERATION_METADATA_KEY)
        .and_then(serde_json::Value::as_u64);
    if reason != Some(state.termination_reason()) || iteration != Some(state.iteration()) {
        return Err(TerminalStateError::MetadataMismatch);
    }
    if state.is_accepted_fixed_point() {
        let final_state = reader.read_latest_state_from_stream(crate::CHECKPOINT_STREAM)?;
        if normalized_composition(&final_state)? != state.composition {
            return Err(TerminalStateError::CheckpointMismatch);
        }
    }
    Ok(state)
}

fn normalized_composition(state: &SystemState) -> Result<Vec<f64>, TerminalStateError> {
    let values = state
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)?
        .iter()
        .copied()
        .collect::<Vec<_>>();
    normalize_copy(&values)
}

fn normalize_copy(values: &[f64]) -> Result<Vec<f64>, TerminalStateError> {
    let mut normalized = values.to_vec();
    normalize_values(&mut normalized)?;
    Ok(normalized)
}

fn normalize_values(values: &mut [f64]) -> Result<(), TerminalStateError> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(TerminalStateError::InvalidComposition);
    }
    let total = values.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(TerminalStateError::InvalidComposition);
    }
    for value in values.iter_mut() {
        *value /= total;
    }
    Ok(())
}

fn validate_composition(values: &[f64]) -> Result<(), TerminalStateError> {
    let mut normalized = values.to_vec();
    normalize_values(&mut normalized)?;
    if values
        .iter()
        .zip(normalized)
        .any(|(value, expected)| (*value - expected).abs() > 1.0e-12)
    {
        return Err(TerminalStateError::InvalidComposition);
    }
    Ok(())
}

/// Failure to configure, produce, or open a terminal-state product.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TerminalStateError {
    /// Sampling cadence and window size must both be nonzero.
    #[error("terminal-state sampling interval and trailing window must be nonzero")]
    InvalidPolicy,
    /// A sampled abundance could not be normalized.
    #[error(
        "terminal-state composition must be nonempty, finite, nonnegative, and have positive mass"
    )]
    InvalidComposition,
    /// Sampled taxon dimensionality changed during one run.
    #[error("terminal-state sample shape changed during one run")]
    ShapeChanged,
    /// No state was available for the trailing estimate.
    #[error("terminal-state trailing window is empty")]
    EmptyWindow,
    /// A completed GLV recording lacks the canonical terminal product.
    #[error("completed GLV recording has no terminal-state product")]
    MissingProduct,
    /// Product identity disagrees with authoritative terminal metadata.
    #[error("terminal-state product disagrees with recording terminal metadata")]
    MetadataMismatch,
    /// An accepted fixed-point product disagrees with the final checkpoint.
    #[error("accepted terminal state disagrees with the final checkpoint")]
    CheckpointMismatch,
    /// A serialized terminal product violates its public contract.
    #[error("invalid terminal-state product")]
    InvalidProduct,
    /// Workflow rejected recording integrity or decoding.
    #[error(transparent)]
    Storage(#[from] StorageError),
    /// The canonical abundance payload was unavailable.
    #[error(transparent)]
    State(#[from] StateError),
    /// Versioned JSON could not be decoded.
    #[error("invalid terminal-state JSON")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use scientific_workflow::prelude::basics::SimulationTime;

    use super::*;

    fn state(iteration: u64, abundance: [f64; 2]) -> SystemState {
        let mut state = crate::load_state_schema()
            .unwrap()
            .create_empty_state(SimulationTime::from_iteration(iteration));
        state
            .insert_payload(ABUNDANCE_FIELD, Array1::from_vec(abundance.to_vec()))
            .unwrap();
        state
    }

    #[test]
    fn maximum_iteration_product_is_the_configured_normalized_trailing_mean() {
        let mut monitor = TerminalStateMonitor::new(TerminalStatePolicy {
            sample_interval_iterations: 1,
            trailing_window_samples: 2,
        })
        .unwrap();
        monitor.observe(&state(0, [1.0, 0.0])).unwrap();
        monitor.observe(&state(1, [0.0, 2.0])).unwrap();
        let final_state = state(2, [3.0, 0.0]);
        monitor.observe(&final_state).unwrap();
        let product = monitor
            .finish(&final_state, &TerminationReason::MaximumIterations)
            .unwrap();
        assert_eq!(
            product.classification(),
            TerminalStateClassification::TrailingAverage
        );
        assert_eq!(product.composition(), [0.5, 0.5]);
        assert_eq!(product.sample_count(), 2);
        assert_eq!(product.first_sample_iteration(), 1);
        assert_eq!(product.last_sample_iteration(), 2);
        assert_eq!(
            TerminalState::from_json_bytes(&product.to_json_bytes().unwrap()).unwrap(),
            product
        );
    }

    #[test]
    fn abundance_only_absorption_uses_the_exact_final_composition() {
        let mut monitor = TerminalStateMonitor::new(TerminalStatePolicy::default()).unwrap();
        monitor.observe_composition(0, &[2.0, 2.0]).unwrap();
        let product = monitor
            .finish_composition(
                3,
                None,
                &[0.0, 4.0],
                "absorbed",
                TerminalStateClassification::AbsorbedState,
            )
            .unwrap();
        assert_eq!(product.composition(), &[0.0, 1.0]);
        assert_eq!(product.sample_count(), 1);
    }
}
