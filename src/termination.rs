//! Deterministic convergence and oscillation monitoring.

use std::collections::VecDeque;

use scientific_workflow::prelude::SystemState;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance};

/// Which canonical state payload supplies composition samples.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminationObservable {
    /// Aggregate taxon abundance.
    GlobalState,
    /// Complete species-last spatial field.
    SpatialField,
}

/// Absolute and state-relative scaling for the model RHS.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualTolerance {
    pub absolute: f64,
    pub relative: f64,
}

/// Whole-window approximate-fixed-point policy.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FixedPointTerminationConfig {
    pub base_window_samples: usize,
    pub confirmation_window_multipliers: Vec<usize>,
    pub composition_tolerance: f64,
    pub relative_mass_tolerance: Option<f64>,
    pub mass_floor: f64,
    pub support_threshold: f64,
    pub residual_tolerance: ResidualTolerance,
}

/// Repeated-cycle policy, deliberately separate from fixed-point detection.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OscillationTerminationConfig {
    pub minimum_period_samples: usize,
    pub maximum_period_samples: usize,
    pub repeated_cycles: usize,
    pub recurrence_tolerance: f64,
    pub minimum_cycle_amplitude: f64,
}

/// Complete opt-in deterministic termination policy.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TerminationPolicy {
    pub start_after_iteration: u64,
    pub sample_interval_iterations: u64,
    pub observable: TerminationObservable,
    pub fixed_point: Option<FixedPointTerminationConfig>,
    pub oscillation: Option<OscillationTerminationConfig>,
}

impl TerminationPolicy {
    /// Builds GLV's standard bounded policy from simple detector toggles.
    ///
    /// Built-in templates use this policy so project authors select scientific
    /// outcomes without configuring the monitor's internal evidence schedule.
    pub(crate) fn automatic(fixed_point: bool, oscillation: bool) -> Option<Self> {
        if !fixed_point && !oscillation {
            return None;
        }
        Some(Self {
            start_after_iteration: 0,
            sample_interval_iterations: 10,
            observable: TerminationObservable::GlobalState,
            fixed_point: fixed_point.then(|| FixedPointTerminationConfig {
                base_window_samples: 16,
                confirmation_window_multipliers: vec![1, 2, 4],
                composition_tolerance: 1.0e-7,
                relative_mass_tolerance: Some(1.0e-7),
                mass_floor: 1.0e-12,
                support_threshold: 1.0e-10,
                residual_tolerance: ResidualTolerance {
                    absolute: 1.0e-10,
                    relative: 1.0e-8,
                },
            }),
            oscillation: oscillation.then_some(OscillationTerminationConfig {
                minimum_period_samples: 2,
                maximum_period_samples: 128,
                repeated_cycles: 3,
                recurrence_tolerance: 1.0e-6,
                minimum_cycle_amplitude: 1.0e-4,
            }),
        })
    }

    pub fn validate(&self) -> Result<(), TerminationError> {
        if self.sample_interval_iterations == 0 {
            return Err(TerminationError::InvalidConfig(
                "sample_interval_iterations must be at least one".to_owned(),
            ));
        }
        if self.fixed_point.is_none() && self.oscillation.is_none() {
            return Err(TerminationError::InvalidConfig(
                "at least one termination detector must be configured".to_owned(),
            ));
        }
        if let Some(config) = &self.fixed_point {
            if config.base_window_samples < 2 {
                return Err(TerminationError::InvalidConfig(
                    "base_window_samples must be at least two".to_owned(),
                ));
            }
            if config.confirmation_window_multipliers.is_empty()
                || config.confirmation_window_multipliers.contains(&0)
            {
                return Err(TerminationError::InvalidConfig(
                    "confirmation_window_multipliers must contain positive values".to_owned(),
                ));
            }
            if config
                .confirmation_window_multipliers
                .windows(2)
                .any(|pair| pair[1] <= pair[0])
            {
                return Err(TerminationError::InvalidConfig(
                    "confirmation_window_multipliers must be strictly increasing".to_owned(),
                ));
            }
            if config
                .confirmation_window_multipliers
                .iter()
                .any(|multiplier| {
                    config
                        .base_window_samples
                        .checked_mul(*multiplier)
                        .is_none()
                })
            {
                return Err(TerminationError::InvalidConfig(
                    "fixed-point window size overflows usize".to_owned(),
                ));
            }
            for (name, value) in [
                ("composition_tolerance", config.composition_tolerance),
                ("mass_floor", config.mass_floor),
                ("support_threshold", config.support_threshold),
                (
                    "residual absolute tolerance",
                    config.residual_tolerance.absolute,
                ),
                (
                    "residual relative tolerance",
                    config.residual_tolerance.relative,
                ),
            ] {
                require_nonnegative_finite(name, value)?;
            }
            if let Some(value) = config.relative_mass_tolerance {
                require_nonnegative_finite("relative_mass_tolerance", value)?;
            }
        }
        if let Some(config) = &self.oscillation {
            if config.minimum_period_samples < 2
                || config.maximum_period_samples < config.minimum_period_samples
                || config.repeated_cycles < 2
            {
                return Err(TerminationError::InvalidConfig(
                    "oscillation periods must start at two, be ordered, and repeat at least twice"
                        .to_owned(),
                ));
            }
            require_nonnegative_finite("recurrence_tolerance", config.recurrence_tolerance)?;
            require_nonnegative_finite("minimum_cycle_amplitude", config.minimum_cycle_amplitude)?;
            if config
                .repeated_cycles
                .checked_add(1)
                .and_then(|cycles| config.maximum_period_samples.checked_mul(cycles))
                .is_none()
            {
                return Err(TerminationError::InvalidConfig(
                    "oscillation history size overflows usize".to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub fn maximum_history_samples(&self) -> usize {
        self.oscillation
            .as_ref()
            .map(|config| config.maximum_period_samples * (config.repeated_cycles + 1))
            .unwrap_or(1)
            .max(1)
    }
}

fn require_nonnegative_finite(name: &str, value: f64) -> Result<(), TerminationError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(TerminationError::InvalidConfig(format!(
            "{name} must be finite and nonnegative"
        )))
    }
}

/// Auditable diagnostics for one accepted fixed point.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FixedPointDiagnostics {
    pub iteration: u64,
    pub completed_windows: usize,
    pub final_window_samples: usize,
    pub maximum_composition_distance: f64,
    pub relative_mass_range: f64,
    pub maximum_scaled_residual: f64,
}

/// Auditable diagnostics for one accepted orbit.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OscillationDiagnostics {
    pub iteration: u64,
    pub period_samples: usize,
    pub repeated_cycles: usize,
    pub maximum_recurrence_distance: f64,
    pub cycle_amplitude: f64,
}

/// Scientific reason emitted by the convergence monitor.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConvergenceReason {
    FixedPoint(FixedPointDiagnostics),
    Oscillation(OscillationDiagnostics),
}

#[derive(Clone, Debug)]
struct Sample {
    iteration: u64,
    composition: Vec<f64>,
    support: Vec<bool>,
    mass: f64,
    scaled_residual: f64,
}

/// Stateful bounded-memory monitor for one simulation task.
pub struct TerminationMonitor {
    policy: TerminationPolicy,
    fixed_stage: usize,
    fixed_window: Vec<Sample>,
    orbit_history: VecDeque<Sample>,
}

impl TerminationMonitor {
    pub fn new(policy: TerminationPolicy) -> Result<Self, TerminationError> {
        policy.validate()?;
        Ok(Self {
            orbit_history: VecDeque::with_capacity(policy.maximum_history_samples()),
            policy,
            fixed_stage: 0,
            fixed_window: Vec::new(),
        })
    }

    pub fn policy(&self) -> &TerminationPolicy {
        &self.policy
    }

    pub fn should_sample(&self, iteration: u64) -> bool {
        iteration >= self.policy.start_after_iteration
            && (iteration - self.policy.start_after_iteration)
                .is_multiple_of(self.policy.sample_interval_iterations)
    }

    pub fn observe(
        &mut self,
        state: &SystemState,
        scaled_residual: f64,
    ) -> Result<Option<ConvergenceReason>, TerminationError> {
        let iteration = state.simulation_time().iteration();
        if !self.should_sample(iteration) {
            return Ok(None);
        }
        if !scaled_residual.is_finite() || scaled_residual < 0.0 {
            return Err(TerminationError::InvalidResidual { scaled_residual });
        }
        let sample = sample_state(
            state,
            self.policy.observable,
            scaled_residual,
            self.support_threshold(),
        )?;

        if let Some(reason) = self.observe_fixed(sample.clone())? {
            return Ok(Some(reason));
        }
        self.observe_oscillation(sample)
    }

    /// Returns whether the configured fixed-point support has one species.
    ///
    /// This is a cheap preflight for callers whose model makes single-species
    /// support absorbing. It does not itself claim convergence.
    pub fn has_single_supported_species(
        &self,
        state: &SystemState,
    ) -> Result<bool, TerminationError> {
        let Some(config) = &self.policy.fixed_point else {
            return Ok(false);
        };
        let sample = sample_state(state, self.policy.observable, 0.0, config.support_threshold)?;
        Ok(sample
            .support
            .iter()
            .filter(|supported| **supported)
            .count()
            == 1)
    }

    /// Evaluates one absorbing single-species state using the model residual.
    ///
    /// Callers must use this shortcut only when their invariant makes
    /// single-species support absorbing. GLV's built-in runner does so for
    /// mean-field and spatial replicator dynamics, but not population GLV.
    pub fn evaluate_absorbing_fixed_point(
        &self,
        state: &SystemState,
        scaled_residual: f64,
    ) -> Result<Option<ConvergenceReason>, TerminationError> {
        if !scaled_residual.is_finite() || scaled_residual < 0.0 {
            return Err(TerminationError::InvalidResidual { scaled_residual });
        }
        let Some(config) = &self.policy.fixed_point else {
            return Ok(None);
        };
        let sample = sample_state(
            state,
            self.policy.observable,
            scaled_residual,
            config.support_threshold,
        )?;
        if sample
            .support
            .iter()
            .filter(|supported| **supported)
            .count()
            != 1
            || scaled_residual > 1.0
        {
            return Ok(None);
        }
        Ok(Some(ConvergenceReason::FixedPoint(FixedPointDiagnostics {
            iteration: sample.iteration,
            completed_windows: 1,
            final_window_samples: 1,
            maximum_composition_distance: 0.0,
            relative_mass_range: 0.0,
            maximum_scaled_residual: scaled_residual,
        })))
    }

    fn support_threshold(&self) -> f64 {
        self.policy
            .fixed_point
            .as_ref()
            .map(|config| config.support_threshold)
            .unwrap_or(0.0)
    }

    fn observe_fixed(
        &mut self,
        sample: Sample,
    ) -> Result<Option<ConvergenceReason>, TerminationError> {
        let Some(config) = &self.policy.fixed_point else {
            return Ok(None);
        };
        if self
            .fixed_window
            .last()
            .is_some_and(|previous| previous.support != sample.support)
        {
            self.fixed_window.clear();
            self.fixed_stage = 0;
        }
        self.fixed_window.push(sample);
        let multiplier = config.confirmation_window_multipliers[self.fixed_stage];
        let required = config
            .base_window_samples
            .checked_mul(multiplier)
            .ok_or_else(|| {
                TerminationError::InvalidConfig(
                    "fixed-point window size overflows usize".to_owned(),
                )
            })?;
        if self.fixed_window.len() < required {
            return Ok(None);
        }

        let diagnostics = evaluate_fixed_window(&self.fixed_window, config, self.fixed_stage + 1)?;
        self.fixed_window.clear();
        if let Some(mut diagnostics) = diagnostics {
            self.fixed_stage += 1;
            if self.fixed_stage == config.confirmation_window_multipliers.len() {
                self.fixed_stage = 0;
                diagnostics.completed_windows = config.confirmation_window_multipliers.len();
                return Ok(Some(ConvergenceReason::FixedPoint(diagnostics)));
            }
        } else {
            self.fixed_stage = 0;
        }
        Ok(None)
    }

    fn observe_oscillation(
        &mut self,
        sample: Sample,
    ) -> Result<Option<ConvergenceReason>, TerminationError> {
        let Some(config) = &self.policy.oscillation else {
            return Ok(None);
        };
        while self.orbit_history.len() >= self.policy.maximum_history_samples() {
            self.orbit_history.pop_front();
        }
        self.orbit_history.push_back(sample);
        Ok(detect_oscillation(&self.orbit_history, config)?.map(ConvergenceReason::Oscillation))
    }
}

fn sample_state(
    state: &SystemState,
    observable: TerminationObservable,
    scaled_residual: f64,
    support_threshold: f64,
) -> Result<Sample, TerminationError> {
    let values = match observable {
        TerminationObservable::GlobalState => state
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .map_err(TerminationError::State)?
            .iter()
            .copied()
            .collect(),
        TerminationObservable::SpatialField => state
            .payload::<SpatialAbundance>(SPACE_FIELD)
            .map_err(TerminationError::State)?
            .as_ref()
            .ok_or(TerminationError::SpatialFieldRequired)?
            .iter()
            .copied()
            .collect(),
    };
    let (composition, mass) = normalize(values)?;
    let support = composition
        .iter()
        .map(|value| *value > support_threshold)
        .collect();
    Ok(Sample {
        iteration: state.simulation_time().iteration(),
        composition,
        support,
        mass,
        scaled_residual,
    })
}

fn normalize(values: Vec<f64>) -> Result<(Vec<f64>, f64), TerminationError> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(TerminationError::InvalidObservable);
    }
    let mass = values.iter().sum::<f64>();
    if !mass.is_finite() || mass <= 0.0 {
        return Err(TerminationError::InvalidObservable);
    }
    Ok((values.into_iter().map(|value| value / mass).collect(), mass))
}

fn evaluate_fixed_window(
    samples: &[Sample],
    config: &FixedPointTerminationConfig,
    completed_windows: usize,
) -> Result<Option<FixedPointDiagnostics>, TerminationError> {
    let first = samples.first().ok_or(TerminationError::InvalidObservable)?;
    if samples.iter().any(|sample| sample.support != first.support) {
        return Ok(None);
    }
    let mut mean = vec![0.0; first.composition.len()];
    for sample in samples {
        if sample.composition.len() != mean.len() {
            return Err(TerminationError::ShapeChanged);
        }
        for (target, value) in mean.iter_mut().zip(&sample.composition) {
            *target += *value;
        }
    }
    for value in &mut mean {
        *value /= samples.len() as f64;
    }
    let maximum_composition_distance = samples
        .iter()
        .map(|sample| jensen_shannon_distance(&sample.composition, &mean))
        .try_fold(0.0_f64, |maximum, value| {
            value.map(|value| maximum.max(value))
        })?;
    let minimum_mass = samples
        .iter()
        .map(|sample| sample.mass)
        .fold(f64::INFINITY, f64::min);
    let maximum_mass = samples
        .iter()
        .map(|sample| sample.mass)
        .fold(0.0_f64, f64::max);
    let mean_mass = samples.iter().map(|sample| sample.mass).sum::<f64>() / samples.len() as f64;
    let relative_mass_range = (maximum_mass - minimum_mass) / mean_mass.max(config.mass_floor);
    let maximum_scaled_residual = samples
        .iter()
        .map(|sample| sample.scaled_residual)
        .fold(0.0_f64, f64::max);
    let mass_passes = config
        .relative_mass_tolerance
        .is_none_or(|tolerance| relative_mass_range <= tolerance);
    if maximum_composition_distance <= config.composition_tolerance
        && mass_passes
        && maximum_scaled_residual <= 1.0
    {
        Ok(Some(FixedPointDiagnostics {
            iteration: samples.last().expect("nonempty window").iteration,
            completed_windows,
            final_window_samples: samples.len(),
            maximum_composition_distance,
            relative_mass_range,
            maximum_scaled_residual,
        }))
    } else {
        Ok(None)
    }
}

fn detect_oscillation(
    history: &VecDeque<Sample>,
    config: &OscillationTerminationConfig,
) -> Result<Option<OscillationDiagnostics>, TerminationError> {
    for period in config.minimum_period_samples..=config.maximum_period_samples {
        let required = period * (config.repeated_cycles + 1);
        if history.len() < required {
            continue;
        }
        let start = history.len() - required;
        let samples = history.iter().skip(start).collect::<Vec<_>>();
        let mut maximum_recurrence_distance = 0.0_f64;
        for cycle in 1..=config.repeated_cycles {
            for offset in 0..period {
                let previous = samples[(cycle - 1) * period + offset];
                let current = samples[cycle * period + offset];
                let distance =
                    jensen_shannon_distance(&previous.composition, &current.composition)?;
                maximum_recurrence_distance = maximum_recurrence_distance.max(distance);
            }
        }
        let anchor = samples[config.repeated_cycles * period];
        let mut cycle_amplitude = 0.0_f64;
        for sample in &samples[config.repeated_cycles * period..] {
            cycle_amplitude = cycle_amplitude.max(jensen_shannon_distance(
                &anchor.composition,
                &sample.composition,
            )?);
        }
        if maximum_recurrence_distance <= config.recurrence_tolerance
            && cycle_amplitude >= config.minimum_cycle_amplitude
        {
            return Ok(Some(OscillationDiagnostics {
                iteration: history.back().expect("required history").iteration,
                period_samples: period,
                repeated_cycles: config.repeated_cycles,
                maximum_recurrence_distance,
                cycle_amplitude,
            }));
        }
    }
    Ok(None)
}

/// Analysis-consistent square-root Jensen-Shannon distance using natural logs.
pub fn jensen_shannon_distance(left: &[f64], right: &[f64]) -> Result<f64, TerminationError> {
    if left.len() != right.len() || left.is_empty() {
        return Err(TerminationError::ShapeChanged);
    }
    let mut divergence = 0.0;
    for (&p, &q) in left.iter().zip(right) {
        if !p.is_finite() || !q.is_finite() || p < 0.0 || q < 0.0 {
            return Err(TerminationError::InvalidObservable);
        }
        let middle = 0.5 * (p + q);
        if p > 0.0 {
            divergence += 0.5 * p * (p / middle).ln();
        }
        if q > 0.0 {
            divergence += 0.5 * q * (q / middle).ln();
        }
    }
    Ok(divergence.max(0.0).sqrt())
}

#[derive(Debug, Error)]
pub enum TerminationError {
    #[error("invalid termination configuration: {0}")]
    InvalidConfig(String),
    #[error("canonical state could not be read: {0}")]
    State(#[source] scientific_workflow::prelude::StateError),
    #[error("spatial_field termination requires a populated spatial state")]
    SpatialFieldRequired,
    #[error("termination observable must be nonempty, finite, nonnegative, and have positive mass")]
    InvalidObservable,
    #[error("termination observable shape changed")]
    ShapeChanged,
    #[error("model returned invalid scaled residual {scaled_residual}")]
    InvalidResidual { scaled_residual: f64 },
    #[error("configured fixed-point monitoring requires a model residual")]
    ResidualUnavailable,
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use scientific_workflow::prelude::{SimulationTime, SystemState};

    use super::*;

    fn fixed_config() -> FixedPointTerminationConfig {
        FixedPointTerminationConfig {
            base_window_samples: 2,
            confirmation_window_multipliers: vec![1, 2],
            composition_tolerance: 1.0e-6,
            relative_mass_tolerance: Some(1.0e-6),
            mass_floor: 1.0e-12,
            support_threshold: 1.0e-9,
            residual_tolerance: ResidualTolerance {
                absolute: 1.0e-8,
                relative: 1.0e-6,
            },
        }
    }

    fn sample(iteration: u64, composition: [f64; 2], residual: f64) -> Sample {
        Sample {
            iteration,
            composition: composition.to_vec(),
            support: composition.map(|value| value > 1.0e-9).to_vec(),
            mass: 1.0,
            scaled_residual: residual,
        }
    }

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
    fn fixed_point_requires_the_rhs_residual_even_when_state_is_confined() {
        let samples = [sample(1, [0.5, 0.5], 2.0), sample(2, [0.5, 0.5], 2.0)];
        assert!(
            evaluate_fixed_window(&samples, &fixed_config(), 1)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn support_change_starts_a_new_window_with_the_transition_sample() {
        let mut monitor = TerminationMonitor::new(TerminationPolicy {
            start_after_iteration: 0,
            sample_interval_iterations: 1,
            observable: TerminationObservable::GlobalState,
            fixed_point: Some(FixedPointTerminationConfig {
                confirmation_window_multipliers: vec![1],
                ..fixed_config()
            }),
            oscillation: None,
        })
        .unwrap();
        assert!(
            monitor
                .observe(&state(0, [0.5, 0.5]), 0.0)
                .unwrap()
                .is_none()
        );
        assert!(
            monitor
                .observe(&state(1, [1.0, 0.0]), 0.0)
                .unwrap()
                .is_none()
        );
        assert!(matches!(
            monitor.observe(&state(2, [1.0, 0.0]), 0.0).unwrap(),
            Some(ConvergenceReason::FixedPoint(_))
        ));
    }

    #[test]
    fn absorbing_shortcut_still_requires_the_rhs_residual() {
        let monitor = TerminationMonitor::new(TerminationPolicy {
            start_after_iteration: 0,
            sample_interval_iterations: 10,
            observable: TerminationObservable::GlobalState,
            fixed_point: Some(fixed_config()),
            oscillation: None,
        })
        .unwrap();
        let monoculture = state(0, [1.0, 0.0]);
        assert!(matches!(
            monitor
                .evaluate_absorbing_fixed_point(&monoculture, 0.0)
                .unwrap(),
            Some(ConvergenceReason::FixedPoint(_))
        ));
        assert!(
            monitor
                .evaluate_absorbing_fixed_point(&monoculture, 2.0)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_recurrent_nontrivial_cycle_is_not_a_fixed_point() {
        let config = OscillationTerminationConfig {
            minimum_period_samples: 2,
            maximum_period_samples: 2,
            repeated_cycles: 2,
            recurrence_tolerance: 1.0e-12,
            minimum_cycle_amplitude: 0.1,
        };
        let history = VecDeque::from([
            sample(1, [0.9, 0.1], 10.0),
            sample(2, [0.1, 0.9], 10.0),
            sample(3, [0.9, 0.1], 10.0),
            sample(4, [0.1, 0.9], 10.0),
            sample(5, [0.9, 0.1], 10.0),
            sample(6, [0.1, 0.9], 10.0),
        ]);
        let detected = detect_oscillation(&history, &config).unwrap().unwrap();
        assert_eq!(detected.period_samples, 2);
        assert!(detected.cycle_amplitude >= config.minimum_cycle_amplitude);
        assert!(
            evaluate_fixed_window(
                &history.iter().cloned().collect::<Vec<_>>(),
                &fixed_config(),
                1,
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn jensen_shannon_distance_is_symmetric_and_support_safe() {
        let left = [1.0, 0.0];
        let right = [0.0, 1.0];
        let forward = jensen_shannon_distance(&left, &right).unwrap();
        let reverse = jensen_shannon_distance(&right, &left).unwrap();
        assert_eq!(forward, reverse);
        assert!(forward.is_finite());
        assert!(forward > 0.0);
    }
}
