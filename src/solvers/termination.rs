/*!
Solver termination checks.

Purpose:
    This module provides shared, opt-in termination logic for non-spatial and
    spatial solvers. It keeps bounded history and evaluates checks only at the
    configured interval so disabled or sparse checks do not materially affect
    the integration hot path.
*/

use std::collections::VecDeque;

use ndarray::{Array1, ArrayD};

use crate::{Mode, SystemState};

/// Which part of a state is compared by steady-state checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminationObservable {
    GlobalState,
    SpatialField,
}

/// Adaptive fixed-point detection over recent checked samples.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveFixedPointConfig {
    pub tolerance: f64,
    pub min_steps: usize,
    pub min_window: usize,
    pub max_window: usize,
    pub stable_checks_required: usize,
}

/// Adaptive oscillatory steady-state detection over recent checked samples.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveOscillationConfig {
    pub tolerance: f64,
    pub min_steps: usize,
    pub min_period: usize,
    pub max_period: usize,
    pub repeats_required: usize,
}

/// Optional steady-state detection.
#[derive(Clone, Copy, Debug)]
pub enum SteadyStateConfig {
    Off,
    Adaptive {
        fixed_point: AdaptiveFixedPointConfig,
        oscillation: Option<AdaptiveOscillationConfig>,
    },
}

/// Termination behavior selected before launching a solver run.
#[derive(Clone, Copy, Debug)]
pub struct TerminationConfig {
    pub monoculture: bool,
    pub survivor_tolerance: Option<f64>,
    pub steady_state: SteadyStateConfig,
    pub observable: TerminationObservable,
    pub check_interval: usize,
}

impl TerminationConfig {
    #[inline]
    pub fn disabled() -> Self {
        Self {
            monoculture: false,
            survivor_tolerance: None,
            steady_state: SteadyStateConfig::Off,
            observable: TerminationObservable::GlobalState,
            check_interval: 1,
        }
    }

    #[inline]
    pub fn monoculture_only(check_interval: usize) -> Self {
        Self {
            monoculture: true,
            survivor_tolerance: None,
            steady_state: SteadyStateConfig::Off,
            observable: TerminationObservable::GlobalState,
            check_interval,
        }
    }

    #[inline]
    pub fn is_disabled(&self) -> bool {
        !self.monoculture && matches!(self.steady_state, SteadyStateConfig::Off)
    }
}

/// Why a solve call stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum TerminationReason {
    MaxSteps,
    Monoculture {
        surviving_index: Option<usize>,
        step: usize,
    },
    FixedPoint {
        observable: TerminationObservable,
        window: usize,
        step: usize,
    },
    OscillatorySteadyState {
        observable: TerminationObservable,
        period: usize,
        step: usize,
    },
}

impl TerminationReason {
    #[inline]
    pub fn is_terminal(&self) -> bool {
        !matches!(self, Self::MaxSteps)
    }
}

/// Solver result including early-termination metadata.
#[derive(Clone)]
pub struct SolveOutcome {
    pub final_state: SystemState<f64>,
    pub steps_run: usize,
    pub reason: TerminationReason,
}

/// Stateful bounded-history termination checker.
pub struct TerminationChecker {
    config: TerminationConfig,
    history: VecDeque<Vec<f64>>,
    stable_fixed_checks: usize,
}

impl TerminationChecker {
    pub fn new(config: TerminationConfig) -> std::io::Result<Option<Self>> {
        if config.is_disabled() {
            return Ok(None);
        }
        if config.check_interval == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "termination check_interval must be >= 1",
            ));
        }

        validate_steady_state_config(config.steady_state)?;

        Ok(Some(Self {
            config,
            history: VecDeque::with_capacity(history_capacity(config.steady_state)),
            stable_fixed_checks: 0,
        }))
    }

    #[inline]
    pub fn should_check(&self, step: usize) -> bool {
        step % self.config.check_interval == 0
    }

    pub fn check(&mut self, gs: &SystemState<f64>, step: usize) -> Option<TerminationReason> {
        if !self.should_check(step) {
            return None;
        }

        if self.config.monoculture {
            if let Some(reason) = monoculture_reason(gs, self.config.survivor_tolerance, step) {
                return Some(reason);
            }
        }

        match self.config.steady_state {
            SteadyStateConfig::Off => None,
            SteadyStateConfig::Adaptive {
                fixed_point,
                oscillation,
            } => {
                let sample = observable_sample(gs, self.config.observable)?;
                self.push_sample(sample, history_capacity(self.config.steady_state));

                if step >= fixed_point.min_steps {
                    if let Some(window) = fixed_point_window(&self.history, fixed_point) {
                        self.stable_fixed_checks += 1;
                        if self.stable_fixed_checks >= fixed_point.stable_checks_required {
                            return Some(TerminationReason::FixedPoint {
                                observable: self.config.observable,
                                window,
                                step,
                            });
                        }
                    } else {
                        self.stable_fixed_checks = 0;
                    }
                }

                if let Some(oscillation) = oscillation {
                    if step >= oscillation.min_steps {
                        if let Some(period) = oscillation_period(&self.history, oscillation) {
                            return Some(TerminationReason::OscillatorySteadyState {
                                observable: self.config.observable,
                                period,
                                step,
                            });
                        }
                    }
                }

                None
            }
        }
    }

    fn push_sample(&mut self, sample: Vec<f64>, capacity: usize) {
        while self.history.len() >= capacity {
            self.history.pop_front();
        }
        self.history.push_back(sample);
    }
}

fn validate_steady_state_config(config: SteadyStateConfig) -> std::io::Result<()> {
    let invalid = |message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message);

    match config {
        SteadyStateConfig::Off => Ok(()),
        SteadyStateConfig::Adaptive {
            fixed_point,
            oscillation,
        } => {
            if fixed_point.tolerance < 0.0 {
                return Err(invalid("fixed-point tolerance must be >= 0"));
            }
            if fixed_point.min_window == 0 {
                return Err(invalid("fixed-point min_window must be >= 1"));
            }
            if fixed_point.max_window < fixed_point.min_window {
                return Err(invalid("fixed-point max_window must be >= min_window"));
            }
            if fixed_point.stable_checks_required == 0 {
                return Err(invalid("fixed-point stable_checks_required must be >= 1"));
            }

            if let Some(oscillation) = oscillation {
                if oscillation.tolerance < 0.0 {
                    return Err(invalid("oscillation tolerance must be >= 0"));
                }
                if oscillation.min_period == 0 {
                    return Err(invalid("oscillation min_period must be >= 1"));
                }
                if oscillation.max_period < oscillation.min_period {
                    return Err(invalid("oscillation max_period must be >= min_period"));
                }
                if oscillation.repeats_required == 0 {
                    return Err(invalid("oscillation repeats_required must be >= 1"));
                }
            }

            Ok(())
        }
    }
}

fn history_capacity(config: SteadyStateConfig) -> usize {
    match config {
        SteadyStateConfig::Off => 0,
        SteadyStateConfig::Adaptive {
            fixed_point,
            oscillation,
        } => {
            let fixed = fixed_point.max_window + 1;
            let oscillation = oscillation
                .map(|cfg| cfg.max_period * (cfg.repeats_required + 1) + 1)
                .unwrap_or(0);
            fixed.max(oscillation).max(1)
        }
    }
}

fn survivor_tolerance(gs: &SystemState<f64>, configured: Option<f64>) -> f64 {
    if let Some(tolerance) = configured {
        return tolerance.max(0.0);
    }

    match gs.mode {
        Mode::Frequency { cutoff } | Mode::Population { cutoff, .. } => cutoff.unwrap_or(0.0),
    }
}

fn monoculture_reason(
    gs: &SystemState<f64>,
    configured_tolerance: Option<f64>,
    step: usize,
) -> Option<TerminationReason> {
    let tolerance = survivor_tolerance(gs, configured_tolerance);
    let mut surviving_index = None;
    let mut survivors = 0usize;

    for (idx, value) in gs.state.iter().copied().enumerate() {
        if value > tolerance {
            survivors += 1;
            if survivors == 1 {
                surviving_index = Some(idx);
            }
            if survivors > 1 {
                return None;
            }
        }
    }

    Some(TerminationReason::Monoculture {
        surviving_index,
        step,
    })
}

fn observable_sample(gs: &SystemState<f64>, observable: TerminationObservable) -> Option<Vec<f64>> {
    match observable {
        TerminationObservable::GlobalState => Some(array1_to_vec(&gs.state)),
        TerminationObservable::SpatialField => gs.space.as_ref().map(arrayd_to_vec),
    }
}

fn array1_to_vec(array: &Array1<f64>) -> Vec<f64> {
    if let Some(slice) = array.as_slice_memory_order() {
        slice.to_vec()
    } else {
        array.iter().copied().collect()
    }
}

fn arrayd_to_vec(array: &ArrayD<f64>) -> Vec<f64> {
    if let Some(slice) = array.as_slice_memory_order() {
        slice.to_vec()
    } else {
        array.iter().copied().collect()
    }
}

fn fixed_point_window(
    history: &VecDeque<Vec<f64>>,
    config: AdaptiveFixedPointConfig,
) -> Option<usize> {
    let available = history.len().saturating_sub(1);
    if available < config.min_window {
        return None;
    }

    let newest = history.back()?;
    let mut window = config.min_window;
    while window <= config.max_window && window <= available {
        let previous = history.get(history.len() - 1 - window)?;
        if linf_distance(newest, previous) <= config.tolerance {
            return Some(window);
        }

        window = window.saturating_mul(2);
        if window == 0 {
            break;
        }
    }

    None
}

fn oscillation_period(
    history: &VecDeque<Vec<f64>>,
    config: AdaptiveOscillationConfig,
) -> Option<usize> {
    let min_required = config.min_period * (config.repeats_required + 1) + 1;
    if history.len() < min_required {
        return None;
    }

    for period in config.min_period..=config.max_period {
        let needed = period * (config.repeats_required + 1) + 1;
        if history.len() < needed {
            continue;
        }

        let last_idx = history.len() - 1;
        let mut repeats_match = true;
        for repeat in 0..config.repeats_required {
            let a_start = last_idx - repeat * period;
            let b_start = a_start - period;

            for offset in 0..period {
                let a = &history[a_start - offset];
                let b = &history[b_start - offset];
                if linf_distance(a, b) > config.tolerance {
                    repeats_match = false;
                    break;
                }
            }

            if !repeats_match {
                break;
            }
        }

        if repeats_match {
            return Some(period);
        }
    }

    None
}

fn linf_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut max_delta = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let delta = (x - y).abs();
        if delta > max_delta {
            max_delta = delta;
        }
    }
    max_delta
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn monoculture_stops_after_one_survivor() {
        let gs = SystemState::from_arrays(
            Mode::Frequency { cutoff: Some(1e-9) },
            0,
            array![1.0, 0.0, 0.0],
            None,
        );
        let reason = monoculture_reason(&gs, None, 10);
        assert_eq!(
            reason,
            Some(TerminationReason::Monoculture {
                surviving_index: Some(0),
                step: 10
            })
        );
    }

    #[test]
    fn fixed_point_uses_bounded_adaptive_windows() {
        let config = TerminationConfig {
            monoculture: false,
            survivor_tolerance: None,
            steady_state: SteadyStateConfig::Adaptive {
                fixed_point: AdaptiveFixedPointConfig {
                    tolerance: 1e-6,
                    min_steps: 0,
                    min_window: 2,
                    max_window: 8,
                    stable_checks_required: 2,
                },
                oscillation: None,
            },
            observable: TerminationObservable::GlobalState,
            check_interval: 1,
        };
        let mut checker = TerminationChecker::new(config).unwrap().unwrap();
        let mut reason = None;

        for step in 1..=6 {
            let gs = SystemState::from_arrays(
                Mode::Frequency { cutoff: None },
                step,
                array![0.4, 0.6],
                None,
            );
            reason = checker.check(&gs, step);
        }

        assert!(matches!(
            reason,
            Some(TerminationReason::FixedPoint { window: 2, .. })
        ));
    }
}
