//! Shared simulation primitives used across model plugins.

use thiserror::Error;

/// A validated positive increment of modeled physical time.
///
/// Constructing this type once at the simulation boundary prevents kernels,
/// noise plugins, and engines from developing inconsistent increment checks.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct TimeStep(f64);

impl TimeStep {
    /// Validates a finite physical-time increment greater than zero.
    pub fn new(value: f64) -> Result<Self, TimeStepError> {
        if !value.is_finite() {
            return Err(TimeStepError::NonFinite { value });
        }
        if value <= 0.0 {
            return Err(TimeStepError::NonPositive { value });
        }
        Ok(Self(value))
    }

    /// Returns the validated increment.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for TimeStep {
    type Error = TimeStepError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<TimeStep> for f64 {
    fn from(value: TimeStep) -> Self {
        value.get()
    }
}

/// Rejection of an invalid physical-time increment.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TimeStepError {
    /// NaN and infinities cannot define a simulation step.
    #[error("physical-time increment must be finite, found {value}")]
    NonFinite {
        /// Rejected value.
        value: f64,
    },
    /// Forward evolution requires a strictly positive increment.
    #[error("physical-time increment must be greater than zero, found {value}")]
    NonPositive {
        /// Rejected value.
        value: f64,
    },
}
