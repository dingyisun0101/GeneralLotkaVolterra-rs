//! Shared composition contract for stochastic updates.

use std::error::Error;
use std::fmt;

use scientific_workflow::system_state::{StateError, SystemState};

use crate::{ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance};

/// One stochastic algorithm plugged into [`Noise`].
///
/// Implementations own RNG and reusable sampling scratch. Total synchronization
/// and final invariant enforcement remain engine responsibilities.
pub trait NoiseAlgorithm {
    /// Algorithm-specific validation or stochastic-update failure.
    type Error: Error + Send + Sync + 'static;

    /// Validates a state domain before evolution begins.
    fn validate(
        &self,
        abundance: &AggregateAbundance,
        space: &SpatialAbundance,
    ) -> Result<(), Self::Error>;

    /// Applies one stochastic update without enforcing final invariants.
    fn apply(
        &mut self,
        abundance: &mut AggregateAbundance,
        space: &mut SpatialAbundance,
        physical_time_increment: f64,
    ) -> Result<(), Self::Error>;
}

/// Shared stochastic component composed with one noise algorithm.
#[derive(Debug)]
pub struct Noise<N> {
    algorithm: N,
}

impl<N> Noise<N> {
    /// Creates a stochastic component from one algorithm.
    pub const fn new(algorithm: N) -> Self {
        Self { algorithm }
    }

    /// Borrows the algorithm, RNG, and scratch immutably.
    pub const fn algorithm(&self) -> &N {
        &self.algorithm
    }

    /// Borrows the algorithm, RNG, and scratch mutably.
    pub const fn algorithm_mut(&mut self) -> &mut N {
        &mut self.algorithm
    }

    /// Returns the algorithm by ownership transfer.
    pub fn into_algorithm(self) -> N {
        self.algorithm
    }
}

impl<N> Noise<N>
where
    N: NoiseAlgorithm,
{
    /// Validates canonical state payloads without mutating them.
    pub fn validate_state(&self, state: &SystemState) -> Result<(), NoiseStepError<N::Error>> {
        let (abundance, space) = state
            .borrow_payloads::<(AggregateAbundance, SpatialAbundance)>((
                ABUNDANCE_FIELD,
                SPACE_FIELD,
            ))
            .map_err(NoiseStepError::State)?;
        self.algorithm
            .validate(abundance, space)
            .map_err(NoiseStepError::Algorithm)
    }

    /// Applies one stochastic update without advancing state time.
    pub fn apply(
        &mut self,
        state: &mut SystemState,
        physical_time_increment: f64,
    ) -> Result<(), NoiseStepError<N::Error>> {
        if !physical_time_increment.is_finite() || physical_time_increment < 0.0 {
            return Err(NoiseStepError::InvalidPhysicalTimeIncrement {
                value: physical_time_increment,
            });
        }
        let (abundance, space) = state
            .borrow_payloads_mut::<(AggregateAbundance, SpatialAbundance)>((
                ABUNDANCE_FIELD,
                SPACE_FIELD,
            ))
            .map_err(NoiseStepError::State)?;
        self.algorithm
            .apply(abundance, space, physical_time_increment)
            .map_err(NoiseStepError::Algorithm)
    }
}

/// Failure while validating or applying a stochastic plugin.
#[derive(Debug)]
#[non_exhaustive]
pub enum NoiseStepError<E> {
    /// Canonical Workflow payload access failed.
    State(StateError),
    /// Noise requires a finite, nonnegative physical-time increment.
    InvalidPhysicalTimeIncrement {
        /// Rejected increment.
        value: f64,
    },
    /// The selected noise algorithm rejected the state or update.
    Algorithm(E),
}

impl<E> fmt::Display for NoiseStepError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(error) => fmt::Display::fmt(error, formatter),
            Self::InvalidPhysicalTimeIncrement { value } => write!(
                formatter,
                "noise physical-time increment must be finite and nonnegative: {value}"
            ),
            Self::Algorithm(error) => write!(formatter, "noise algorithm failed: {error}"),
        }
    }
}

impl<E> Error for NoiseStepError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::State(error) => Some(error),
            Self::InvalidPhysicalTimeIncrement { .. } => None,
            Self::Algorithm(error) => Some(error),
        }
    }
}
