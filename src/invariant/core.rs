//! Shared contract for abundance-domain invariants.

use std::error::Error;
use std::fmt;

use scientific_workflow::system_state::{StateError, SystemState};
use thiserror::Error as ThisError;

use crate::{
    ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, TotalAbundance,
};

/// Absolute and relative tolerance used for invariant consistency checks.
pub const INVARIANT_TOLERANCE: f64 = 1.0e-12;

/// One policy that validates and restores a scientific abundance domain.
///
/// Implementations receive all three canonical payloads together so aggregate,
/// spatial, and total values can be synchronized within one checked Workflow
/// tuple borrow. Policies never advance simulation time.
pub trait InvariantPolicy {
    /// Policy-specific validation or enforcement failure.
    type Error: Error + Send + Sync + 'static;

    /// Validates a complete state domain before evolution begins.
    fn validate(
        &self,
        abundance: &AggregateAbundance,
        space: &SpatialAbundance,
        total: &TotalAbundance,
    ) -> Result<(), Self::Error>;

    /// Restores the policy's domain and synchronizes derived payloads.
    ///
    /// Implementations must finish fallible calculations and validation before
    /// mutating payloads. Returning an error after a partial mutation violates
    /// the plugin contract.
    fn enforce(
        &mut self,
        abundance: &mut AggregateAbundance,
        space: &mut SpatialAbundance,
        total: &mut TotalAbundance,
    ) -> Result<(), Self::Error>;
}

/// Validates a complete canonical state through one invariant policy.
pub fn validate_state<I>(policy: &I, state: &SystemState) -> Result<(), InvariantError<I::Error>>
where
    I: InvariantPolicy,
{
    let (abundance, space, total) = state
        .borrow_payloads::<(AggregateAbundance, SpatialAbundance, TotalAbundance)>((
            ABUNDANCE_FIELD,
            SPACE_FIELD,
            TOTAL_FIELD,
        ))
        .map_err(InvariantError::State)?;
    policy
        .validate(abundance, space, total)
        .map_err(InvariantError::Policy)
}

/// Enforces one invariant through a coordinated mutable state borrow.
pub fn enforce_state<I>(
    policy: &mut I,
    state: &mut SystemState,
) -> Result<(), InvariantError<I::Error>>
where
    I: InvariantPolicy,
{
    let (abundance, space, total) = state
        .borrow_payloads_mut::<(AggregateAbundance, SpatialAbundance, TotalAbundance)>((
            ABUNDANCE_FIELD,
            SPACE_FIELD,
            TOTAL_FIELD,
        ))
        .map_err(InvariantError::State)?;
    policy
        .enforce(abundance, space, total)
        .map_err(InvariantError::Policy)
}

/// Failure while validating or enforcing an invariant policy.
#[derive(Debug)]
#[non_exhaustive]
pub enum InvariantError<E> {
    /// Canonical Workflow payload access failed.
    State(StateError),
    /// The selected policy rejected the state or enforcement operation.
    Policy(E),
}

impl<E> fmt::Display for InvariantError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(error) => fmt::Display::fmt(error, formatter),
            Self::Policy(error) => write!(formatter, "invariant policy failed: {error}"),
        }
    }
}

impl<E> Error for InvariantError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::State(error) => Some(error),
            Self::Policy(error) => Some(error),
        }
    }
}

/// Shared configuration, layout, and domain failures for built-in policies.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum InvariantPolicyError {
    /// Every model requires at least one species.
    #[error("invariant species dimension must be greater than zero")]
    EmptySpecies,
    /// Cutoff configuration must be finite and nonnegative.
    #[error("invariant cutoff must be finite and nonnegative, found {value}")]
    InvalidCutoff {
        /// Rejected cutoff.
        value: f64,
    },
    /// Carrying capacity must be finite and nonnegative when configured.
    #[error("carrying capacity must be finite and nonnegative, found {value}")]
    InvalidCarryingCapacity {
        /// Rejected capacity.
        value: f64,
    },
    /// Aggregate abundance disagrees with configured species count.
    #[error("abundance length {actual} does not match species count {expected}")]
    SpeciesMismatch {
        /// Configured species count.
        expected: usize,
        /// State abundance length.
        actual: usize,
    },
    /// A non-spatial policy received spatial storage.
    #[error("aggregate frequency invariant requires `space = None`")]
    UnexpectedSpace,
    /// A spatial policy received no spatial storage.
    #[error("spatial invariant requires populated `space`")]
    SpaceRequired,
    /// Spatial storage must have at least one axis.
    #[error("spatial abundance must have a species axis")]
    MissingSpeciesAxis,
    /// The species-last spatial dimension disagrees with configuration.
    #[error("space species axis has length {actual}, expected {expected}")]
    SpaceSpeciesMismatch {
        /// Configured species count.
        expected: usize,
        /// State species-axis length.
        actual: usize,
    },
    /// Spatial storage must contain at least one cell.
    #[error("spatial abundance must contain at least one cell")]
    EmptySpatialDomain,
    /// Hot-loop policies require standard row-major storage.
    #[error("spatial abundance must use standard contiguous row-major storage")]
    NonStandardSpaceLayout,
    /// A state value is not finite.
    #[error("{field} value at linear index {linear_index} is not finite: {value}")]
    NonFiniteValue {
        /// Canonical field name.
        field: &'static str,
        /// Row-major linear index.
        linear_index: usize,
        /// Rejected value.
        value: f64,
    },
    /// A state value is negative.
    #[error("{field} value at linear index {linear_index} is negative: {value}")]
    NegativeValue {
        /// Canonical field name.
        field: &'static str,
        /// Row-major linear index.
        linear_index: usize,
        /// Rejected value.
        value: f64,
    },
    /// A frequency vector or cell is not normalized.
    #[error("frequency cell {cell} has sum {sum}, expected 1")]
    SimplexViolation {
        /// Zero for aggregate frequency, otherwise spatial cell index.
        cell: usize,
        /// Observed sum.
        sum: f64,
    },
    /// Aggregate abundance is inconsistent with spatial storage.
    #[error(
        "aggregate abundance for species {species} is {actual}, expected spatial aggregate {expected}"
    )]
    AggregateMismatch {
        /// Zero-based species index.
        species: usize,
        /// Expected aggregate.
        expected: f64,
        /// Stored aggregate.
        actual: f64,
    },
    /// Total abundance is inconsistent with the policy convention.
    #[error("total abundance is {actual}, expected {expected}")]
    TotalMismatch {
        /// Policy-derived total.
        expected: f64,
        /// Stored total.
        actual: f64,
    },
    /// A state exceeds its configured global carrying capacity.
    #[error("population sum {total} exceeds carrying capacity {capacity}")]
    CarryingCapacityExceeded {
        /// Observed exact population sum.
        total: f64,
        /// Configured limit.
        capacity: f64,
    },
}

pub(crate) fn validate_species_and_cutoff(
    species: usize,
    cutoff: f64,
) -> Result<(), InvariantPolicyError> {
    if species == 0 {
        return Err(InvariantPolicyError::EmptySpecies);
    }
    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(InvariantPolicyError::InvalidCutoff { value: cutoff });
    }
    Ok(())
}

pub(crate) fn validate_abundance_values(
    abundance: &AggregateAbundance,
) -> Result<(), InvariantPolicyError> {
    validate_values(ABUNDANCE_FIELD, abundance.iter().copied())
}

pub(crate) fn validate_space_values(space: &[f64]) -> Result<(), InvariantPolicyError> {
    validate_values(SPACE_FIELD, space.iter().copied())
}

fn validate_values(
    field: &'static str,
    values: impl Iterator<Item = f64>,
) -> Result<(), InvariantPolicyError> {
    for (linear_index, value) in values.enumerate() {
        if !value.is_finite() {
            return Err(InvariantPolicyError::NonFiniteValue {
                field,
                linear_index,
                value,
            });
        }
        if value < 0.0 {
            return Err(InvariantPolicyError::NegativeValue {
                field,
                linear_index,
                value,
            });
        }
    }
    Ok(())
}

pub(crate) fn close(expected: f64, actual: f64) -> bool {
    let scale = expected.abs().max(actual.abs());
    (expected - actual).abs() <= INVARIANT_TOLERANCE * (1.0 + scale)
}
