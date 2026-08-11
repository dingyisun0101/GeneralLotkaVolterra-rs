//! Shared contract for abundance-domain invariants.

use std::error::Error;
use std::fmt;

use scientific_workflow::system_state::{StateError, SystemState};

use crate::{
    ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, TotalAbundance,
};

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
