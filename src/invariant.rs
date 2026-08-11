//! Swappable abundance-domain invariant policies.
//!
//! Policies synchronize aggregate abundance, optional spatial abundance, and
//! total abundance after deterministic and stochastic updates.

pub mod core;
mod policies;

pub use core::{
    INVARIANT_TOLERANCE, InvariantError, InvariantPolicy, InvariantPolicyError, enforce_state,
    validate_state,
};
pub use policies::{FrequencyInvariant, LocalFrequencyInvariant, PopulationInvariant};
