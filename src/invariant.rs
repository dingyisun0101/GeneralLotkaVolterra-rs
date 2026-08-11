//! Swappable abundance-domain invariant policies.
//!
//! Policies synchronize aggregate abundance, optional spatial abundance, and
//! total abundance after deterministic and stochastic updates.

pub mod core;
pub mod frequency;
pub mod local_frequency;
pub mod population;

pub use core::{
    INVARIANT_TOLERANCE, InvariantError, InvariantPolicy, InvariantPolicyError, enforce_state,
    validate_state,
};
pub use frequency::FrequencyInvariant;
pub use local_frequency::LocalFrequencyInvariant;
pub use population::PopulationInvariant;
