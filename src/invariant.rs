//! Swappable abundance-domain invariant policies.
//!
//! Policies synchronize aggregate abundance, optional spatial abundance, and
//! total abundance after deterministic and stochastic updates.

pub mod core;

pub use core::{InvariantError, InvariantPolicy, enforce_state, validate_state};
