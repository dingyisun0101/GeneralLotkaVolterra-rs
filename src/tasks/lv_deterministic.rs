/*!
Deterministic GLV task placeholder.

Purpose:
    This module reserves the deterministic GLV task entry point while the
    implemented solver stack remains replicator-form.
*/

use std::io::{Error, ErrorKind, Result};

/// Placeholder for deterministic GLV task wiring.
///
/// The current solver stack is replicator-form only. Callers should use
/// `tasks::replicator_deterministic::run` until a dedicated GLV RHS/integrator
/// is introduced.
pub fn run() -> Result<()> {
    Err(Error::new(
        ErrorKind::Unsupported,
        "lv_deterministic is not implemented; use replicator_deterministic::run",
    ))
}
