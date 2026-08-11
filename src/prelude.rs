//! Minimal imports for ordinary GLV users.
//!
//! Select one built-in template and point [`run`] at its conventional Workflow
//! `config` directory. Model assembly, task expansion, progress, recording,
//! verification, and output-scope creation are handled by the crate.
//!
//! ```no_run
//! use general_lotka_volterra_rs::prelude::*;
//!
//! let execution = run(
//!     GlvTemplate::MeanFieldReplicator,
//!     "examples/mean_field_replicator/config",
//! )?;
//! println!("{}", execution.directory().display());
//!
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub use crate::{GlvTemplate, run};
