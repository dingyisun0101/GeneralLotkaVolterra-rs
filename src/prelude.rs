//! Minimal imports for registering built-in GLV tasks with a Study.
//!
//! GLV owns workload loading, model construction, recording, and validation.
//! Applications own phase composition, scheduling, and study execution.
//!
//! ```no_run
//! use general_lotka_volterra_rs::prelude::*;
//!
//! use scientific_workflow::prelude::study::{Phase, Study};
//!
//! let template = GlvTemplate::MeanFieldReplicator;
//! let workload = GlvWorkload::load("examples/mean_field_replicator", template)?;
//! let simulation = workload
//!     .register(Phase::builder(1, "GLV simulation"))
//!     .build()?;
//! Study::builder(workload.record_path())
//!     .phase(simulation)
//!     .build()?
//!     .run_phases([1])?;
//!
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub use crate::{GlvInputs, GlvTemplate, GlvWorkload, GlvWorkloadError, load_glv_inputs};
