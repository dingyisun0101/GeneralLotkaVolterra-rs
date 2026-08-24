//! Minimal imports for registering built-in GLV tasks with a Study.
//!
//! GLV owns model-specific workload loading, construction, recording, and
//! validation. Applications own the Workflow execution scope, phase
//! composition, scheduling, and study execution.
//!
//! ```no_run
//! use general_lotka_volterra_rs::prelude::*;
//!
//! use scientific_workflow::prelude::basics::ExecutionScope;
//! use scientific_workflow::prelude::study::{Phase, Study};
//!
//! let template = GlvTemplate::MeanFieldReplicator;
//! let execution = ExecutionScope::open_or_create("output/replicate_0")?;
//! let workload = GlvWorkload::load(
//!     "examples/mean_field_replicator",
//!     template,
//!     execution,
//! )?;
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

pub use crate::{
    ComputePool, ComputePoolError, GlvInputs, GlvTemplate, GlvWorkload, GlvWorkloadError,
    with_threads,
};
