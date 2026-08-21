//! Minimal imports for registering built-in GLV tasks with a Workflow runtime.
//!
//! GLV owns workload loading, model construction, recording, and validation.
//! Applications own phase composition, scheduling, and runtime execution.
//!
//! ```no_run
//! use general_lotka_volterra_rs::prelude::*;
//!
//! use scientific_workflow::prelude::runtime::{Phase, WorkflowRuntime};
//!
//! let template = GlvTemplate::MeanFieldReplicator;
//! let workload = GlvWorkload::load("examples/mean_field_replicator", template)?;
//! let simulation = workload
//!     .register(Phase::builder(1, "GLV simulation"))
//!     .build()?;
//! WorkflowRuntime::builder(workload.execution_record_path())
//!     .phase(simulation)
//!     .build()?
//!     .run_phases([1])?;
//!
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub use crate::project::load_glv_project;
pub use crate::{GlvTemplate, GlvWorkload, GlvWorkloadError};
