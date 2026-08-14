//! Minimal imports for registering built-in GLV tasks with a Workflow runtime.
//!
//! Applications own project loading, execution scopes, phases, scheduling, and
//! runtime execution. GLV owns model construction, evolution, recording, and
//! validation inside [`GlvTemplate::run_task`].
//!
//! ```no_run
//! use general_lotka_volterra_rs::prelude::*;
//!
//! use scientific_workflow::prelude::basics::ExecutionScope;
//! use scientific_workflow::prelude::runtime::{Phase, WorkflowRuntime};
//!
//! let project = load_glv_project("examples/mean_field_replicator")?;
//! let execution = ExecutionScope::create_generated(project.resolve_path("recordings")?)?;
//! let task_execution = execution.clone();
//! let template = GlvTemplate::MeanFieldReplicator;
//! let simulation = Phase::builder(1, "GLV simulation")
//!     .progress_tasks_from_project(&project, template.as_str(), move |context| {
//!         template.run_task(&task_execution, context)
//!     })
//!     .build()?;
//! WorkflowRuntime::builder()
//!     .phase(simulation)
//!     .build()?
//!     .run_phases([1])?;
//!
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub use crate::GlvTemplate;
pub use crate::project::load_glv_project;
