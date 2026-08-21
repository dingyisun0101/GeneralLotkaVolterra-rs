//! Deterministic mean-field replicator project.

use std::error::Error;
use std::path::PathBuf;

use general_lotka_volterra_rs::prelude::*;
use scientific_workflow::prelude::runtime::{Phase, WorkflowRuntime};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let workload_directory = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let template = GlvTemplate::MeanFieldReplicator;
    let workload = GlvWorkload::load(workload_directory, template)?;
    let simulation = workload
        .register(Phase::builder(1, "mean-field replicator"))
        .display_tasks_by(template.as_str(), ["/cutoff"])
        .build()?;
    WorkflowRuntime::builder(workload.execution_record_path())
        .phase(simulation)
        .build()?
        .run_phases([1])?;
    println!("results: {}", workload.execution().directory().display());
    Ok(())
}
