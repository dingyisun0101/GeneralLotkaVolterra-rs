//! Spatial General Lotka–Volterra population project.

use std::error::Error;
use std::path::PathBuf;

use general_lotka_volterra_rs::prelude::*;
use scientific_workflow::prelude::runtime::{Phase, WorkflowRuntime};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let workload_directory = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let template = GlvTemplate::SpatialGeneralLotkaVolterra;
    let workload = GlvWorkload::load(workload_directory, template)?;
    let execution = workload.execution().clone();
    let simulation = workload
        .register(Phase::builder(1, "spatial general Lotka-Volterra"))
        .display_tasks_by(template.as_str(), ["/cutoff"])
        .max_concurrent_workloads(1)
        .queue_capacity(1)
        .build()?;
    WorkflowRuntime::builder()
        .phase(simulation)
        .build()?
        .run_phases([1])?;
    println!("results: {}", execution.directory().display());
    Ok(())
}
