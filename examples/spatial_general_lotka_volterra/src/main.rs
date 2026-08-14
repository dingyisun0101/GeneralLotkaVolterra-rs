//! Spatial General Lotka–Volterra population project.

use std::error::Error;
use std::path::{Path, PathBuf};

use general_lotka_volterra_rs::prelude::*;
use scientific_workflow::prelude::basics::ExecutionScope;
use scientific_workflow::prelude::runtime::{Phase, WorkflowRuntime};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let config = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config"));
    let project_root = config
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let project = load_glv_project(project_root)?;
    let execution = ExecutionScope::create_generated(project.resolve_path("recordings")?)?;
    let task_execution = execution.clone();
    let template = GlvTemplate::SpatialGeneralLotkaVolterra;
    let simulation = Phase::builder(1, "spatial general Lotka-Volterra")
        .progress_tasks_from_project(&project, template.as_str(), move |context| {
            template.run_task(&task_execution, context)
        })
        .display_tasks_by(template.as_str(), ["cutoff"])
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
