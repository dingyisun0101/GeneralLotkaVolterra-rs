//! Spatial local-frequency replicator inputs.

use std::error::Error;
use std::path::PathBuf;

use general_lotka_volterra_rs::prelude::*;
use scientific_workflow::prelude::basics::{ProjectPaths, ReplicateExecutor, StudySettings};
use scientific_workflow::prelude::study::{Phase, Study};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let workload_directory = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let settings = StudySettings::load(&workload_directory)?;
    let output_root = ProjectPaths::load(&workload_directory)?.resolve_path("recordings")?;
    let Some(replicate) =
        ReplicateExecutor::new(settings.replicate_settings(), output_root)
            .dispatch_current_executable()?
    else {
        return Ok(());
    };
    let template = GlvTemplate::SpatialReplicator;
    let workload = GlvWorkload::load(
        workload_directory,
        template,
        replicate.execution_scope().clone(),
    )?;
    let simulation = workload
        .register(Phase::builder(1, "spatial replicator"))
        .build()?;
    Study::builder(workload.record_path())
        .phase(simulation)
        .build()?
        .run_phases([1])?;
    println!("results: {}", workload.execution().directory().display());
    Ok(())
}
