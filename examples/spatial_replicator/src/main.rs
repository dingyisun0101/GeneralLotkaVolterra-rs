//! Spatial local-frequency replicator workflow.

use std::error::Error;
use std::path::PathBuf;

use general_lotka_volterra_rs::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let project = load_glv_project(project_root)?;
    let scope = ExecutionScope::create_generated(project.resolve_path("recordings")?)?;
    let reporter = ProgressReporter::for_project(&project).start()?;
    reporter.report(format!("execution scope: {}", scope.directory().display()))?;

    for task in project.task_configs() {
        run_task(&scope, &reporter, task)?;
    }

    reporter.complete(format!(
        "spatial replicator execution complete: {}",
        scope.directory().display()
    ))?;
    Ok(())
}

fn run_task(
    scope: &ExecutionScope,
    reporter: &ProgressReporter,
    task: TaskConfig,
) -> Result<(), Box<dyn Error>> {
    // Both Workflow values and GLV domain enums decode directly from JSON.
    let spatial_shape: Vec<usize> = task.decode_value("spatial_shape")?;
    let initial_cell: Vec<f64> = task.decode_value("initial_cell")?;
    let growth = Array1::from_vec(task.decode_value("growth")?);
    let diffusion_coefficients = Array1::from_vec(task.decode_value("diffusion")?);
    let spacing = task.decode_value("spacing")?;
    let boundary: Boundary = task.decode_value("boundary")?;
    let cutoff = task.decode_value("cutoff")?;
    let time_step = TimeStep::new(task.decode_value("physical_time_increment")?)?;
    let maximum_iterations = task.decode_value("maximum_iterations")?;
    let recording_config: GlvRecordingConfig = task.decode_value("recording")?;

    let progress = reporter.start_task(&task, 0, Some(maximum_iterations))?;
    progress.set_phase("resolving interaction matrix");
    let species = initial_cell.len();
    let interaction =
        JsonInteractionSource::resolved_file(task.resolve_path("interaction_matrix")?)
            .resolve(species)?;
    let persisted = persist_interaction_matrix(scope, &interaction)?;

    // Spatial payloads are standard-contiguous and species-last. Repeating one
    // cell initializes every grid point with the same local composition.
    let cells = spatial_shape.iter().product::<usize>();
    let mut shape = spatial_shape;
    shape.push(species);
    let initial_space = ArrayD::from_shape_vec(IxDyn(&shape), initial_cell.repeat(cells))?;
    let diffusion = Diffusion::new(diffusion_coefficients, spacing, boundary)?;
    let config = SpatialReplicatorConfig::new(shape, growth, diffusion, cutoff, time_step);
    let mut simulation = SpatialReplicator::new(initial_space, interaction, config)?;

    let metadata = GlvRecordingMetadata::new(
        simulation.kind(),
        simulation.abundance_representation(),
        task.parameters(),
        persisted.descriptor(),
        simulation.rng_record(),
    )?;
    let recording_directory = scope.task_recording_directory(task.task_ordinal());
    let mut recording = GlvRecording::start(
        &recording_directory,
        recording_config,
        metadata,
        simulation.state(),
    )?;

    progress.set_phase("evolving");
    while simulation.state().simulation_time().iteration() < maximum_iterations {
        let time = simulation.step()?;
        recording.observe_state(simulation.state())?;
        progress.set_iteration(time.iteration())?;
    }

    progress.set_phase("validating recording");
    recording.complete(simulation.state(), TerminationReason::MaximumIterations)?;
    verify_completed_glv_checkpoint(recording_directory, simulation.state())?;
    progress.complete()?;
    Ok(())
}
