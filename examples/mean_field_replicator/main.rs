//! Sequential Workflow-native mean-field replicator example.

#[path = "../support.rs"]
mod support;

use std::error::Error;

use general_lotka_volterra_rs::kernel::{
    InteractionSource, JsonInteractionSource, persist_interaction_matrix,
};
use general_lotka_volterra_rs::recording::{GlvRecording, GlvRecordingMetadata, TerminationReason};
use general_lotka_volterra_rs::{MeanFieldReplicator, MeanFieldReplicatorConfig, TimeStep};
use ndarray::Array1;
use scientific_workflow::prelude::{ExecutionScope, ProgressReporter, TaskConfig};

use support::{RecordingInputs, load_project, recording_config, validate_completed_recording};

struct TaskInputs {
    initial_abundance: Array1<f64>,
    growth: Array1<f64>,
    cutoff: f64,
    physical_time_increment: f64,
    maximum_iterations: u64,
    recording: RecordingInputs,
}

impl TaskInputs {
    fn decode(task: &TaskConfig) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            initial_abundance: Array1::from_vec(task.decode_value("initial_abundance")?),
            growth: Array1::from_vec(task.decode_value("growth")?),
            cutoff: task.decode_value("cutoff")?,
            physical_time_increment: task.decode_value("physical_time_increment")?,
            maximum_iterations: task.decode_value("maximum_iterations")?,
            recording: task.decode_value("recording")?,
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let project = load_project("mean_field_replicator")?;
    let scope = ExecutionScope::create_generated(project.resolve_path("recordings")?)?;
    let reporter = ProgressReporter::for_project(&project).start()?;
    reporter.report(format!("execution scope: {}", scope.directory().display()))?;

    for task in project.task_configs() {
        run_task(&scope, &reporter, task)?;
    }

    reporter.complete(format!(
        "mean-field replicator execution complete: {}",
        scope.directory().display()
    ))?;
    Ok(())
}

fn run_task(
    scope: &ExecutionScope,
    reporter: &ProgressReporter,
    task: TaskConfig,
) -> Result<(), Box<dyn Error>> {
    let inputs = TaskInputs::decode(&task)?;
    let progress = reporter.start_task(&task, 0, Some(inputs.maximum_iterations))?;
    progress.set_phase("resolving interaction matrix");

    let species = inputs.initial_abundance.len();
    let interaction =
        JsonInteractionSource::resolved_file(task.resolve_path("interaction_matrix")?)
            .resolve(species)?;
    let persisted = persist_interaction_matrix(scope, &interaction)?;

    progress.set_phase("constructing simulation");
    let config = MeanFieldReplicatorConfig::new(
        inputs.growth,
        inputs.cutoff,
        TimeStep::new(inputs.physical_time_increment)?,
    );
    let mut simulation = MeanFieldReplicator::new(inputs.initial_abundance, interaction, config)?;
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
        recording_config(inputs.recording)?,
        metadata,
        simulation.state(),
    )?;

    progress.set_phase("evolving");
    while simulation.state().simulation_time().iteration() < inputs.maximum_iterations {
        let time = simulation.step()?;
        recording.observe_state(simulation.state())?;
        progress.set_iteration(time.iteration())?;
    }

    progress.set_phase("validating recording");
    recording.complete(simulation.state(), TerminationReason::MaximumIterations)?;
    validate_completed_recording(recording_directory, simulation.state())?;
    progress.complete()?;
    Ok(())
}
