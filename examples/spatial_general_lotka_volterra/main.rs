//! Sequential Workflow-native spatial General Lotka–Volterra example.

#[path = "../support.rs"]
mod support;

use std::error::Error;

use general_lotka_volterra_rs::kernel::{
    Boundary, Diffusion, InteractionSource, JsonInteractionSource, persist_interaction_matrix,
};
use general_lotka_volterra_rs::recording::{GlvRecording, GlvRecordingMetadata, TerminationReason};
use general_lotka_volterra_rs::{
    SpatialGeneralLotkaVolterra, SpatialGeneralLotkaVolterraConfig, TimeStep,
};
use ndarray::{Array1, ArrayD, IxDyn};
use scientific_workflow::prelude::{ExecutionScope, ProgressReporter, TaskConfig};
use serde::Deserialize;

use support::{RecordingInputs, load_project, recording_config, validate_completed_recording};

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum BoundaryInput {
    Periodic,
    Neumann,
}

impl From<BoundaryInput> for Boundary {
    fn from(value: BoundaryInput) -> Self {
        match value {
            BoundaryInput::Periodic => Self::Periodic,
            BoundaryInput::Neumann => Self::Neumann,
        }
    }
}

struct TaskInputs {
    spatial_shape: Vec<usize>,
    initial_cell: Vec<f64>,
    growth: Array1<f64>,
    diffusion: Array1<f64>,
    spacing: Vec<f64>,
    boundary: Boundary,
    cutoff: f64,
    carrying_capacity: Option<f64>,
    physical_time_increment: f64,
    maximum_iterations: u64,
    recording: RecordingInputs,
}

impl TaskInputs {
    fn decode(task: &TaskConfig) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            spatial_shape: task.decode_value("spatial_shape")?,
            initial_cell: task.decode_value("initial_cell")?,
            growth: Array1::from_vec(task.decode_value("growth")?),
            diffusion: Array1::from_vec(task.decode_value("diffusion")?),
            spacing: task.decode_value("spacing")?,
            boundary: task.decode_value::<BoundaryInput>("boundary")?.into(),
            cutoff: task.decode_value("cutoff")?,
            carrying_capacity: task.decode_value("carrying_capacity")?,
            physical_time_increment: task.decode_value("physical_time_increment")?,
            maximum_iterations: task.decode_value("maximum_iterations")?,
            recording: task.decode_value("recording")?,
        })
    }

    fn initial_space(&self) -> Result<ArrayD<f64>, ndarray::ShapeError> {
        let cells = self.spatial_shape.iter().product::<usize>();
        let mut shape = self.spatial_shape.clone();
        shape.push(self.initial_cell.len());
        ArrayD::from_shape_vec(IxDyn(&shape), self.initial_cell.repeat(cells))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let project = load_project("spatial_general_lotka_volterra")?;
    let scope = ExecutionScope::create_generated(project.resolve_path("recordings")?)?;
    let reporter = ProgressReporter::for_project(&project).start()?;
    reporter.report(format!("execution scope: {}", scope.directory().display()))?;
    for task in project.task_configs() {
        run_task(&scope, &reporter, task)?;
    }
    reporter.complete(format!(
        "spatial General Lotka–Volterra execution complete: {}",
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

    let species = inputs.initial_cell.len();
    let interaction =
        JsonInteractionSource::resolved_file(task.resolve_path("interaction_matrix")?)
            .resolve(species)?;
    let persisted = persist_interaction_matrix(scope, &interaction)?;
    let mut shape = inputs.spatial_shape.clone();
    shape.push(species);
    let initial_space = inputs.initial_space()?;
    let diffusion = Diffusion::new(inputs.diffusion, inputs.spacing, inputs.boundary)?;
    let config = SpatialGeneralLotkaVolterraConfig::new(
        shape,
        inputs.growth,
        diffusion,
        inputs.cutoff,
        inputs.carrying_capacity,
        TimeStep::new(inputs.physical_time_increment)?,
    );
    let mut simulation = SpatialGeneralLotkaVolterra::new(initial_space, interaction, config)?;
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
