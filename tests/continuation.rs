use std::fs;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use general_lotka_volterra_rs::interaction::{
    InteractionMatrix, load_verified_interaction_matrix, persist_interaction_matrix,
};
use general_lotka_volterra_rs::reading::open_completed_glv_recording;
use general_lotka_volterra_rs::recording::{GlvRecording, GlvRecordingMetadata, TerminationReason};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AggregateAbundance, CHECKPOINT_STREAM, MeanFieldReplicator,
    MeanFieldReplicatorConfig, SIGNAL_STREAM, SPACE_FIELD, SPACE_STREAM, SpatialAbundance,
    TOTAL_FIELD, TimeStep, TotalAbundance,
};
use ndarray::{Array1, arr2};
use scientific_workflow::configuration::{ParameterSpace, TaskParameters};
use scientific_workflow::execution::ExecutionScope;
use scientific_workflow::storage::{SamplingInterval, StateStreamConfig};
use scientific_workflow::system_state::SystemState;
use scientific_workflow::time_series::StateSeries;

static WORKSPACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Workspace {
    root: PathBuf,
}

impl Workspace {
    fn new() -> Self {
        let sequence = WORKSPACE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "glv-continuation-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        Self { root }
    }

    fn task_parameters(&self) -> TaskParameters {
        let config = self.root.join("config");
        fs::create_dir(&config).unwrap();
        fs::write(
            config.join("fixed.json"),
            r#"{"growth":[0.15,-0.08],"cutoff":0.0,"physical_time_increment":0.05}"#,
        )
        .unwrap();
        fs::write(
            config.join("sweep.json"),
            r#"{"mode":"cartesian","axes":[]}"#,
        )
        .unwrap();
        ParameterSpace::load(config).unwrap().task(0).unwrap()
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            eprintln!("failed to clean {}: {error}", self.root.display());
        }
    }
}

fn stream(name: &str, fields: &[&str], interval: u64) -> StateStreamConfig {
    StateStreamConfig::new(
        name,
        fields.iter().copied(),
        SamplingInterval::iterations(interval).unwrap(),
        Some((
            NonZeroU64::new(1_024).unwrap(),
            NonZeroU64::new(8_192).unwrap(),
        )),
    )
}

fn recording_config() -> Vec<StateStreamConfig> {
    vec![
        stream(SIGNAL_STREAM, &[ABUNDANCE_FIELD, TOTAL_FIELD], 2),
        stream(
            SPACE_STREAM,
            &[ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD],
            3,
        ),
        stream(
            CHECKPOINT_STREAM,
            &[ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD],
            1,
        ),
    ]
}

fn simulation_config() -> MeanFieldReplicatorConfig {
    MeanFieldReplicatorConfig::new(
        Array1::from_vec(vec![0.15, -0.08]),
        0.0,
        TimeStep::new(0.05).unwrap(),
    )
}

fn interaction() -> InteractionMatrix {
    InteractionMatrix::from_array(arr2(&[[-0.2, 0.7], [-0.4, 0.1]]), 2).unwrap()
}

fn advance(
    simulation: &mut MeanFieldReplicator,
    recording: &mut GlvRecording,
    final_iteration: u64,
) {
    while simulation.state().simulation_time().iteration() < final_iteration {
        simulation.step().unwrap();
        recording.observe_state(simulation.state()).unwrap();
    }
}

fn assert_state_equal(actual: &SystemState, expected: &SystemState, compare_space: bool) {
    assert_eq!(actual.simulation_time(), expected.simulation_time());
    assert_eq!(
        actual
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        expected
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap()
    );
    if compare_space {
        assert_eq!(
            actual.payload::<SpatialAbundance>(SPACE_FIELD).unwrap(),
            expected.payload::<SpatialAbundance>(SPACE_FIELD).unwrap()
        );
    }
    assert_eq!(
        actual.payload::<TotalAbundance>(TOTAL_FIELD).unwrap(),
        expected.payload::<TotalAbundance>(TOTAL_FIELD).unwrap()
    );
}

fn assert_series_equal(actual: &StateSeries, expected: &StateSeries, compare_space: bool) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_state_equal(actual, expected, compare_space);
    }
}

#[test]
fn deterministic_continuation_matches_uninterrupted_state_and_samples() {
    let workspace = Workspace::new();
    let task = workspace.task_parameters();

    let uninterrupted_scope =
        ExecutionScope::create_named(&workspace.root, "uninterrupted").unwrap();
    let uninterrupted_interaction = interaction();
    let uninterrupted_artifact =
        persist_interaction_matrix(&uninterrupted_scope, &uninterrupted_interaction).unwrap();
    let uninterrupted_metadata = GlvRecordingMetadata::new(
        general_lotka_volterra_rs::simulation::SimulationKind::MeanFieldReplicator,
        general_lotka_volterra_rs::AbundanceRepresentation::RelativeFrequency,
        &task,
        uninterrupted_artifact.descriptor(),
        None,
    )
    .unwrap();
    let uninterrupted_directory = uninterrupted_scope.task_recording_directory(0);
    let mut uninterrupted = MeanFieldReplicator::new(
        Array1::from_vec(vec![0.35, 0.65]),
        uninterrupted_interaction,
        simulation_config(),
    )
    .unwrap();
    let mut uninterrupted_recording = GlvRecording::start(
        &uninterrupted_directory,
        recording_config(),
        uninterrupted_metadata,
        uninterrupted.state(),
    )
    .unwrap();
    advance(&mut uninterrupted, &mut uninterrupted_recording, 6);
    let uninterrupted_completed = uninterrupted_recording
        .complete(uninterrupted.state(), TerminationReason::MaximumIterations)
        .unwrap();
    assert_eq!(uninterrupted_completed.timing().continuation_count(), 0);

    let resumed_scope = ExecutionScope::create_named(&workspace.root, "resumed").unwrap();
    let resumed_interaction = interaction();
    let resumed_artifact =
        persist_interaction_matrix(&resumed_scope, &resumed_interaction).unwrap();
    let resumed_metadata = GlvRecordingMetadata::new(
        general_lotka_volterra_rs::simulation::SimulationKind::MeanFieldReplicator,
        general_lotka_volterra_rs::AbundanceRepresentation::RelativeFrequency,
        &task,
        resumed_artifact.descriptor(),
        None,
    )
    .unwrap();
    let resumed_directory = resumed_scope.task_recording_directory(0);
    let mut resumed = MeanFieldReplicator::new(
        Array1::from_vec(vec![0.35, 0.65]),
        resumed_interaction,
        simulation_config(),
    )
    .unwrap();
    let mut resumed_recording = GlvRecording::start(
        &resumed_directory,
        recording_config(),
        resumed_metadata.clone(),
        resumed.state(),
    )
    .unwrap();
    advance(&mut resumed, &mut resumed_recording, 3);
    drop(resumed_recording);
    drop(resumed);

    let verified_interaction =
        load_verified_interaction_matrix(resumed_scope.directory(), resumed_artifact.descriptor())
            .unwrap();
    let (mut resumed_recording, checkpoint) = GlvRecording::continue_from_latest_checkpoint(
        &resumed_directory,
        recording_config(),
        resumed_metadata,
    )
    .unwrap();
    assert_eq!(checkpoint.simulation_time().iteration(), 3);
    let mut resumed = MeanFieldReplicator::from_state(
        checkpoint,
        general_lotka_volterra_rs::AbundanceRepresentation::RelativeFrequency,
        verified_interaction,
        simulation_config(),
    )
    .unwrap();
    advance(&mut resumed, &mut resumed_recording, 6);
    let resumed_completed = resumed_recording
        .complete(resumed.state(), TerminationReason::MaximumIterations)
        .unwrap();
    assert_eq!(resumed_completed.timing().continuation_count(), 1);
    assert_state_equal(resumed.state(), uninterrupted.state(), true);

    let uninterrupted_reader = open_completed_glv_recording(&uninterrupted_directory).unwrap();
    let resumed_reader = open_completed_glv_recording(&resumed_directory).unwrap();
    for stream in [SIGNAL_STREAM, SPACE_STREAM, CHECKPOINT_STREAM] {
        let uninterrupted_series = uninterrupted_reader
            .read_stream_as_state_series(stream)
            .unwrap();
        let resumed_series = resumed_reader.read_stream_as_state_series(stream).unwrap();
        assert_series_equal(
            &resumed_series,
            &uninterrupted_series,
            stream != SIGNAL_STREAM,
        );
    }
}
