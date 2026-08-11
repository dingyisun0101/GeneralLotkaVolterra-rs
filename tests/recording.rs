use std::fmt::Write as _;
use std::fs;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use general_lotka_volterra_rs::invariant::FrequencyInvariant;
use general_lotka_volterra_rs::kernel::{
    BoundaryCondition, Diffusion, InMemorySource, InteractionSource, Kernel, KernelCore,
    MeanFieldReplicatorRk4, persist_interaction_matrix,
};
use general_lotka_volterra_rs::noise::{
    DEMOGRAPHIC_GAUSSIAN_RNG_NAMESPACE, DemographicGaussian, Noise, NoiseDomain,
};
use general_lotka_volterra_rs::reading::open_completed_glv_recording;
use general_lotka_volterra_rs::recording::{
    ABUNDANCE_REPRESENTATION_METADATA_KEY, COMPLETED_ITERATION_METADATA_KEY, GlvRecording,
    GlvRecordingError, GlvRecordingMetadata, MODEL_KIND_METADATA_KEY, RecordingMetadataError,
    TASK_ORDINAL_METADATA_KEY, TERMINATION_REASON_METADATA_KEY, TerminationReason,
};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AbundanceRepresentation, AggregateAbundance, CHECKPOINT_STREAM,
    MeanFieldReplicator, MeanFieldReplicatorConfig, SIGNAL_STREAM, SPACE_FIELD, SPACE_STREAM,
    SpatialAbundance, SpatialReplicator, SpatialReplicatorConfig, TOTAL_FIELD, TimeStep,
    TotalAbundance,
};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use physics_in_parallel::rng::RngConfig;
use scientific_workflow::configuration::{ParameterSpace, TaskParameters};
use scientific_workflow::execution::ExecutionScope;
use scientific_workflow::rng_record::{RNG_RECORDS_METADATA_KEY, RngRecord};
use scientific_workflow::storage::{SamplingInterval, StateStreamConfig, StorageError};
use serde_json::Value;
use sha2::{Digest, Sha256};

static WORKSPACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Workspace {
    root: PathBuf,
}

impl Workspace {
    fn new(label: &str) -> Self {
        let sequence = WORKSPACE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "glv-recording-{label}-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        Self { root }
    }

    fn task_parameters(&self, fixed: &str, label: &str) -> TaskParameters {
        let config = self.root.join(label);
        fs::create_dir(&config).unwrap();
        fs::write(config.join("fixed.json"), fixed).unwrap();
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
            eprintln!(
                "failed to clean test workspace {}: {error}",
                self.root.display()
            );
        }
    }
}

fn stream(
    name: &str,
    fields: &[&str],
    interval: u64,
    max_chunk_bytes: u64,
    queue_bytes: u64,
) -> StateStreamConfig {
    StateStreamConfig::new(
        name,
        fields.iter().copied(),
        SamplingInterval::iterations(interval).unwrap(),
        Some((
            NonZeroU64::new(max_chunk_bytes).unwrap(),
            NonZeroU64::new(queue_bytes).unwrap(),
        )),
    )
}

fn recording_config(queue_bytes: u64) -> Vec<StateStreamConfig> {
    vec![
        stream(
            SIGNAL_STREAM,
            &[ABUNDANCE_FIELD, TOTAL_FIELD],
            2,
            128,
            queue_bytes,
        ),
        stream(
            SPACE_STREAM,
            &[ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD],
            3,
            160,
            queue_bytes,
        ),
        stream(
            CHECKPOINT_STREAM,
            &[ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD],
            4,
            192,
            queue_bytes,
        ),
    ]
}

fn make_simulation(
    interaction: general_lotka_volterra_rs::kernel::InteractionMatrix,
) -> MeanFieldReplicator {
    MeanFieldReplicator::new(
        Array1::from_vec(vec![0.4, 0.6]),
        interaction,
        MeanFieldReplicatorConfig::new(Array1::zeros(2), 0.0, TimeStep::new(0.1).unwrap()),
    )
    .unwrap()
}

fn make_stochastic_simulation(
    interaction: general_lotka_volterra_rs::kernel::InteractionMatrix,
) -> MeanFieldReplicator<MeanFieldReplicatorRk4, DemographicGaussian> {
    let time_step = TimeStep::new(0.1).unwrap();
    let state = MeanFieldReplicator::new(
        Array1::from_vec(vec![0.4, 0.6]),
        interaction.clone(),
        MeanFieldReplicatorConfig::new(Array1::zeros(2), 0.0, time_step),
    )
    .unwrap()
    .into_state();
    MeanFieldReplicator::from_plugins(
        state,
        AbundanceRepresentation::RelativeFrequency,
        Kernel::new(
            KernelCore::new(interaction),
            MeanFieldReplicatorRk4::new(Array1::zeros(2)).unwrap(),
        ),
        Noise::new(
            DemographicGaussian::new(
                0.05,
                RngConfig::new(Some(42), None, None),
                NoiseDomain::aggregate(2).unwrap(),
            )
            .unwrap(),
        ),
        FrequencyInvariant::new(2, 0.0).unwrap(),
        time_step,
    )
    .unwrap()
}

fn metadata(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path.join("metadata.json")).unwrap()).unwrap()
}

fn stream_metadata<'a>(metadata: &'a Value, name: &str) -> &'a Value {
    metadata["streams"]
        .as_array()
        .unwrap()
        .iter()
        .find(|stream| stream["name"] == name)
        .unwrap()
}

fn recorded_iterations(recording: &Path, stream: &Value) -> Vec<u64> {
    let directory = recording.join(stream["directory"].as_str().unwrap());
    let mut iterations = Vec::new();
    for chunk in stream["chunks"].as_array().unwrap() {
        let bytes = fs::read(directory.join(chunk["file"].as_str().unwrap())).unwrap();
        for line in bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
        {
            let record: Value = serde_json::from_slice(line).unwrap();
            iterations.push(record["iteration"].as_u64().unwrap());
        }
    }
    iterations
}

fn verify_chunk_integrity(recording: &Path, stream: &Value) -> u64 {
    let directory = recording.join(stream["directory"].as_str().unwrap());
    let mut total = 0_u64;
    for chunk in stream["chunks"].as_array().unwrap() {
        let bytes = fs::read(directory.join(chunk["file"].as_str().unwrap())).unwrap();
        assert_eq!(chunk["bytes"].as_u64().unwrap(), bytes.len() as u64);
        let digest = Sha256::digest(&bytes);
        let mut checksum = String::from("sha256:");
        for byte in digest {
            write!(&mut checksum, "{byte:02x}").unwrap();
        }
        assert_eq!(chunk["checksum"], checksum);
        assert_eq!(
            chunk["records"].as_u64().unwrap(),
            bytes.iter().filter(|byte| **byte == b'\n').count() as u64
        );
        total += bytes.len() as u64;
    }
    total
}

fn count_metadata_files(directory: &Path) -> usize {
    fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .map(|path| {
            if path.is_dir() {
                count_metadata_files(&path)
            } else {
                usize::from(path.file_name().is_some_and(|name| name == "metadata.json"))
            }
        })
        .sum()
}

fn physical_time_at(iteration: u64) -> f64 {
    let mut physical_time = 0.0;
    for _ in 0..iteration {
        physical_time += 0.1;
    }
    physical_time
}

fn json_files(directory: &Path, output: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            json_files(&path, output);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            output.push(path);
        }
    }
}

#[test]
fn workflow_records_all_glv_streams_metadata_terminal_state_and_integrity() {
    let workspace = Workspace::new("complete");
    let scope = ExecutionScope::create_named(&workspace.root, "execution").unwrap();
    let task =
        workspace.task_parameters(r#"{"seed":7,"physical_time_increment":0.1}"#, "task-config");
    let interaction = InMemorySource::new(Array2::zeros((2, 2)))
        .resolve(2)
        .unwrap();
    let persisted = persist_interaction_matrix(&scope, &interaction).unwrap();
    let mut simulation = make_simulation(interaction);
    let creation = GlvRecordingMetadata::new(
        simulation.kind(),
        simulation.abundance_representation(),
        &task,
        persisted.descriptor(),
        simulation.rng_record(),
    )
    .unwrap();
    let recording_directory = scope.task_recording_directory(task.task_ordinal());
    let mut recording = GlvRecording::start(
        &recording_directory,
        recording_config(8_192),
        creation,
        simulation.state(),
    )
    .unwrap();

    for _ in 0..5 {
        simulation.step().unwrap();
        recording.observe_state(simulation.state()).unwrap();
    }
    let completed = recording
        .complete(simulation.state(), TerminationReason::MaximumIterations)
        .unwrap();

    assert_eq!(completed.directory(), recording_directory);
    assert!(completed.timing().created_at_utc().ends_with('Z'));
    assert!(completed.timing().finalized_at_utc().ends_with('Z'));
    assert_eq!(completed.timing().continuation_count(), 0);
    assert_eq!(
        completed.terminal_metadata()[TERMINATION_REASON_METADATA_KEY],
        "maximum_iterations"
    );
    assert_eq!(
        completed.terminal_metadata()[COMPLETED_ITERATION_METADATA_KEY],
        5
    );
    assert_eq!(
        completed
            .stream_summary(SIGNAL_STREAM)
            .unwrap()
            .record_count(),
        4
    );
    assert_eq!(
        completed
            .stream_summary(SPACE_STREAM)
            .unwrap()
            .record_count(),
        3
    );
    assert_eq!(
        completed
            .stream_summary(CHECKPOINT_STREAM)
            .unwrap()
            .record_count(),
        3
    );

    let document = metadata(&recording_directory);
    assert_eq!(document["status"]["state"], "complete");
    assert_eq!(
        document["user_metadata"][MODEL_KIND_METADATA_KEY],
        "mean_field_replicator"
    );
    assert_eq!(
        document["user_metadata"][ABUNDANCE_REPRESENTATION_METADATA_KEY],
        "relative_frequency"
    );
    assert_eq!(document["user_metadata"][TASK_ORDINAL_METADATA_KEY], 0);
    assert_eq!(document["user_metadata"]["seed"], 7);
    assert_eq!(
        document["user_metadata"]["interaction_matrix"]["sha256"],
        persisted.descriptor().sha256()
    );
    assert_eq!(
        document["user_metadata"]["interaction_matrix"]["path"],
        persisted.descriptor().path()
    );
    assert_eq!(
        document["terminal_metadata"][COMPLETED_ITERATION_METADATA_KEY],
        5
    );
    assert_eq!(document["streams"][0]["name"], SIGNAL_STREAM);
    assert_eq!(document["streams"][1]["name"], SPACE_STREAM);
    assert_eq!(document["streams"][2]["name"], CHECKPOINT_STREAM);

    let signal = stream_metadata(&document, SIGNAL_STREAM);
    let space = stream_metadata(&document, SPACE_STREAM);
    let checkpoint = stream_metadata(&document, CHECKPOINT_STREAM);
    assert_eq!(
        signal["sampling_interval"],
        serde_json::json!({"iterations": 2})
    );
    assert_eq!(
        space["sampling_interval"],
        serde_json::json!({"iterations": 3})
    );
    assert_eq!(
        checkpoint["sampling_interval"],
        serde_json::json!({"iterations": 4})
    );
    assert_eq!(
        recorded_iterations(&recording_directory, signal),
        [0, 2, 4, 5]
    );
    assert_eq!(recorded_iterations(&recording_directory, space), [0, 3, 5]);
    assert_eq!(
        recorded_iterations(&recording_directory, checkpoint),
        [0, 4, 5]
    );
    assert_eq!(
        signal["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["abundance", "total"]
    );
    for stream in [space, checkpoint] {
        assert_eq!(
            stream["fields"]
                .as_array()
                .unwrap()
                .iter()
                .map(|field| field["name"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["abundance", "space", "total"]
        );
    }
    let checkpoint_directory = recording_directory.join(checkpoint["directory"].as_str().unwrap());
    let first_checkpoint_chunk = &checkpoint["chunks"][0];
    let first_checkpoint_bytes =
        fs::read(checkpoint_directory.join(first_checkpoint_chunk["file"].as_str().unwrap()))
            .unwrap();
    let first_checkpoint_line = first_checkpoint_bytes
        .split(|byte| *byte == b'\n')
        .find(|line| !line.is_empty())
        .unwrap();
    let first_checkpoint: Value = serde_json::from_slice(first_checkpoint_line).unwrap();
    assert!(first_checkpoint["values"]["space"].is_null());

    for name in [SIGNAL_STREAM, SPACE_STREAM, CHECKPOINT_STREAM] {
        let stream = stream_metadata(&document, name);
        let encoded_bytes = verify_chunk_integrity(&recording_directory, stream);
        let summary = completed.stream_summary(name).unwrap();
        assert_eq!(summary.encoded_bytes(), encoded_bytes);
        assert_eq!(
            summary.chunk_count(),
            stream["chunks"].as_array().unwrap().len() as u64
        );
    }
    assert_eq!(count_metadata_files(&recording_directory), 1);
    let mut recording_json_files = Vec::new();
    json_files(&recording_directory, &mut recording_json_files);
    assert_eq!(
        recording_json_files,
        [recording_directory.join("metadata.json")]
    );

    let reader = open_completed_glv_recording(&recording_directory).unwrap();
    let signal_series = reader.read_stream_as_state_series(SIGNAL_STREAM).unwrap();
    let space_series = reader.read_stream_as_state_series(SPACE_STREAM).unwrap();
    assert_eq!(
        signal_series
            .iter()
            .map(|state| state.simulation_time().iteration())
            .collect::<Vec<_>>(),
        [0, 2, 4, 5]
    );
    for state in signal_series.iter() {
        let time = state.simulation_time();
        assert_eq!(
            time.physical_time(),
            Some(physical_time_at(time.iteration()))
        );
        assert_eq!(
            state
                .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
                .unwrap(),
            &Array1::from_vec(vec![0.4, 0.6])
        );
        assert_eq!(*state.payload::<TotalAbundance>(TOTAL_FIELD).unwrap(), 1.0);
    }
    assert_eq!(
        space_series
            .iter()
            .map(|state| state.simulation_time().iteration())
            .collect::<Vec<_>>(),
        [0, 3, 5]
    );
    for state in space_series.iter() {
        assert!(
            state
                .payload::<SpatialAbundance>(SPACE_FIELD)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn stochastic_noise_identity_is_written_once_in_creation_metadata() {
    let workspace = Workspace::new("rng-record");
    let scope = ExecutionScope::create_named(&workspace.root, "execution").unwrap();
    let task = workspace.task_parameters(r#"{"seed":42,"noise_sigma":0.05}"#, "task-config");
    let interaction = InMemorySource::new(Array2::zeros((2, 2)))
        .resolve(2)
        .unwrap();
    let persisted = persist_interaction_matrix(&scope, &interaction).unwrap();
    let simulation = make_stochastic_simulation(interaction);
    let creation = GlvRecordingMetadata::new(
        simulation.kind(),
        simulation.abundance_representation(),
        &task,
        persisted.descriptor(),
        simulation.rng_record(),
    )
    .unwrap();
    let recording_directory = scope.task_recording_directory(0);
    GlvRecording::start(
        &recording_directory,
        recording_config(8_192),
        creation,
        simulation.state(),
    )
    .unwrap()
    .complete(simulation.state(), TerminationReason::Requested)
    .unwrap();

    let document = metadata(&recording_directory);
    let record =
        &document["user_metadata"][RNG_RECORDS_METADATA_KEY][DEMOGRAPHIC_GAUSSIAN_RNG_NAMESPACE];
    assert_eq!(record["method"], "chacha12+standard_normal");
    assert_eq!(record["version"], "rand_chacha-0.10");
    assert_eq!(record["key_encoding"], "u64_decimal");
    assert_eq!(record["key"], "42");
    let decoded = RngRecord::from_metadata(
        document["user_metadata"].as_object().unwrap(),
        DEMOGRAPHIC_GAUSSIAN_RNG_NAMESPACE,
    )
    .unwrap()
    .unwrap();
    assert_eq!(decoded.key(), "42");
    assert!(
        document["streams"]
            .as_array()
            .unwrap()
            .iter()
            .all(|stream| {
                !serde_json::to_string(stream)
                    .unwrap()
                    .contains("chacha12+standard_normal")
            })
    );
}

#[test]
fn recording_metadata_and_failure_lifecycle_fail_closed() {
    let workspace = Workspace::new("lifecycle");
    let scope = ExecutionScope::create_named(&workspace.root, "execution").unwrap();
    let task = workspace.task_parameters(r#"{"seed":9}"#, "task-config");
    let interaction = InMemorySource::new(Array2::zeros((2, 2)))
        .resolve(2)
        .unwrap();
    let persisted = persist_interaction_matrix(&scope, &interaction).unwrap();
    let mut simulation = make_simulation(interaction);

    assert!(matches!(
        GlvRecordingMetadata::new(
            simulation.kind(),
            AbundanceRepresentation::AbsoluteCount,
            &task,
            persisted.descriptor(),
            simulation.rng_record(),
        ),
        Err(RecordingMetadataError::RepresentationMismatch { .. })
    ));
    let reserved = workspace.task_parameters(r#"{"model_kind":"collision"}"#, "reserved");
    assert!(matches!(
        GlvRecordingMetadata::new(
            simulation.kind(),
            simulation.abundance_representation(),
            &reserved,
            persisted.descriptor(),
            simulation.rng_record(),
        ),
        Err(RecordingMetadataError::ReservedTaskParameter { .. })
    ));

    let creation = GlvRecordingMetadata::new(
        simulation.kind(),
        simulation.abundance_representation(),
        &task,
        persisted.descriptor(),
        simulation.rng_record(),
    )
    .unwrap();
    let failed_directory = scope.task_recording_directory(1);
    let mut failed = GlvRecording::start(
        &failed_directory,
        recording_config(8_192),
        creation.clone(),
        simulation.state(),
    )
    .unwrap();
    simulation.step().unwrap();
    failed.observe_state(simulation.state()).unwrap();
    failed
        .mark_failed(simulation.state(), "intentional numerical stop")
        .unwrap();
    let failed_metadata = metadata(&failed_directory);
    assert_eq!(failed_metadata["status"]["state"], "failed");
    assert_eq!(
        failed_metadata["status"]["message"],
        "intentional numerical stop"
    );
    assert_eq!(
        failed_metadata["terminal_metadata"][COMPLETED_ITERATION_METADATA_KEY],
        1
    );
    assert!(failed_metadata["timing"]["finalized_at_utc"].is_string());

    let interrupted_directory = scope.task_recording_directory(2);
    let interrupted = GlvRecording::start(
        &interrupted_directory,
        recording_config(8_192),
        creation.clone(),
        simulation.state(),
    )
    .unwrap();
    drop(interrupted);
    let interrupted_metadata = metadata(&interrupted_directory);
    assert_eq!(interrupted_metadata["status"]["state"], "running");
    assert!(interrupted_metadata.get("terminal_metadata").is_none());

    let bounded_directory = scope.task_recording_directory(3);
    let bounded_simulation = make_simulation(
        InMemorySource::new(Array2::zeros((2, 2)))
            .resolve(2)
            .unwrap(),
    );
    let bounded_error = match GlvRecording::start(
        &bounded_directory,
        recording_config(1),
        creation,
        bounded_simulation.state(),
    ) {
        Ok(_) => panic!("one-byte queue unexpectedly accepted a GLV record"),
        Err(error) => error,
    };
    assert!(
        matches!(
            bounded_error,
            GlvRecordingError::Storage(StorageError::RecordTooLarge { limit: 1, .. })
        ),
        "unexpected bounded-queue error: {bounded_error:?}"
    );
    assert_eq!(metadata(&bounded_directory)["status"]["state"], "running");
}

#[test]
fn completed_reader_round_trips_populated_spatial_payload_and_exact_time() {
    let workspace = Workspace::new("spatial-read");
    let scope = ExecutionScope::create_named(&workspace.root, "execution").unwrap();
    let task = workspace.task_parameters(r#"{"seed":11}"#, "task-config");
    let interaction = InMemorySource::new(Array2::zeros((2, 2)))
        .resolve(2)
        .unwrap();
    let persisted = persist_interaction_matrix(&scope, &interaction).unwrap();
    let initial_space = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.4, 0.6]).unwrap();
    let mut simulation = SpatialReplicator::new(
        initial_space.clone(),
        interaction,
        SpatialReplicatorConfig::new(
            Array1::zeros(2),
            Diffusion::unit_spacing(Array1::zeros(2), &[1], BoundaryCondition::Periodic).unwrap(),
            0.0,
            TimeStep::new(0.1).unwrap(),
        ),
    )
    .unwrap();
    let creation = GlvRecordingMetadata::new(
        simulation.kind(),
        simulation.abundance_representation(),
        &task,
        persisted.descriptor(),
        simulation.rng_record(),
    )
    .unwrap();
    let recording_directory = scope.task_recording_directory(0);
    let all_iterations = vec![
        stream(
            SIGNAL_STREAM,
            &[ABUNDANCE_FIELD, TOTAL_FIELD],
            1,
            1_024,
            8_192,
        ),
        stream(
            SPACE_STREAM,
            &[ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD],
            1,
            1_024,
            8_192,
        ),
        stream(
            CHECKPOINT_STREAM,
            &[ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD],
            1,
            1_024,
            8_192,
        ),
    ];
    let mut recording = GlvRecording::start(
        &recording_directory,
        all_iterations,
        creation,
        simulation.state(),
    )
    .unwrap();
    simulation.step().unwrap();
    recording.observe_state(simulation.state()).unwrap();
    recording
        .complete(simulation.state(), TerminationReason::MaximumIterations)
        .unwrap();

    let reader = open_completed_glv_recording(&recording_directory).unwrap();
    let series = reader.read_stream_as_state_series(SPACE_STREAM).unwrap();
    assert_eq!(series.len(), 2);
    for state in series.iter() {
        let time = state.simulation_time();
        assert_eq!(
            time.physical_time(),
            Some(physical_time_at(time.iteration()))
        );
        assert_eq!(
            state
                .payload::<SpatialAbundance>(SPACE_FIELD)
                .unwrap()
                .as_ref()
                .unwrap(),
            &initial_space
        );
    }
}
