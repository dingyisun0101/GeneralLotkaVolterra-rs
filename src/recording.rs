//! Scientific Workflow recording integration for concrete GLV simulations.
//!
//! This module configures Workflow state streams and lifecycle metadata. It
//! does not implement sampling, serialization, queues, chunks, checksums,
//! timing, or filesystem layout; those remain owned by `SystemStateWriter`.

use std::num::NonZeroU64;
use std::path::Path;

use scientific_workflow::configuration::TaskParameters;
use scientific_workflow::rng_record::{RNG_RECORDS_METADATA_KEY, RngRecord, RngRecordError};
use scientific_workflow::storage::{
    CompletedRecording, SamplingInterval, StateStreamConfig, StorageError, SystemStateWriter,
    SystemStateWriterBuilder, TimeAxisMetadata,
};
use scientific_workflow::system_state::{StateError, SystemState};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::kernel::{INTERACTION_MATRIX_METADATA_KEY, InteractionArtifactDescriptor};
use crate::reading::glv_json_decoders;
use crate::simulation::SimulationKind;
use crate::{
    ABUNDANCE_FIELD, AbundanceRepresentation, CHECKPOINT_STREAM, SIGNAL_STREAM, SPACE_FIELD,
    SPACE_STREAM, TOTAL_FIELD, load_state_schema,
};

/// Creation-metadata key containing the concrete model identity.
pub const MODEL_KIND_METADATA_KEY: &str = "model_kind";

/// Creation-metadata key containing abundance interpretation.
pub const ABUNDANCE_REPRESENTATION_METADATA_KEY: &str = "abundance_representation";

/// Creation-metadata key containing the resolved Workflow task ordinal.
pub const TASK_ORDINAL_METADATA_KEY: &str = "task_ordinal";

/// Terminal-metadata key containing the typed successful termination reason.
pub const TERMINATION_REASON_METADATA_KEY: &str = "termination_reason";

/// Terminal-metadata key containing the last successfully completed iteration.
pub const COMPLETED_ITERATION_METADATA_KEY: &str = "completed_iteration";

const RESERVED_CREATION_KEYS: [&str; 5] = [
    MODEL_KIND_METADATA_KEY,
    ABUNDANCE_REPRESENTATION_METADATA_KEY,
    INTERACTION_MATRIX_METADATA_KEY,
    TASK_ORDINAL_METADATA_KEY,
    RNG_RECORDS_METADATA_KEY,
];

/// Nonzero storage bounds for one independently sampled stream.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamRecordingConfig {
    sampling_interval: SamplingInterval,
    max_chunk_bytes: NonZeroU64,
    queue_bytes: NonZeroU64,
}

impl StreamRecordingConfig {
    /// Creates a stream configuration whose invalid zero values are unrepresentable.
    pub const fn new(
        sampling_interval: SamplingInterval,
        max_chunk_bytes: NonZeroU64,
        queue_bytes: NonZeroU64,
    ) -> Self {
        Self {
            sampling_interval,
            max_chunk_bytes,
            queue_bytes,
        }
    }

    /// Returns the writer-owned sampling interval.
    pub const fn sampling_interval(self) -> SamplingInterval {
        self.sampling_interval
    }

    /// Returns the rollover target for immutable chunk files.
    pub const fn max_chunk_bytes(self) -> NonZeroU64 {
        self.max_chunk_bytes
    }

    /// Returns the strict encoded-byte queue budget.
    pub const fn queue_bytes(self) -> NonZeroU64 {
        self.queue_bytes
    }
}

/// Independent signal, space, and checkpoint stream configuration.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GlvRecordingConfig {
    signal: StreamRecordingConfig,
    space: StreamRecordingConfig,
    checkpoint: StreamRecordingConfig,
}

impl GlvRecordingConfig {
    /// Creates one mandatory configuration for each canonical stream.
    pub const fn new(
        signal: StreamRecordingConfig,
        space: StreamRecordingConfig,
        checkpoint: StreamRecordingConfig,
    ) -> Self {
        Self {
            signal,
            space,
            checkpoint,
        }
    }

    /// Returns aggregate signal-stream configuration.
    pub const fn signal(self) -> StreamRecordingConfig {
        self.signal
    }

    /// Returns spatial analysis-stream configuration.
    pub const fn space(self) -> StreamRecordingConfig {
        self.space
    }

    /// Returns deterministic checkpoint-stream configuration.
    pub const fn checkpoint(self) -> StreamRecordingConfig {
        self.checkpoint
    }
}

/// Fully assembled immutable creation metadata for one simulation recording.
#[derive(Clone, Debug)]
pub struct GlvRecordingMetadata {
    model_kind: SimulationKind,
    abundance_representation: AbundanceRepresentation,
    values: Map<String, Value>,
}

impl GlvRecordingMetadata {
    /// Combines exact resolved task parameters with GLV and matrix provenance.
    pub fn new(
        model_kind: SimulationKind,
        abundance_representation: AbundanceRepresentation,
        task: &TaskParameters,
        interaction: &InteractionArtifactDescriptor,
        rng_record: Option<&RngRecord>,
    ) -> Result<Self, RecordingMetadataError> {
        let expected = model_kind.abundance_representation();
        if abundance_representation != expected {
            return Err(RecordingMetadataError::RepresentationMismatch {
                model_kind,
                expected,
                actual: abundance_representation,
            });
        }
        for key in RESERVED_CREATION_KEYS {
            if task.contains(key) {
                return Err(RecordingMetadataError::ReservedTaskParameter {
                    key: key.to_owned(),
                });
            }
        }
        let mut values = task
            .iter()
            .map(|(key, value)| (key.to_owned(), value.clone()))
            .collect::<Map<_, _>>();
        values.insert(
            TASK_ORDINAL_METADATA_KEY.to_owned(),
            Value::from(task.task_ordinal()),
        );
        values.insert(
            MODEL_KIND_METADATA_KEY.to_owned(),
            Value::from(model_kind.as_str()),
        );
        values.insert(
            ABUNDANCE_REPRESENTATION_METADATA_KEY.to_owned(),
            Value::from(abundance_representation.as_str()),
        );
        if interaction.insert_into_metadata(&mut values).is_some() {
            return Err(RecordingMetadataError::ReservedTaskParameter {
                key: INTERACTION_MATRIX_METADATA_KEY.to_owned(),
            });
        }
        if let Some(record) = rng_record {
            record.insert_into_metadata(&mut values)?;
        }
        Ok(Self {
            model_kind,
            abundance_representation,
            values,
        })
    }

    /// Returns the concrete simulation identity.
    pub const fn model_kind(&self) -> SimulationKind {
        self.model_kind
    }

    /// Returns the immutable abundance representation.
    pub const fn abundance_representation(&self) -> AbundanceRepresentation {
        self.abundance_representation
    }

    /// Borrows the complete Workflow user-metadata map.
    pub const fn values(&self) -> &Map<String, Value> {
        &self.values
    }

    /// Adds another application-namespaced Workflow RNG record.
    pub fn with_rng_record(mut self, record: &RngRecord) -> Result<Self, RecordingMetadataError> {
        record.insert_into_metadata(&mut self.values)?;
        Ok(self)
    }

    fn into_values(self) -> Map<String, Value> {
        self.values
    }
}

/// Successful scientific stopping condition committed at recording completion.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TerminationReason {
    /// The configured maximum completed iteration was reached.
    MaximumIterations,
    /// Only one species remains feasible under the model invariant.
    Monoculture,
    /// No species remains feasible under the model invariant.
    Extinction,
    /// External orchestration requested a successful stop.
    Requested,
}

impl TerminationReason {
    /// Returns the stable terminal-metadata value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaximumIterations => "maximum_iterations",
            Self::Monoculture => "monoculture",
            Self::Extinction => "extinction",
            Self::Requested => "requested",
        }
    }
}

/// Rejection while composing creation-time recording metadata.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RecordingMetadataError {
    /// A Workflow RNG record was invalid or reused a namespace.
    #[error(transparent)]
    RngRecord(#[from] RngRecordError),
    /// A task parameter would overwrite GLV-owned provenance.
    #[error("resolved task parameter `{key}` collides with reserved recording metadata")]
    ReservedTaskParameter {
        /// Conflicting exact parameter key.
        key: String,
    },
    /// Model identity and abundance interpretation disagree.
    #[error(
        "model {model_kind:?} requires representation {}, found {}",
        expected.as_str(),
        actual.as_str()
    )]
    RepresentationMismatch {
        /// Concrete model whose metadata is being assembled.
        model_kind: SimulationKind,
        /// Structurally required representation.
        expected: AbundanceRepresentation,
        /// Rejected representation.
        actual: AbundanceRepresentation,
    },
}

/// Failure while configuring or operating the Workflow writer.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GlvRecordingError {
    /// The canonical GLV schema could not be loaded.
    #[error("canonical GLV state schema could not be loaded: {0}")]
    StateSchema(#[source] StateError),
    /// Scientific Workflow rejected recording configuration or lifecycle work.
    #[error(transparent)]
    Storage(#[from] StorageError),
}

/// Sole GLV-facing owner of one Scientific Workflow state writer.
pub struct GlvRecording {
    writer: SystemStateWriter,
}

impl GlvRecording {
    /// Creates a new recording and immediately offers the initial state once.
    ///
    /// If initial observation fails, Workflow leaves the created recording in
    /// running state as interruption evidence; no false terminal transition is
    /// inferred by this layer.
    pub fn start(
        directory: impl AsRef<Path>,
        config: GlvRecordingConfig,
        metadata: GlvRecordingMetadata,
        initial_state: &SystemState,
    ) -> Result<Self, GlvRecordingError> {
        let schema = load_state_schema().map_err(GlvRecordingError::StateSchema)?;
        let mut writer = recording_builder(directory.as_ref(), &schema, config, metadata)
            .create_new_recording()?;
        writer.observe_state(initial_state)?;
        Ok(Self { writer })
    }

    /// Reopens a running recording and reconstructs its newest complete checkpoint.
    ///
    /// Scientific Workflow validates the complete writer configuration,
    /// recovers any open tails, and enforces sealed-checkpoint integrity before
    /// this method receives either the state or append-capable writer. The
    /// returned checkpoint is not observed again; orchestration records only
    /// subsequently successful evolution steps.
    pub fn continue_from_latest_checkpoint(
        directory: impl AsRef<Path>,
        config: GlvRecordingConfig,
        metadata: GlvRecordingMetadata,
    ) -> Result<(Self, SystemState), GlvRecordingError> {
        let schema = load_state_schema().map_err(GlvRecordingError::StateSchema)?;
        let (writer, state) = recording_builder(directory.as_ref(), &schema, config, metadata)
            .continue_recording_from_latest_checkpoint(CHECKPOINT_STREAM, glv_json_decoders()?)?;
        Ok((Self { writer }, state))
    }

    /// Returns the Workflow-owned recording directory.
    pub fn recording_directory(&self) -> &Path {
        self.writer.recording_directory()
    }

    /// Offers one successfully evolved state to writer-owned sampling.
    pub fn observe_state(&mut self, state: &SystemState) -> Result<(), GlvRecordingError> {
        self.writer.observe_state(state)?;
        Ok(())
    }

    /// Records the final state exactly once and commits successful termination.
    pub fn complete(
        self,
        final_state: &SystemState,
        reason: TerminationReason,
    ) -> Result<CompletedRecording, GlvRecordingError> {
        let mut terminal_metadata = Map::new();
        terminal_metadata.insert(
            TERMINATION_REASON_METADATA_KEY.to_owned(),
            Value::from(reason.as_str()),
        );
        terminal_metadata.insert(
            COMPLETED_ITERATION_METADATA_KEY.to_owned(),
            Value::from(final_state.simulation_time().iteration()),
        );
        Ok(self
            .writer
            .complete_recording_with_final_state_and_terminal_metadata(
                final_state,
                terminal_metadata,
            )?)
    }

    /// Marks an intentional simulation failure without recording failed state.
    pub fn mark_failed(
        self,
        last_successful_state: &SystemState,
        message: impl Into<String>,
    ) -> Result<(), GlvRecordingError> {
        let mut terminal_metadata = Map::new();
        terminal_metadata.insert(
            COMPLETED_ITERATION_METADATA_KEY.to_owned(),
            Value::from(last_successful_state.simulation_time().iteration()),
        );
        self.writer
            .mark_recording_failed_with_terminal_metadata(message, terminal_metadata)?;
        Ok(())
    }
}

fn recording_builder(
    directory: &Path,
    schema: &scientific_workflow::system_state::SystemStateSchema,
    config: GlvRecordingConfig,
    metadata: GlvRecordingMetadata,
) -> SystemStateWriterBuilder {
    SystemStateWriter::builder(directory, schema)
        .with_time_axis_metadata(
            TimeAxisMetadata::new("iteration").with_physical_time_name("physical_time"),
        )
        .with_user_metadata(metadata.into_values())
        .add_state_stream(state_stream(
            SIGNAL_STREAM,
            [ABUNDANCE_FIELD, TOTAL_FIELD],
            config.signal(),
        ))
        .add_state_stream(state_stream(
            SPACE_STREAM,
            [ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD],
            config.space(),
        ))
        .add_state_stream(state_stream(
            CHECKPOINT_STREAM,
            [ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD],
            config.checkpoint(),
        ))
}

fn state_stream<const FIELDS: usize>(
    name: &'static str,
    fields: [&'static str; FIELDS],
    config: StreamRecordingConfig,
) -> StateStreamConfig {
    StateStreamConfig::new(
        name,
        fields,
        config.sampling_interval(),
        config.max_chunk_bytes(),
        config.queue_bytes(),
    )
}
