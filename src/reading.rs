//! Thin typed reading configuration over Scientific Workflow storage.
//!
//! Completed-stream integrity, decoding, series assembly, and metadata access
//! remain owned by `StoredStateSeriesReader`.

use std::path::{Path, PathBuf};

use scientific_workflow::storage::{
    JsonPayloadDecoderRegistry, StorageError, StoredStateSeriesReader,
};
use scientific_workflow::system_state::{StateError, SystemState};
use thiserror::Error;

use crate::{
    ABUNDANCE_FIELD, AggregateAbundance, CHECKPOINT_STREAM, SPACE_FIELD, SpatialAbundance,
    TOTAL_FIELD, TotalAbundance,
};

/// Registers direct Serde decoders for every canonical GLV state payload.
pub fn glv_json_decoders() -> Result<JsonPayloadDecoderRegistry, StorageError> {
    JsonPayloadDecoderRegistry::with_capacity(3)
        .with_json_field::<AggregateAbundance>(ABUNDANCE_FIELD)?
        .with_json_field::<SpatialAbundance>(SPACE_FIELD)?
        .with_json_field::<TotalAbundance>(TOTAL_FIELD)
}

/// Opens one completed Workflow recording with canonical GLV decoders.
pub fn open_completed_glv_recording(
    directory: impl AsRef<Path>,
) -> Result<StoredStateSeriesReader, StorageError> {
    StoredStateSeriesReader::open_completed_recording(directory, glv_json_decoders()?)
}

/// Reopens a completed recording and verifies its final canonical checkpoint.
pub fn verify_completed_glv_checkpoint(
    directory: impl AsRef<Path>,
    expected: &SystemState,
) -> Result<(), GlvCheckpointVerificationError> {
    let directory = directory.as_ref();
    let reader = open_completed_glv_recording(directory)?;
    let recorded = reader.read_latest_state_from_stream(CHECKPOINT_STREAM)?;
    let matches = recorded.simulation_time() == expected.simulation_time()
        && recorded.payload::<AggregateAbundance>(ABUNDANCE_FIELD)?
            == expected.payload::<AggregateAbundance>(ABUNDANCE_FIELD)?
        && recorded.payload::<SpatialAbundance>(SPACE_FIELD)?
            == expected.payload::<SpatialAbundance>(SPACE_FIELD)?
        && recorded.payload::<TotalAbundance>(TOTAL_FIELD)?
            == expected.payload::<TotalAbundance>(TOTAL_FIELD)?;
    if matches {
        Ok(())
    } else {
        Err(GlvCheckpointVerificationError::FinalStateMismatch {
            directory: directory.to_path_buf(),
        })
    }
}

/// Failure while verifying a completed recording's final GLV checkpoint.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GlvCheckpointVerificationError {
    /// Workflow rejected recording integrity, decoding, or reconstruction.
    #[error(transparent)]
    Storage(#[from] StorageError),
    /// A reconstructed checkpoint lacked a canonical typed payload.
    #[error(transparent)]
    State(#[from] StateError),
    /// The verified checkpoint differs from the expected authoritative state.
    #[error("completed checkpoint does not equal the final simulation state in {directory}")]
    FinalStateMismatch {
        /// Completed recording that contained the mismatched checkpoint.
        directory: PathBuf,
    },
}
