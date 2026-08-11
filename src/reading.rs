//! Thin typed reading configuration over Scientific Workflow storage.
//!
//! Completed-stream integrity, decoding, series assembly, and metadata access
//! remain owned by `StoredStateSeriesReader`.

use std::path::Path;

use scientific_workflow::storage::{
    JsonPayloadDecoderRegistry, StorageError, StoredStateSeriesReader,
};

use crate::{
    ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, TotalAbundance,
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
