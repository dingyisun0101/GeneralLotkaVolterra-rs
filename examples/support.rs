use std::error::Error;
use std::io;
use std::num::NonZeroU64;
use std::path::PathBuf;

use general_lotka_volterra_rs::reading::open_completed_glv_recording;
use general_lotka_volterra_rs::recording::{GlvRecordingConfig, StreamRecordingConfig};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AggregateAbundance, CHECKPOINT_STREAM, SIGNAL_STREAM, SPACE_FIELD,
    SPACE_STREAM, SpatialAbundance, TOTAL_FIELD, TotalAbundance,
};
use scientific_workflow::prelude::{SamplingInterval, ScientificProject, SystemState};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct StreamInputs {
    sampling_interval: u64,
    max_chunk_bytes: u64,
    queue_bytes: u64,
}

#[derive(Deserialize)]
pub struct RecordingInputs {
    signal: StreamInputs,
    space: StreamInputs,
    checkpoint: StreamInputs,
}

pub fn load_project(default_name: &str) -> Result<ScientificProject, Box<dyn Error>> {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("examples")
                .join(default_name)
        });
    let project = ScientificProject::load(root)?;
    validate_state_schema(&project)?;
    Ok(project)
}

pub fn recording_config(inputs: RecordingInputs) -> Result<GlvRecordingConfig, Box<dyn Error>> {
    Ok(GlvRecordingConfig::new(
        stream_config(SIGNAL_STREAM, inputs.signal)?,
        stream_config(SPACE_STREAM, inputs.space)?,
        stream_config(CHECKPOINT_STREAM, inputs.checkpoint)?,
    ))
}

pub fn validate_completed_recording(
    directory: impl Into<PathBuf>,
    final_state: &SystemState,
) -> Result<(), Box<dyn Error>> {
    let directory = directory.into();
    let reader = open_completed_glv_recording(&directory)?;
    let recorded = reader.read_latest_state_from_stream(CHECKPOINT_STREAM)?;
    if recorded.simulation_time() != final_state.simulation_time()
        || recorded.payload::<AggregateAbundance>(ABUNDANCE_FIELD)?
            != final_state.payload::<AggregateAbundance>(ABUNDANCE_FIELD)?
        || recorded.payload::<SpatialAbundance>(SPACE_FIELD)?
            != final_state.payload::<SpatialAbundance>(SPACE_FIELD)?
        || recorded.payload::<TotalAbundance>(TOTAL_FIELD)?
            != final_state.payload::<TotalAbundance>(TOTAL_FIELD)?
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "completed checkpoint does not equal the final simulation state in {}",
                directory.display()
            ),
        )
        .into());
    }
    Ok(())
}

fn stream_config(
    stream: &'static str,
    inputs: StreamInputs,
) -> Result<StreamRecordingConfig, Box<dyn Error>> {
    let sampling_interval = SamplingInterval::iterations(inputs.sampling_interval)
        .ok_or_else(|| nonzero_error(stream, "sampling_interval"))?;
    let max_chunk_bytes = NonZeroU64::new(inputs.max_chunk_bytes)
        .ok_or_else(|| nonzero_error(stream, "max_chunk_bytes"))?;
    let queue_bytes =
        NonZeroU64::new(inputs.queue_bytes).ok_or_else(|| nonzero_error(stream, "queue_bytes"))?;
    Ok(StreamRecordingConfig::new(
        sampling_interval,
        max_chunk_bytes,
        queue_bytes,
    ))
}

fn nonzero_error(stream: &str, field: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("{stream} {field} must be nonzero"),
    )
}

fn validate_state_schema(project: &ScientificProject) -> Result<(), io::Error> {
    let actual = project
        .state_schema()
        .field_schemas()
        .iter()
        .map(|field| field.name())
        .collect::<Vec<_>>();
    let expected = [ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD];
    if actual == expected {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("state schema fields must be {expected:?}, found {actual:?}"),
        ))
    }
}
