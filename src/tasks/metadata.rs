/*!
Task metadata output.

Purpose:
    Task runners persist one `metadata.json` file beside signal/space output so
    downstream tools can inspect what was requested, what ran, why it stopped,
    and how many chunks were written.
*/

use std::fs::{File, create_dir_all, read_to_string, remove_dir_all, remove_file};
use std::io::{BufWriter, Error, ErrorKind, Result};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::io::WriterStats;
use crate::solvers::termination::TerminationReason;
use crate::{SIGNAL_OUTPUT_FILE_SIZE, SPACE_OUTPUT_FILE_SIZE};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TaskOutcome {
    pub task: String,
    pub model: String,
    pub output_label: String,
    pub requested_steps: usize,
    pub steps_run: usize,
    pub start_time: usize,
    pub end_time: usize,
    pub dt: f64,
    pub signal_save_interval: usize,
    pub space_save_interval: Option<usize>,
    pub num_species: usize,
    pub spatial_shape: Option<Vec<usize>>,
    pub cutoff: Option<f64>,
    pub carrying_capacity: Option<f64>,
    pub survivor_tolerance: Option<f64>,
    pub termination_reason: TerminationReason,
    pub signal: WriterStats,
    pub space: Option<WriterStats>,
    pub signal_chunk_bytes: usize,
    pub space_chunk_bytes: Option<usize>,
}

impl TaskOutcome {
    pub fn non_spatial(
        task: &str,
        model: &str,
        output_label: &str,
        requested_steps: usize,
        dt: f64,
        save_interval: usize,
        steps_run: usize,
        termination_reason: TerminationReason,
        signal: WriterStats,
        num_species: usize,
        cutoff: Option<f64>,
        survivor_tolerance: Option<f64>,
    ) -> Self {
        Self {
            task: task.to_owned(),
            model: model.to_owned(),
            output_label: output_label.to_owned(),
            requested_steps,
            steps_run,
            start_time: 0,
            end_time: steps_run,
            dt,
            signal_save_interval: save_interval,
            space_save_interval: None,
            num_species,
            spatial_shape: None,
            cutoff,
            carrying_capacity: None,
            survivor_tolerance,
            termination_reason,
            signal,
            space: None,
            signal_chunk_bytes: SIGNAL_OUTPUT_FILE_SIZE,
            space_chunk_bytes: None,
        }
    }

    pub fn spatial(
        task: &str,
        model: &str,
        output_label: &str,
        requested_steps: usize,
        dt: f64,
        save_interval: usize,
        steps_run: usize,
        termination_reason: TerminationReason,
        signal: WriterStats,
        space: WriterStats,
        num_species: usize,
        spatial_shape: &[usize],
        cutoff: Option<f64>,
        carrying_capacity: Option<f64>,
        survivor_tolerance: Option<f64>,
    ) -> Self {
        Self {
            task: task.to_owned(),
            model: model.to_owned(),
            output_label: output_label.to_owned(),
            requested_steps,
            steps_run,
            start_time: 0,
            end_time: steps_run,
            dt,
            signal_save_interval: save_interval,
            space_save_interval: Some(save_interval),
            num_species,
            spatial_shape: Some(spatial_shape.to_vec()),
            cutoff,
            carrying_capacity,
            survivor_tolerance,
            termination_reason,
            signal,
            space: Some(space),
            signal_chunk_bytes: SIGNAL_OUTPUT_FILE_SIZE,
            space_chunk_bytes: Some(SPACE_OUTPUT_FILE_SIZE),
        }
    }
}

pub fn prepare_output_dir(output_path: &Path) -> Result<()> {
    if output_path.is_file() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("output path {} is a file", output_path.display()),
        ));
    }

    create_dir_all(output_path).map_err(|e| {
        Error::new(
            e.kind(),
            format!(
                "prepare_output_dir: create dir {}: {e}",
                output_path.display()
            ),
        )
    })?;

    remove_child_dir(output_path, "signal")?;
    remove_child_dir(output_path, "space")?;

    let metadata = output_path.join("metadata.json");
    if metadata.exists() {
        remove_file(&metadata).map_err(|e| {
            Error::new(
                e.kind(),
                format!("prepare_output_dir: remove {}: {e}", metadata.display()),
            )
        })?;
    }

    Ok(())
}

pub fn output_label(output_path: &Path) -> String {
    output_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output")
        .to_owned()
}

fn remove_child_dir(output_path: &Path, name: &str) -> Result<()> {
    let path = output_path.join(name);
    if path.exists() {
        if !path.is_dir() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("output child {} is not a directory", path.display()),
            ));
        }

        remove_dir_all(&path).map_err(|e| {
            Error::new(
                e.kind(),
                format!("prepare_output_dir: remove {}: {e}", path.display()),
            )
        })?;
    }

    Ok(())
}

pub fn save_metadata(output_path: &Path, outcome: &TaskOutcome) -> Result<()> {
    create_dir_all(output_path).map_err(|e| {
        Error::new(
            e.kind(),
            format!("save_metadata: create dir {}: {e}", output_path.display()),
        )
    })?;

    let file_path = output_path.join("metadata.json");
    let file = File::create(&file_path).map_err(|e| {
        Error::new(
            e.kind(),
            format!("save_metadata: create {}: {e}", file_path.display()),
        )
    })?;

    serde_json::to_writer_pretty(BufWriter::new(file), outcome).map_err(|e| {
        Error::new(
            ErrorKind::InvalidData,
            format!("save_metadata: serialize {}: {e}", file_path.display()),
        )
    })
}

pub fn load_metadata(path: &Path) -> Result<TaskOutcome> {
    let raw = read_to_string(path).map_err(|e| {
        Error::new(
            e.kind(),
            format!("load_metadata: read {}: {e}", path.display()),
        )
    })?;

    serde_json::from_str(&raw).map_err(|e| {
        Error::new(
            ErrorKind::InvalidData,
            format!("load_metadata: deserialize {}: {e}", path.display()),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solvers::termination::TerminationConfig;
    use crate::tasks::replicator_deterministic;
    use ndarray::Array2;
    use std::fs;

    #[test]
    fn deterministic_task_writes_metadata_matching_signal_files() {
        let output_path =
            std::env::temp_dir().join(format!("glv_metadata_integration_{}", std::process::id()));
        let _ = fs::remove_dir_all(&output_path);
        fs::create_dir_all(output_path.join("space")).expect("stale space dir");
        fs::write(output_path.join("space/1.json"), "{}").expect("stale space file");
        fs::write(output_path.join("metadata.json"), "{}").expect("stale metadata");

        let interaction = Array2::zeros((2, 2));
        let outcome = replicator_deterministic::run(
            &interaction,
            None,
            1e-12,
            0.01,
            4,
            2,
            &output_path,
            None,
            TerminationConfig::disabled(),
        )
        .expect("task succeeds");

        let metadata_path = output_path.join("metadata.json");
        let loaded = load_metadata(&metadata_path).expect("metadata loads");
        let signal_files = fs::read_dir(output_path.join("signal"))
            .expect("signal dir")
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("json"))
            .count();

        assert_eq!(outcome.steps_run, 4);
        assert_eq!(loaded.steps_run, 4);
        assert_eq!(loaded.end_time, 4);
        assert_eq!(loaded.signal.files, signal_files);
        assert_eq!(loaded.signal.files, outcome.signal.files);
        assert_eq!(loaded.signal.samples, 3);
        assert_eq!(loaded.num_species, 2);
        assert_eq!(loaded.model, "well_mixed_replicator");
        assert!(metadata_path.is_file());
        assert!(output_path.join("signal/1.json").is_file());
        assert!(!output_path.join("space").exists());

        let _ = fs::remove_dir_all(output_path);
    }
}
