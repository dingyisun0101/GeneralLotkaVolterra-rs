use std::path::PathBuf;

use general_lotka_volterra_rs::kernel::{InteractionSource, JsonInteractionSource};
use general_lotka_volterra_rs::{ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD};
use scientific_workflow::prelude::ScientificProject;
use serde::Deserialize;

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct StreamInputs {
    sampling_interval: u64,
    max_chunk_bytes: u64,
    queue_bytes: u64,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct RecordingInputs {
    signal: StreamInputs,
    space: StreamInputs,
    checkpoint: StreamInputs,
}

fn example_root(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

#[test]
fn mean_field_example_is_a_complete_lazy_workflow_project() {
    let project = ScientificProject::load(example_root("mean_field_replicator")).unwrap();
    assert_eq!(project.task_count(), 2);
    assert_eq!(
        project
            .state_schema()
            .field_schemas()
            .iter()
            .map(|field| field.name())
            .collect::<Vec<_>>(),
        [ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD]
    );

    let mut tasks = project.task_configs();
    for (ordinal, expected_cutoff) in [0.0, 1e-8].into_iter().enumerate() {
        let task = tasks.next().unwrap();
        assert_eq!(task.task_ordinal(), ordinal as u64);
        assert_eq!(
            task.decode_value::<Vec<f64>>("initial_abundance").unwrap(),
            [0.5, 0.3, 0.2]
        );
        assert_eq!(
            task.decode_value::<Vec<f64>>("growth").unwrap(),
            [0.0, 0.0, 0.0]
        );
        assert_eq!(task.decode_value::<f64>("cutoff").unwrap(), expected_cutoff);
        assert_eq!(
            task.decode_value::<f64>("physical_time_increment").unwrap(),
            0.005
        );
        assert_eq!(task.decode_value::<u64>("maximum_iterations").unwrap(), 100);
        let recording = task.decode_value::<RecordingInputs>("recording").unwrap();
        assert_eq!(recording.signal.sampling_interval, 10);
        assert_eq!(recording.space.sampling_interval, 25);
        assert_eq!(recording.checkpoint.sampling_interval, 50);

        let matrix =
            JsonInteractionSource::resolved_file(task.resolve_path("interaction_matrix").unwrap())
                .resolve(3)
                .unwrap();
        assert_eq!(matrix.species(), 3);
        assert_eq!(
            task.resolve_path("recordings").unwrap(),
            example_root("mean_field_replicator").join("output")
        );
    }
    assert!(tasks.next().is_none());
}
