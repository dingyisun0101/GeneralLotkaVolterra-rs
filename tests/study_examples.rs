use std::path::PathBuf;

use general_lotka_volterra_rs::interaction::InteractionMatrix;
use general_lotka_volterra_rs::kernel::BoundaryCondition;
use general_lotka_volterra_rs::{ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD};
use general_lotka_volterra_rs::{GlvInputs, load_state_schema};
use general_lotka_volterra_rs::{INTERACTION_SOURCE_KEY, InteractionSource};
use physics_in_parallel::prelude::basic::RngConfig;
use scientific_workflow::prelude::basics::{SamplingInterval, StateStreamConfig};

fn example_root(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

#[test]
fn mean_field_example_is_a_complete_lazy_workflow_study() {
    let root = example_root("mean_field_replicator");
    let inputs = GlvInputs::load(&root).unwrap();
    assert!(root.join("Cargo.toml").is_file());
    assert!(root.join("README.md").is_file());
    assert!(root.join("src/main.rs").is_file());
    assert_eq!(inputs.combination_count(), 2);
    assert_eq!(
        load_state_schema()
            .unwrap()
            .field_schemas()
            .iter()
            .map(|field| field.name())
            .collect::<Vec<_>>(),
        [ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD]
    );

    let mut tasks = inputs.combinations();
    for (ordinal, expected_cutoff) in [0.0, 1e-8].into_iter().enumerate() {
        let task = tasks.next().unwrap();
        assert_eq!(task.ordinal(), ordinal as u64);
        assert_eq!(
            task.decode_value::<Vec<f64>>("/initial_condition").unwrap(),
            [0.5, 0.3, 0.2]
        );
        assert_eq!(
            task.decode_value::<Vec<f64>>("/growth").unwrap(),
            [0.0, 0.0, 0.0]
        );
        assert_eq!(
            task.decode_value::<f64>("/extinction_cutoff").unwrap(),
            expected_cutoff
        );
        assert_eq!(task.decode_value::<f64>("/time_step").unwrap(), 0.005);
        assert_eq!(
            task.decode_value::<u64>("/maximum_iterations").unwrap(),
            100
        );
        let recording = task
            .decode_value::<Vec<StateStreamConfig>>("/recordings")
            .unwrap();
        assert_eq!(recording.len(), 2);
        assert_eq!(
            recording[0].sampling_interval(),
            SamplingInterval::iterations(10).unwrap()
        );
        assert_eq!(
            recording[1].sampling_interval(),
            SamplingInterval::iterations(50).unwrap()
        );

        let input: InteractionSource = task.decode_value(INTERACTION_SOURCE_KEY).unwrap();
        let matrix =
            InteractionMatrix::load_json(task.resolve_path(&input.path_key).unwrap()).unwrap();
        assert_eq!(matrix.species(), 3);
        assert_eq!(
            task.resolve_path("recordings").unwrap(),
            example_root("mean_field_replicator").join("output")
        );
    }
    assert!(tasks.next().is_none());
}

#[test]
fn every_user_example_is_an_independent_glv_crate_and_study() {
    for name in [
        "mean_field_replicator",
        "mean_field_replicator_demographic",
        "spatial_replicator",
        "spatial_general_lotka_volterra",
    ] {
        let root = example_root(name);
        assert!(root.join("Cargo.toml").is_file(), "{name} manifest");
        assert!(root.join("README.md").is_file(), "{name} guide");
        assert!(root.join("src/main.rs").is_file(), "{name} binary");
        assert!(
            !root.join("config/state.json").exists(),
            "{name} uses GLV's crate-owned schema"
        );
        let inputs = GlvInputs::load(&root).unwrap();
        assert!(
            inputs.combination_count() > 0,
            "{name} has at least one configuration"
        );
        for configuration in inputs.combinations() {
            configuration
                .decode_value::<Vec<StateStreamConfig>>("/recordings")
                .unwrap();
            let input: InteractionSource =
                configuration.decode_value(INTERACTION_SOURCE_KEY).unwrap();
            assert!(
                configuration
                    .resolve_path(&input.path_key)
                    .unwrap()
                    .is_file()
            );
        }
    }
}

#[test]
fn domain_configuration_decodes_without_application_mirror_types() {
    assert_eq!(
        serde_json::from_str::<BoundaryCondition>("\"periodic\"").unwrap(),
        BoundaryCondition::Periodic
    );
    assert_eq!(
        serde_json::from_str::<BoundaryCondition>("\"neumann\"").unwrap(),
        BoundaryCondition::Neumann
    );

    let inputs = GlvInputs::load(example_root("mean_field_replicator_demographic")).unwrap();
    let seeds = inputs
        .combinations()
        .map(|task| task.decode_value::<RngConfig>("/rng").unwrap().seed())
        .collect::<Vec<_>>();
    assert_eq!(seeds, [Some(7), Some(11)]);
}
