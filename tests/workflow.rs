use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ecological_model_core::initial_state::{
    InitialStateArtifactReference, InitialStateRecipe, persist_initial_state,
};
use ecological_model_core::inputs::EcologicalInputs;
use ecological_model_core::interaction::{
    InteractionArtifactReference, InteractionMatrix, persist_interaction_matrix,
};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, TotalAbundance,
};
use physics_in_parallel::prelude::basic::{RngConfig, SquareLatticeConfig};
use scientific_workflow::persistence::{JsonPayloadDecoderRegistry, StoredStateSeriesReader};
use scientific_workflow::runtime::{TaskRunKind, execute};
use scientific_workflow::study::Study;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TempProject(PathBuf);

impl TempProject {
    fn new() -> Self {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("glv-workflow-{}-{sequence}", std::process::id()));
        let configs = root.join("wf_configs");
        fs::create_dir_all(&configs).unwrap();
        let inputs = prepared_inputs(&root.join("prepared"));
        fs::write(
            configs.join("study.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "seed": 73,
                "phases": {
                    "simulate": {
                        "tasks": [{"execution_unit": "glv"}]
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            configs.join("parameters.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "glv": {
                    "identity": "mean-field",
                    "inputs": inputs,
                    "model": {
                        "kind": "mean_field_replicator_demographic",
                        "growth": 0.0,
                        "extinction_cutoff": 0.0,
                        "time_step": 0.1,
                        "sigma": 0.0
                    },
                    "recording": {
                        "signal_interval": 1,
                        "space_interval": 1,
                        "checkpoint_interval": 1
                    },
                    "observation": {"mode": "terminal_only"},
                    "maximum_iterations": 2
                }
            }))
            .unwrap(),
        )
        .unwrap();
        Self(root)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempProject {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("failed to clean {}: {error}", self.0.display());
        }
    }
}

fn prepared_inputs(root: &Path) -> EcologicalInputs {
    let interaction = InteractionMatrix::from_rows(vec![vec![0.0, 0.0], vec![0.0, 0.0]]).unwrap();
    let interaction = persist_interaction_matrix(root, &interaction).unwrap();
    let initial = InitialStateRecipe::BalancedUniform {
        rng: RngConfig::new(Some(101), None),
    }
    .create(SquareLatticeConfig::periodic(&[8, 8]), 2)
    .unwrap();
    let initial = persist_initial_state(root, &initial).unwrap();
    EcologicalInputs::new(
        InteractionArtifactReference::new(root.to_path_buf(), interaction.into_descriptor()),
        InitialStateArtifactReference::new(root.to_path_buf(), initial.into_descriptor()),
    )
    .unwrap()
}

fn decoders() -> JsonPayloadDecoderRegistry {
    JsonPayloadDecoderRegistry::with_capacity(3)
        .with_json_field::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap()
        .with_json_field::<SpatialAbundance>(SPACE_FIELD)
        .unwrap()
        .with_json_field::<TotalAbundance>(TOTAL_FIELD)
        .unwrap()
}

#[test]
fn workflow_uses_prepared_inputs_and_records_requested_noise_seed() {
    let project = TempProject::new();
    let summary = execute(Study::load(project.path()).unwrap()).unwrap();
    let task = &summary.replicates()[0].phases()[0].tasks()[0];
    let TaskRunKind::ExecutionUnit {
        execution_unit,
        members,
    } = task.kind()
    else {
        panic!("expected execution-unit result");
    };
    assert_eq!(execution_unit.as_ref(), "glv");
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].identity(), "mean-field");
    assert_eq!(members[0].final_iteration(), 2);

    let reader = StoredStateSeriesReader::open_completed_recording(
        members[0].output_directory(),
        decoders(),
    )
    .unwrap();
    let final_state = reader.read_latest_state_from_stream("checkpoint").unwrap();
    assert_eq!(final_state.time().iteration(), 2);
    assert_eq!(
        final_state
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap()
            .as_slice(),
        &[0.5, 0.5]
    );
    assert!(
        final_state
            .payload::<SpatialAbundance>(SPACE_FIELD)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        *final_state.payload::<TotalAbundance>(TOTAL_FIELD).unwrap(),
        1.0
    );

    let requests = reader.user_metadata()["workflow"]["seed_derivation"]["requests"]
        .as_array()
        .unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["scope"], "member");
    assert_eq!(requests[0]["member_identity"], "mean-field");
    assert_eq!(requests[0]["purpose"], "noise");
    assert!(requests[0]["seed"].is_u64());

    let completion = &reader.terminal_metadata()["completion_reason"];
    assert_eq!(completion["kind"], "maximum_iterations");
    assert_eq!(completion["terminal_state"]["iteration"], 2);
    assert_eq!(
        completion["terminal_state"]["composition"],
        serde_json::json!([0.5, 0.5])
    );
}

#[test]
fn checked_in_examples_use_the_standard_provider_without_state_files() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
    for name in [
        "mean_field_replicator",
        "mean_field_replicator_demographic",
        "spatial_general_lotka_volterra",
        "spatial_replicator",
    ] {
        let project = repository.join("examples").join(name);
        assert!(
            !project.join("wf_configs/states").exists(),
            "{name} must obtain its schema from GlvUnit's standard provider"
        );
        Study::load(&project).unwrap_or_else(|error| {
            panic!("{name} must preflight through Workflow's public boundary: {error}")
        });
    }
}
