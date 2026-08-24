use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AbundanceRepresentation, AggregateAbundance, CHECKPOINT_STREAM, SIGNAL_STREAM,
    SPACE_FIELD, SPACE_STREAM, SpatialAbundance, TOTAL_FIELD, TotalAbundance, load_state_schema,
    state_schema_path,
};
use physics_in_parallel::prelude::basic::Tensor;
use scientific_workflow::system_state::{SimulationTime, SystemState, SystemStateSchema};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn initial_time() -> SimulationTime {
    SimulationTime::from_iteration_and_physical_time(0, 0.0).expect("zero is a valid physical time")
}

fn assemble_state(
    schema: &SystemStateSchema,
    abundance: AggregateAbundance,
    space: SpatialAbundance,
    total: TotalAbundance,
) -> SystemState {
    let mut state = schema.create_empty_state(initial_time());
    assert!(
        state
            .insert_payload(ABUNDANCE_FIELD, abundance)
            .unwrap()
            .is_none()
    );
    assert!(state.insert_payload(SPACE_FIELD, space).unwrap().is_none());
    assert!(state.insert_payload(TOTAL_FIELD, total).unwrap().is_none());
    state
}

#[test]
fn canonical_schema_has_one_exact_round_trip() {
    let schema = load_state_schema().expect("canonical schema loads");
    let fields = schema.field_schemas();
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].name(), ABUNDANCE_FIELD);
    assert_eq!(fields[1].name(), SPACE_FIELD);
    assert_eq!(fields[2].name(), TOTAL_FIELD);

    let checked_in = fs::read_to_string(state_schema_path()).expect("schema file is readable");
    let generated = schema
        .to_json_template()
        .expect("schema serializes to its canonical JSON representation");
    assert_eq!(generated, checked_in.trim_end());

    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "glv-state-schema-round-trip-{}-{sequence}",
        std::process::id()
    ));
    fs::create_dir(&directory).expect("unique temporary directory is created");
    let path = directory.join("state.json");
    fs::write(&path, generated).expect("round-trip template is written");
    let restored = SystemStateSchema::load_json_template(&path).expect("round trip reloads");
    assert_eq!(restored.to_json_template().unwrap(), checked_in.trim_end());
    fs::remove_dir_all(directory).expect("temporary directory is removed");
}

#[test]
fn non_spatial_and_spatial_models_share_the_populated_space_slot() {
    let schema = load_state_schema().unwrap();

    let non_spatial = assemble_state(&schema, Tensor::from_vec(&[2], vec![0.25, 0.75]), None, 1.0);
    assert_eq!(non_spatial.populated_field_count(), 3);
    assert!(
        non_spatial
            .payload::<SpatialAbundance>(SPACE_FIELD)
            .unwrap()
            .is_none()
    );

    let space = Tensor::from_vec(&[2, 3, 2], vec![0.5; 12]);
    let spatial = assemble_state(
        &schema,
        Tensor::from_vec(&[2], vec![3.0, 3.0]),
        Some(space.clone()),
        6.0,
    );
    assert_eq!(spatial.populated_field_count(), 3);
    assert_eq!(
        spatial
            .payload::<SpatialAbundance>(SPACE_FIELD)
            .unwrap()
            .as_ref(),
        Some(&space)
    );
}

#[test]
fn coordinated_spatial_mutation_updates_abundance_space_and_total() {
    let schema = load_state_schema().unwrap();
    let mut state = assemble_state(
        &schema,
        Tensor::from_vec(&[2], vec![1.0, 2.0]),
        Some(Tensor::from_vec(&[1, 2], vec![1.0; 2])),
        3.0,
    );

    let (abundance, space, total) = state
        .borrow_payloads_mut::<(AggregateAbundance, SpatialAbundance, TotalAbundance)>((
            ABUNDANCE_FIELD,
            SPACE_FIELD,
            TOTAL_FIELD,
        ))
        .expect("distinct state fields support coordinated mutation");
    abundance.as_mut_slice()[0] = 4.0;
    space
        .as_mut()
        .expect("spatial state is present")
        .as_mut_slice()[0] = 4.0;
    *total = abundance.sum_serial();

    assert_eq!(
        state
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap()
            .as_slice()[0],
        4.0
    );
    assert_eq!(*state.payload::<TotalAbundance>(TOTAL_FIELD).unwrap(), 6.0);
}

#[test]
fn payload_insertion_and_extraction_preserve_tensor_ownership() {
    let schema = load_state_schema().unwrap();
    let abundance = Tensor::from_vec(&[3], vec![1.0, 2.0, 3.0]);
    let original_pointer = abundance.as_slice().as_ptr();
    let mut state = assemble_state(&schema, abundance, None, 6.0);

    let stored_pointer = state
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .expect("aggregate abundance is present")
        .as_slice()
        .as_ptr();
    assert_eq!(stored_pointer, original_pointer);

    let extracted = state
        .take_payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .expect("aggregate abundance can be extracted");
    assert_eq!(extracted.as_slice().as_ptr(), original_pointer);
    assert_eq!(extracted, Tensor::from_vec(&[3], vec![1.0, 2.0, 3.0]));
}

#[test]
fn canonical_stream_names_are_distinct() {
    assert_eq!(SIGNAL_STREAM, "signal");
    assert_eq!(SPACE_STREAM, "space");
    assert_eq!(CHECKPOINT_STREAM, "checkpoint");
    assert_ne!(SIGNAL_STREAM, SPACE_STREAM);
    assert_ne!(SIGNAL_STREAM, CHECKPOINT_STREAM);
    assert_ne!(SPACE_STREAM, CHECKPOINT_STREAM);
}

#[test]
fn abundance_representation_has_stable_metadata_values() {
    for (representation, expected) in [
        (
            AbundanceRepresentation::RelativeFrequency,
            "relative_frequency",
        ),
        (AbundanceRepresentation::AbsoluteCount, "absolute_count"),
    ] {
        assert_eq!(representation.as_str(), expected);
        assert_eq!(
            serde_json::to_string(&representation).unwrap(),
            format!("\"{expected}\"")
        );
        assert_eq!(
            serde_json::from_str::<AbundanceRepresentation>(&format!("\"{expected}\"")).unwrap(),
            representation
        );
    }
}
