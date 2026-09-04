mod support;
use support::*;

use general_lotka_volterra_rs::invariant::{
    FrequencyInvariant, InvariantPolicyError, LocalFrequencyInvariant, PopulationInvariant,
    enforce_state, validate_state,
};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD,
    TotalAbundance, ecological_state_schema,
};
use physics_in_parallel::prelude::basic::Tensor;
use scientific_workflow::prelude::{StateTime, SystemState};

fn make_state(abundance: Vec<f64>, space: SpatialAbundance, total: f64) -> SystemState {
    let mut state = ecological_state_schema()
        .resolve()
        .unwrap()
        .create_empty_state(StateTime::from_iteration_and_physical_time(0, 0.0).unwrap());
    state
        .insert_payload(ABUNDANCE_FIELD, dense_tensor(&[abundance.len()], abundance))
        .unwrap();
    state.insert_payload(SPACE_FIELD, space).unwrap();
    state.insert_payload(TOTAL_FIELD, total).unwrap();
    state
}

fn space(shape: &[usize], values: Vec<f64>) -> Tensor<f64> {
    dense_tensor(shape, values)
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "{actual} != {expected}"
    );
}

#[test]
fn aggregate_frequency_repairs_nonfinite_cutoff_and_empty_mass() {
    let mut policy = FrequencyInvariant::new(4, 0.1).unwrap();
    let mut state = make_state(vec![f64::NAN, -1.0, 0.05, 0.95], None, 99.0);
    enforce_state(&mut policy, &mut state).unwrap();
    assert_eq!(
        state
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &dense_tensor(&[4], vec![0.0, 0.0, 0.0, 1.0])
    );
    assert_eq!(*state.payload::<TotalAbundance>(TOTAL_FIELD).unwrap(), 1.0);
    validate_state(&policy, &state).unwrap();

    let mut empty = make_state(vec![0.01, 0.02, 0.03, 0.04], None, 0.0);
    enforce_state(&mut policy, &mut empty).unwrap();
    assert_eq!(
        empty
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &dense_tensor(&[4], vec![0.25; 4])
    );
    assert!(matches!(
        FrequencyInvariant::new(4, f64::NAN),
        Err(InvariantPolicyError::InvalidCutoff { .. })
    ));
}

#[test]
fn local_frequency_normalizes_every_cell_and_refreshes_the_mean() {
    let mut policy = LocalFrequencyInvariant::new(3, 0.25).unwrap();
    let scratch_len = policy.scratch_len();
    let mut state = make_state(
        vec![9.0, 9.0, 9.0],
        Some(space(&[2, 3], vec![f64::NAN, 0.2, 0.8, -1.0, 0.0, 0.0])),
        99.0,
    );

    enforce_state(&mut policy, &mut state).unwrap();
    let spatial = state
        .payload::<SpatialAbundance>(SPACE_FIELD)
        .unwrap()
        .as_ref()
        .unwrap();
    assert_eq!(
        spatial,
        &space(
            &[2, 3],
            vec![0.0, 0.0, 1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        )
    );
    let abundance = state
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap();
    assert_close(abundance.as_slice()[0], 1.0 / 6.0);
    assert_close(abundance.as_slice()[1], 1.0 / 6.0);
    assert_close(abundance.as_slice()[2], 2.0 / 3.0);
    assert_eq!(*state.payload::<TotalAbundance>(TOTAL_FIELD).unwrap(), 1.0);
    assert_eq!(policy.scratch_len(), scratch_len);
    validate_state(&policy, &state).unwrap();
}

#[test]
fn population_enforces_capacity_and_preserves_rounded_total_convention() {
    let mut policy = PopulationInvariant::new(2, 0.5, Some(4.0)).unwrap();
    let scratch_len = policy.scratch_len();
    let mut state = make_state(
        vec![99.0, 99.0],
        Some(space(&[2, 2], vec![0.1, 1.9, f64::NAN, 3.0])),
        99.0,
    );

    enforce_state(&mut policy, &mut state).unwrap();
    let spatial = state
        .payload::<SpatialAbundance>(SPACE_FIELD)
        .unwrap()
        .as_ref()
        .unwrap();
    assert_close(spatial.sum_serial(), 4.0);
    assert_eq!(spatial.as_slice()[0], 0.0);
    assert_eq!(spatial.as_slice()[2], 0.0);
    let abundance = state
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap();
    assert_close(abundance.as_slice()[0], 0.0);
    assert_close(abundance.as_slice()[1], 4.0);
    assert_eq!(*state.payload::<TotalAbundance>(TOTAL_FIELD).unwrap(), 4.0);
    assert_eq!(policy.scratch_len(), scratch_len);
    validate_state(&policy, &state).unwrap();

    let mut rounded = PopulationInvariant::new(2, 0.0, None).unwrap();
    let mut rounded_state = make_state(vec![0.0, 0.0], Some(space(&[1, 2], vec![0.6, 0.6])), 0.0);
    enforce_state(&mut rounded, &mut rounded_state).unwrap();
    assert_close(
        rounded_state
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap()
            .sum_serial(),
        1.2,
    );
    assert_eq!(
        *rounded_state
            .payload::<TotalAbundance>(TOTAL_FIELD)
            .unwrap(),
        1.0
    );

    let mut zero_capacity = PopulationInvariant::new(2, 0.0, Some(0.0)).unwrap();
    let mut zero_state = make_state(vec![1.0, 1.0], Some(space(&[1, 2], vec![1.0, 1.0])), 2.0);
    enforce_state(&mut zero_capacity, &mut zero_state).unwrap();
    assert_eq!(
        zero_state
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &zero_tensor(&[2])
    );
    assert_eq!(
        zero_state
            .payload::<SpatialAbundance>(SPACE_FIELD)
            .unwrap()
            .as_ref()
            .unwrap(),
        &space(&[1, 2], vec![0.0, 0.0])
    );
    assert_eq!(
        *zero_state.payload::<TotalAbundance>(TOTAL_FIELD).unwrap(),
        0.0
    );

    assert!(matches!(
        PopulationInvariant::new(2, 0.0, Some(-1.0)),
        Err(InvariantPolicyError::InvalidCarryingCapacity { .. })
    ));
}

#[test]
fn invariant_domain_errors_happen_before_payload_mutation() {
    let mut policy = LocalFrequencyInvariant::new(2, 0.0).unwrap();
    let mut state = make_state(
        vec![0.5, 0.5],
        Some(space(&[2, 3], vec![1.0 / 3.0; 6])),
        1.0,
    );
    let abundance_before = state
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap()
        .clone();
    let space_before = state
        .payload::<SpatialAbundance>(SPACE_FIELD)
        .unwrap()
        .clone();

    assert!(enforce_state(&mut policy, &mut state).is_err());
    assert_eq!(
        state
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &abundance_before
    );
    assert_eq!(
        state.payload::<SpatialAbundance>(SPACE_FIELD).unwrap(),
        &space_before
    );
}
