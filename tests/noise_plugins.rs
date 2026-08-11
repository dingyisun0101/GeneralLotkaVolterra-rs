use std::mem::size_of;

use general_lotka_volterra_rs::noise::{
    DemographicGaussian, NoNoise, Noise, NoiseDomain, NoisePluginError, ProportionalGaussian,
};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, TimeStep,
    TotalAbundance, load_state_schema,
};
use ndarray::{Array1, ArrayD, IxDyn};
use scientific_workflow::system_state::{SimulationTime, SystemState};

fn state(abundance: Vec<f64>, space: SpatialAbundance, total: f64) -> SystemState {
    let mut state = load_state_schema()
        .unwrap()
        .create_empty_state(SimulationTime::from_iteration_and_physical_time(5, 2.0).unwrap());
    state
        .insert_payload(ABUNDANCE_FIELD, Array1::from_vec(abundance))
        .unwrap();
    state.insert_payload(SPACE_FIELD, space).unwrap();
    state.insert_payload(TOTAL_FIELD, total).unwrap();
    state
}

fn spatial(values: Vec<f64>) -> SpatialAbundance {
    Some(ArrayD::from_shape_vec(IxDyn(&[2, 3]), values).unwrap())
}

#[test]
fn no_noise_is_zero_sized_and_changes_nothing() {
    assert_eq!(size_of::<NoNoise>(), 0);
    let mut noise = Noise::new(NoNoise);
    let mut state = state(vec![0.25, 0.75], None, 1.0);
    let time = state.simulation_time();
    let abundance = state
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap()
        .clone();
    noise
        .apply(&mut state, TimeStep::new(0.1).unwrap())
        .unwrap();
    assert_eq!(state.simulation_time(), time);
    assert_eq!(
        state
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &abundance
    );
}

#[test]
fn demographic_noise_is_seeded_reproducible_and_reuses_scratch() {
    let domain = NoiseDomain::aggregate(3).unwrap();
    let algorithm_a = DemographicGaussian::new(0.2, 17, domain.clone()).unwrap();
    let algorithm_b = DemographicGaussian::new(0.2, 17, domain).unwrap();
    let capacities = algorithm_a.scratch_capacities();
    let mut noise_a = Noise::new(algorithm_a);
    let mut noise_b = Noise::new(algorithm_b);
    let mut state_a = state(vec![4.0, 9.0, 16.0], None, 29.0);
    let mut state_b = state(vec![4.0, 9.0, 16.0], None, 29.0);
    let initial_time = state_a.simulation_time();

    for _ in 0..3 {
        noise_a
            .apply(&mut state_a, TimeStep::new(0.05).unwrap())
            .unwrap();
        noise_b
            .apply(&mut state_b, TimeStep::new(0.05).unwrap())
            .unwrap();
    }
    assert_eq!(
        state_a
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        state_b
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap()
    );
    assert_ne!(
        state_a
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &Array1::from_vec(vec![4.0, 9.0, 16.0])
    );
    assert_eq!(noise_a.algorithm().scratch_capacities(), capacities);
    assert_eq!(state_a.simulation_time(), initial_time);
    assert_eq!(
        *state_a.payload::<TotalAbundance>(TOTAL_FIELD).unwrap(),
        29.0
    );
}

#[test]
fn proportional_spatial_noise_updates_only_space_reproducibly() {
    let domain = NoiseDomain::spatial(vec![2, 3]).unwrap();
    let mut noise_a = Noise::new(ProportionalGaussian::new(0.05, 99, domain.clone()).unwrap());
    let mut noise_b = Noise::new(ProportionalGaussian::new(0.05, 99, domain).unwrap());
    let mut state_a = state(
        vec![0.3, 0.3, 0.4],
        spatial(vec![0.2, 0.3, 0.5, 0.4, 0.2, 0.4]),
        1.0,
    );
    let mut state_b = state(
        vec![0.3, 0.3, 0.4],
        spatial(vec![0.2, 0.3, 0.5, 0.4, 0.2, 0.4]),
        1.0,
    );
    let abundance_before = state_a
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap()
        .clone();

    noise_a
        .apply(&mut state_a, TimeStep::new(0.1).unwrap())
        .unwrap();
    noise_b
        .apply(&mut state_b, TimeStep::new(0.1).unwrap())
        .unwrap();
    assert_eq!(
        state_a.payload::<SpatialAbundance>(SPACE_FIELD).unwrap(),
        state_b.payload::<SpatialAbundance>(SPACE_FIELD).unwrap()
    );
    assert_eq!(
        state_a
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &abundance_before
    );
    let spatial = state_a
        .payload::<SpatialAbundance>(SPACE_FIELD)
        .unwrap()
        .as_ref()
        .unwrap();
    for cell in spatial.as_slice().unwrap().chunks_exact(3) {
        assert!((cell.iter().sum::<f64>() - 1.0).abs() <= 1.0e-12);
    }
}

#[test]
fn invalid_noise_configuration_and_inputs_fail_before_mutation() {
    assert!(matches!(
        DemographicGaussian::new(-0.1, 1, NoiseDomain::aggregate(2).unwrap()),
        Err(NoisePluginError::InvalidSigma { .. })
    ));
    assert!(matches!(
        NoiseDomain::spatial(Vec::<usize>::new()),
        Err(NoisePluginError::MissingSpeciesAxis)
    ));

    let mut noise =
        Noise::new(ProportionalGaussian::new(0.1, 5, NoiseDomain::aggregate(2).unwrap()).unwrap());
    let mut state = state(vec![f64::NAN, 1.0], None, 1.0);
    let before = state
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap()
        .clone();
    assert!(
        noise
            .apply(&mut state, TimeStep::new(0.1).unwrap())
            .is_err()
    );
    let after = state
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap();
    assert!(before[0].is_nan() && after[0].is_nan());
    assert_eq!(after[1], before[1]);
}
