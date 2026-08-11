use general_lotka_volterra_rs::engine::Engine;
use general_lotka_volterra_rs::invariant::{
    FrequencyInvariant, LocalFrequencyInvariant, PopulationInvariant,
};
use general_lotka_volterra_rs::kernel::{
    Boundary, Diffusion, InMemorySource, InteractionSource, Kernel, KernelAlgorithmError,
    KernelCore, MeanFieldReplicatorRk4, SpatialGeneralLotkaVolterraRk2, SpatialLayout,
    SpatialReplicatorRk2,
};
use general_lotka_volterra_rs::noise::{NoNoise, Noise};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, TimeStep,
    TotalAbundance, load_state_schema,
};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use scientific_workflow::system_state::{SimulationTime, SystemState};
use serde_json::Value;

const LEGACY_FIXTURE: &str = include_str!("fixtures/legacy_baseline.json");

fn legacy_case(name: &str) -> Value {
    let fixture: Value = serde_json::from_str(LEGACY_FIXTURE).unwrap();
    fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == name)
        .unwrap()
        .clone()
}

fn values(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect()
}

fn state(abundance: Vec<f64>, space: SpatialAbundance, total: f64) -> SystemState {
    let time = SimulationTime::from_iteration_and_physical_time(0, 0.0).unwrap();
    let mut state = load_state_schema().unwrap().create_empty_state(time);
    state
        .insert_payload(ABUNDANCE_FIELD, Array1::from_vec(abundance))
        .unwrap();
    state.insert_payload(SPACE_FIELD, space).unwrap();
    state.insert_payload(TOTAL_FIELD, total).unwrap();
    state
}

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-12_f64.max(1.0e-12 * expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:.17e} != {expected:.17e} (tolerance {tolerance:.3e})"
    );
}

fn spatial_payload(inputs: &Value) -> SpatialAbundance {
    let shape = inputs["initial_space_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    Some(ArrayD::from_shape_vec(IxDyn(&shape), values(&inputs["initial_space_values"])).unwrap())
}

fn spatial_shape(inputs: &Value) -> Vec<usize> {
    inputs["initial_space_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect()
}

fn matrix(inputs: &Value, species: usize) -> KernelCore {
    let interaction =
        Array2::from_shape_vec((species, species), values(&inputs["interaction_values"])).unwrap();
    KernelCore::new(InMemorySource::new(interaction).resolve(species).unwrap())
}

fn diffusion(inputs: &Value) -> Diffusion {
    let boundary = match inputs["boundary"].as_str().unwrap() {
        "periodic" => Boundary::Periodic,
        "neumann" => Boundary::Neumann,
        value => panic!("unsupported fixture boundary {value}"),
    };
    Diffusion::new(
        Array1::from_vec(values(&inputs["diffusion"])),
        values(&inputs["spacing"]),
        boundary,
    )
    .unwrap()
}

fn assert_expected_spatial_state(state: &SystemState, expected: &Value) {
    assert_eq!(
        state.simulation_time().iteration(),
        expected["final_iteration"].as_u64().unwrap()
    );
    let abundance = state
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap();
    for (actual, expected) in abundance
        .iter()
        .copied()
        .zip(values(&expected["abundance"]))
    {
        assert_close(actual, expected);
    }
    let space = state
        .payload::<SpatialAbundance>(SPACE_FIELD)
        .unwrap()
        .as_ref()
        .unwrap();
    let expected_shape = expected["space"]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    assert_eq!(space.shape(), expected_shape);
    for (actual, expected) in space
        .iter()
        .copied()
        .zip(values(&expected["space"]["values"]))
    {
        assert_close(actual, expected);
    }
    assert_close(
        *state.payload::<TotalAbundance>(TOTAL_FIELD).unwrap(),
        expected["total"].as_f64().unwrap(),
    );
}

#[test]
fn mean_field_replicator_matches_legacy_well_mixed_trajectory() {
    let case = legacy_case("well_mixed_replicator");
    let inputs = &case["inputs"];
    let expected = &case["expected"];
    let species = inputs["growth"].as_array().unwrap().len();
    let interaction =
        Array2::from_shape_vec((species, species), values(&inputs["interaction_values"])).unwrap();
    let resolved = InMemorySource::new(interaction).resolve(species).unwrap();
    let algorithm =
        MeanFieldReplicatorRk4::new(Array1::from_vec(values(&inputs["growth"]))).unwrap();
    assert_eq!(algorithm.scratch_lengths(), [species; 8]);
    let kernel = Kernel::new(KernelCore::new(resolved), algorithm);
    let noise = Noise::new(NoNoise);
    let invariant = FrequencyInvariant::new(species, inputs["cutoff"].as_f64().unwrap()).unwrap();
    let time_step = TimeStep::new(inputs["dt"].as_f64().unwrap()).unwrap();
    let mut engine = Engine::new(
        state(values(&inputs["initial_abundance"]), None, 1.0),
        kernel,
        noise,
        invariant,
        time_step,
    )
    .unwrap();

    for _ in 0..inputs["steps"].as_u64().unwrap() {
        engine.step().unwrap();
    }

    assert_eq!(
        engine.state().simulation_time().iteration(),
        expected["final_iteration"].as_u64().unwrap()
    );
    let abundance = engine
        .state()
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap();
    for (actual, expected) in abundance
        .iter()
        .copied()
        .zip(values(&expected["abundance"]))
    {
        assert_close(actual, expected);
    }
    assert_close(
        *engine
            .state()
            .payload::<TotalAbundance>(TOTAL_FIELD)
            .unwrap(),
        expected["total"].as_f64().unwrap(),
    );
}

#[test]
fn spatial_replicator_matches_legacy_trajectory() {
    let case = legacy_case("spatial_replicator");
    let inputs = &case["inputs"];
    let species = inputs["growth"].as_array().unwrap().len();
    let shape = spatial_shape(inputs);
    let elements: usize = shape.iter().product();
    let algorithm = SpatialReplicatorRk2::new(
        shape,
        Array1::from_vec(values(&inputs["growth"])),
        diffusion(inputs),
    )
    .unwrap();
    assert_eq!(algorithm.scratch_lengths(), [elements; 3]);
    let mut engine = Engine::new(
        state(vec![0.35, 0.35, 0.3], spatial_payload(inputs), 1.0),
        Kernel::new(matrix(inputs, species), algorithm),
        Noise::new(NoNoise),
        LocalFrequencyInvariant::new(species, inputs["cutoff"].as_f64().unwrap()).unwrap(),
        TimeStep::new(inputs["dt"].as_f64().unwrap()).unwrap(),
    )
    .unwrap();

    for _ in 0..inputs["steps"].as_u64().unwrap() {
        engine.step().unwrap();
    }

    assert_expected_spatial_state(engine.state(), &case["expected"]);
}

#[test]
fn spatial_glv_matches_legacy_trajectory_and_rounded_total() {
    let case = legacy_case("spatial_glv");
    let inputs = &case["inputs"];
    let species = inputs["growth"].as_array().unwrap().len();
    let shape = spatial_shape(inputs);
    let elements: usize = shape.iter().product();
    let algorithm = SpatialGeneralLotkaVolterraRk2::new(
        shape,
        Array1::from_vec(values(&inputs["growth"])),
        diffusion(inputs),
    )
    .unwrap();
    assert_eq!(algorithm.scratch_lengths(), [elements; 3]);
    let mut engine = Engine::new(
        state(vec![3.5, 3.8], spatial_payload(inputs), 7.0),
        Kernel::new(matrix(inputs, species), algorithm),
        Noise::new(NoNoise),
        PopulationInvariant::new(
            species,
            inputs["cutoff"].as_f64().unwrap(),
            inputs["carrying_capacity"].as_f64(),
        )
        .unwrap(),
        TimeStep::new(inputs["dt"].as_f64().unwrap()).unwrap(),
    )
    .unwrap();

    for _ in 0..inputs["steps"].as_u64().unwrap() {
        engine.step().unwrap();
    }

    assert_expected_spatial_state(engine.state(), &case["expected"]);
    let exact_sum = engine
        .state()
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap()
        .sum();
    assert_close(exact_sum, 7.396962628613997);
    assert_eq!(
        *engine
            .state()
            .payload::<TotalAbundance>(TOTAL_FIELD)
            .unwrap(),
        exact_sum.round()
    );
}

#[test]
fn spatial_facilities_validate_layout_diffusion_and_stability() {
    let layout = SpatialLayout::new(vec![2, 3, 4]).unwrap();
    assert_eq!(layout.shape(), [2, 3, 4]);
    assert_eq!(layout.spatial_dimensions(), 2);
    assert_eq!(layout.species(), 4);
    assert_eq!(layout.cells(), 6);
    assert_eq!(layout.elements(), 24);
    assert!(matches!(
        SpatialLayout::new(vec![4]),
        Err(KernelAlgorithmError::SpatialRank)
    ));
    assert!(matches!(
        Diffusion::new(Array1::from_vec(vec![-0.1]), vec![1.0], Boundary::Neumann),
        Err(KernelAlgorithmError::InvalidDiffusion { .. })
    ));

    let diffusion =
        Diffusion::new(Array1::from_vec(vec![0.5]), vec![1.0], Boundary::Neumann).unwrap();
    assert_eq!(diffusion.boundary(), Boundary::Neumann);
    let algorithm =
        SpatialGeneralLotkaVolterraRk2::new(vec![2, 1], Array1::zeros(1), diffusion).unwrap();
    algorithm
        .validate_time_step(TimeStep::new(1.0).unwrap())
        .unwrap();
    assert!(matches!(
        algorithm.validate_time_step(TimeStep::new(1.01).unwrap()),
        Err(KernelAlgorithmError::UnstableTimeStep { .. })
    ));
}
