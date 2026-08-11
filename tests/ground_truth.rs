use general_lotka_volterra_rs::prelude::*;
use serde_json::Value;

const GROUND_TRUTH: &str = include_str!("fixtures/ground_truth.json");

fn fixture() -> Value {
    serde_json::from_str(GROUND_TRUTH).expect("ground-truth fixture is valid JSON")
}

fn values(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("fixture value is an array")
        .iter()
        .map(|value| value.as_f64().expect("fixture array value is finite f64"))
        .collect()
}

fn assert_close(actual: impl Iterator<Item = f64>, expected: &[f64], tolerance: f64) {
    let actual = actual.collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "ground-truth mismatch at {index}: {actual:.17e} != {expected:.17e}; error {error:.3e} > {tolerance:.3e}"
        );
    }
}

#[test]
fn mean_field_replicator_matches_independent_high_resolution_ground_truth() {
    let fixture = fixture();
    let case = &fixture["cases"]["mean_field"];
    let initial = values(&case["initial_abundance"]);
    let species = initial.len();
    let matrix = Array2::from_shape_vec((species, species), values(&case["interaction"])).unwrap();
    let interaction = InMemorySource::new(matrix).resolve(species).unwrap();
    let time_step = TimeStep::new(fixture["rust_time_step"].as_f64().unwrap()).unwrap();
    let mut simulation = MeanFieldReplicator::new(
        Array1::from_vec(initial),
        interaction,
        MeanFieldReplicatorConfig::new(
            Array1::from_vec(values(&case["growth"])),
            case["cutoff"].as_f64().unwrap(),
            time_step,
        ),
    )
    .unwrap();

    for _ in 0..fixture["rust_iterations"].as_u64().unwrap() {
        simulation.step().unwrap();
    }

    let abundance = simulation
        .state()
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap();
    assert_close(
        abundance.iter().copied(),
        &values(&case["expected"]),
        case["tolerance"].as_f64().unwrap(),
    );
}

#[test]
fn spatial_glv_matches_independent_ground_truth_with_and_without_diffusion() {
    let fixture = fixture();
    let time_step = TimeStep::new(fixture["rust_time_step"].as_f64().unwrap()).unwrap();
    for name in ["spatial_no_diffusion", "spatial_periodic_diffusion"] {
        let case = &fixture["cases"][name];
        let shape = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let species = *shape.last().unwrap();
        let matrix =
            Array2::from_shape_vec((species, species), values(&case["interaction"])).unwrap();
        let interaction = InMemorySource::new(matrix).resolve(species).unwrap();
        let initial =
            ArrayD::from_shape_vec(IxDyn(&shape), values(&case["initial_space"])).unwrap();
        let boundary: Boundary = serde_json::from_value(case["boundary"].clone()).unwrap();
        let diffusion = Diffusion::unit_spacing(
            Array1::from_vec(values(&case["diffusion"])),
            shape.len() - 1,
            boundary,
        )
        .unwrap();
        let mut simulation = SpatialGeneralLotkaVolterra::new(
            initial,
            interaction,
            SpatialGeneralLotkaVolterraConfig::new(
                shape,
                Array1::from_vec(values(&case["growth"])),
                diffusion,
                case["cutoff"].as_f64().unwrap(),
                None,
                time_step,
            ),
        )
        .unwrap();

        for _ in 0..fixture["rust_iterations"].as_u64().unwrap() {
            simulation.step().unwrap();
        }

        let space = simulation
            .state()
            .payload::<SpatialAbundance>(SPACE_FIELD)
            .unwrap()
            .as_ref()
            .unwrap();
        assert_close(
            space.iter().copied(),
            &values(&case["expected"]),
            case["tolerance"].as_f64().unwrap(),
        );
    }
}
