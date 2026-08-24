mod support;

use general_lotka_volterra_rs::invariant::FrequencyInvariant;
use general_lotka_volterra_rs::kernel::{
    BoundaryCondition, Diffusion, Kernel, KernelCore, MeanFieldReplicatorRk4,
};
use general_lotka_volterra_rs::noise::{DemographicGaussian, Noise, NoiseDomain};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AggregateAbundance, MeanFieldReplicator, MeanFieldReplicatorConfig,
    SPACE_FIELD, SpatialAbundance, SpatialGeneralLotkaVolterra, SpatialGeneralLotkaVolterraConfig,
    SpatialReplicator, SpatialReplicatorConfig, TOTAL_FIELD, TimeStep, TotalAbundance,
};
use physics_in_parallel::prelude::basic::{
    RandType, RngConfig, SquareLatticeConfig, Tensor, TensorRandFiller,
};
use scientific_workflow::prelude::basics::SystemState;
use support::interaction_from_array;

#[derive(Clone, Copy)]
enum NaiveBoundary {
    Periodic,
    Neumann,
}

#[derive(Clone, Copy)]
enum NaiveDynamics {
    Replicator,
    Glv,
}

fn tensor(values: &[f64]) -> Tensor<f64> {
    Tensor::from_vec(&[values.len()], values.to_vec())
}

fn interaction(
    species: usize,
    values: &[f64],
) -> general_lotka_volterra_rs::interaction::InteractionMatrix {
    interaction_from_array(physics_in_parallel::prelude::basic::DenseMatrix::from_vec(
        species,
        species,
        values.to_vec(),
    ))
    .unwrap()
}

fn matrix_vector(matrix: &[f64], species: usize, input: &[f64]) -> Vec<f64> {
    (0..species)
        .map(|row| {
            (0..species)
                .map(|column| matrix[row * species + column] * input[column])
                .sum()
        })
        .collect()
}

fn mean_field_rhs(abundance: &[f64], growth: &[f64], matrix: &[f64]) -> Vec<f64> {
    let species = abundance.len();
    let interactions = matrix_vector(matrix, species, abundance);
    let mean_fitness: f64 = (0..species)
        .map(|index| abundance[index] * (growth[index] + interactions[index]))
        .sum();
    (0..species)
        .map(|index| abundance[index] * (growth[index] + interactions[index] - mean_fitness))
        .collect()
}

fn naive_mean_field_rk4(abundance: &[f64], growth: &[f64], matrix: &[f64], dt: f64) -> Vec<f64> {
    let k1 = mean_field_rhs(abundance, growth, matrix);
    let temporary: Vec<_> = abundance
        .iter()
        .zip(&k1)
        .map(|(value, rate)| value + 0.5 * dt * rate)
        .collect();
    let k2 = mean_field_rhs(&temporary, growth, matrix);
    let temporary: Vec<_> = abundance
        .iter()
        .zip(&k2)
        .map(|(value, rate)| value + 0.5 * dt * rate)
        .collect();
    let k3 = mean_field_rhs(&temporary, growth, matrix);
    let temporary: Vec<_> = abundance
        .iter()
        .zip(&k3)
        .map(|(value, rate)| value + dt * rate)
        .collect();
    let k4 = mean_field_rhs(&temporary, growth, matrix);
    abundance
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let proposed =
                value + dt / 6.0 * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]);
            if proposed.is_finite() && proposed > 0.0 {
                proposed
            } else {
                0.0
            }
        })
        .collect()
}

fn enforce_frequency(values: &mut [f64], cutoff: f64) {
    for value in values.iter_mut() {
        if !value.is_finite() || *value <= 0.0 || *value < cutoff {
            *value = 0.0;
        }
    }
    let sum: f64 = values.iter().sum();
    if sum > 0.0 {
        values.iter_mut().for_each(|value| *value /= sum);
    } else {
        values.fill(1.0 / values.len() as f64);
    }
}

fn naive_mean_field_step(
    abundance: &mut Vec<f64>,
    growth: &[f64],
    matrix: &[f64],
    cutoff: f64,
    dt: f64,
) {
    *abundance = naive_mean_field_rk4(abundance, growth, matrix, dt);
    enforce_frequency(abundance, cutoff);
}

fn apply_naive_demographic_noise(abundance: &mut [f64], normal: &mut [f64], sigma: f64, dt: f64) {
    let denominator: f64 = abundance.iter().map(|value| value.sqrt()).sum();
    let eta_bar = if denominator > 0.0 {
        abundance
            .iter()
            .zip(normal.iter())
            .map(|(value, eta)| value.sqrt() * eta)
            .sum::<f64>()
            / denominator
    } else {
        0.0
    };
    let scale = sigma * dt.sqrt();
    for (value, eta) in abundance.iter().zip(normal.iter_mut()) {
        let proposed = value + scale * value.sqrt() * (*eta - eta_bar);
        *eta = if proposed.is_finite() && proposed > 0.0 {
            proposed
        } else {
            0.0
        };
    }
    abundance.copy_from_slice(normal);
}

fn normalized_neighbor(
    cell: usize,
    axis: usize,
    offset: isize,
    shape: &[usize],
    boundary: NaiveBoundary,
) -> usize {
    let stride: usize = shape[axis + 1..].iter().product();
    let coordinate = (cell / stride) % shape[axis];
    let side = shape[axis] as isize;
    let shifted = coordinate as isize + offset;
    let normalized = match boundary {
        NaiveBoundary::Periodic => shifted.rem_euclid(side),
        NaiveBoundary::Neumann => shifted.clamp(0, side - 1),
    } as usize;
    cell - coordinate * stride + normalized * stride
}

#[allow(clippy::too_many_arguments)]
fn naive_spatial_rhs(
    values: &[f64],
    shape: &[usize],
    spacing: &[f64],
    boundary: NaiveBoundary,
    growth: &[f64],
    matrix: &[f64],
    diffusion: &[f64],
    dynamics: NaiveDynamics,
) -> Vec<f64> {
    let species = growth.len();
    let cells: usize = shape.iter().product();
    let mut output = vec![0.0; values.len()];
    for cell in 0..cells {
        let base = cell * species;
        let local = &values[base..base + species];
        let interactions = matrix_vector(matrix, species, local);
        let mean_fitness = match dynamics {
            NaiveDynamics::Glv => 0.0,
            NaiveDynamics::Replicator => (0..species)
                .map(|index| local[index] * (growth[index] + interactions[index]))
                .sum(),
        };
        for component in 0..species {
            let mut laplacian = 0.0;
            for (axis, spacing) in spacing.iter().copied().enumerate() {
                let plus = normalized_neighbor(cell, axis, 1, shape, boundary);
                let minus = normalized_neighbor(cell, axis, -1, shape, boundary);
                laplacian += (values[plus * species + component]
                    + values[minus * species + component]
                    - 2.0 * local[component])
                    / spacing.powi(2);
            }
            let fitness = growth[component] + interactions[component];
            let reaction = match dynamics {
                NaiveDynamics::Glv => local[component] * fitness,
                NaiveDynamics::Replicator => local[component] * (fitness - mean_fitness),
            };
            output[base + component] = reaction + diffusion[component] * laplacian;
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn naive_spatial_rk2(
    values: &[f64],
    shape: &[usize],
    spacing: &[f64],
    boundary: NaiveBoundary,
    growth: &[f64],
    matrix: &[f64],
    diffusion: &[f64],
    dynamics: NaiveDynamics,
    dt: f64,
) -> Vec<f64> {
    let k1 = naive_spatial_rhs(
        values, shape, spacing, boundary, growth, matrix, diffusion, dynamics,
    );
    let midpoint: Vec<_> = values
        .iter()
        .zip(&k1)
        .map(|(value, rate)| value + 0.5 * dt * rate)
        .collect();
    let k2 = naive_spatial_rhs(
        &midpoint, shape, spacing, boundary, growth, matrix, diffusion, dynamics,
    );
    values
        .iter()
        .zip(k2)
        .map(|(value, rate)| value + dt * rate)
        .collect()
}

fn enforce_local_frequency(values: &mut [f64], species: usize, cutoff: f64) -> Vec<f64> {
    let cells = values.len() / species;
    let mut abundance = vec![0.0; species];
    for cell in values.chunks_exact_mut(species) {
        enforce_frequency(cell, cutoff);
        for (total, value) in abundance.iter_mut().zip(cell) {
            *total += *value;
        }
    }
    abundance
        .iter_mut()
        .for_each(|value| *value /= cells as f64);
    abundance
}

fn enforce_population(
    values: &mut [f64],
    species: usize,
    cutoff: f64,
    carrying_capacity: Option<f64>,
) -> (Vec<f64>, f64) {
    let mut abundance = vec![0.0; species];
    for cell in values.chunks_exact_mut(species) {
        for (component, value) in cell.iter_mut().enumerate() {
            if !value.is_finite() || *value <= 0.0 || *value < cutoff {
                *value = 0.0;
            }
            abundance[component] += *value;
        }
    }
    let mut exact_total: f64 = abundance.iter().sum();
    if let Some(capacity) = carrying_capacity {
        if capacity == 0.0 {
            values.fill(0.0);
            abundance.fill(0.0);
            exact_total = 0.0;
        } else if exact_total > capacity {
            let scale = capacity / exact_total;
            values.iter_mut().for_each(|value| *value *= scale);
            abundance.iter_mut().for_each(|value| *value *= scale);
            exact_total = capacity;
        }
    }
    (abundance, exact_total.round().max(0.0))
}

fn assert_close(label: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2.0e-12 * (1.0 + expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}] {actual:.17e} != {expected:.17e}"
        );
    }
}

fn assert_state(
    state: &SystemState,
    expected_abundance: &[f64],
    expected_space: Option<&[f64]>,
    expected_total: f64,
    expected_iteration: u64,
    dt: f64,
) {
    let abundance = state
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap();
    assert_close("abundance", abundance.as_slice(), expected_abundance);
    let space = state.payload::<SpatialAbundance>(SPACE_FIELD).unwrap();
    match (space.as_ref(), expected_space) {
        (Some(actual), Some(expected)) => assert_close("space", actual.as_slice(), expected),
        (None, None) => {}
        _ => panic!("spatial presence differs from naive reference"),
    }
    let total = *state.payload::<TotalAbundance>(TOTAL_FIELD).unwrap();
    assert!((total - expected_total).abs() <= 2.0e-12);
    assert_eq!(state.simulation_time().iteration(), expected_iteration);
    let expected_time = expected_iteration as f64 * dt;
    assert!((state.simulation_time().physical_time().unwrap() - expected_time).abs() <= 2.0e-12);
}

#[test]
fn deterministic_mean_field_matches_naive_solver_at_every_step() {
    let initial = [0.52, 0.31, 0.14, 0.03];
    let growth = [0.18, -0.07, 0.11, -0.02];
    let matrix = [
        -0.24, 0.31, -0.08, 0.12, -0.17, -0.19, 0.27, 0.04, 0.09, -0.22, -0.13, 0.25, 0.14, 0.06,
        -0.18, -0.21,
    ];
    let cutoff = 0.04;
    let dt = 0.0375;
    let mut simulation = MeanFieldReplicator::new(
        tensor(&initial),
        interaction(4, &matrix),
        MeanFieldReplicatorConfig::new(tensor(&growth), cutoff, TimeStep::new(dt).unwrap()),
    )
    .unwrap();
    let mut expected = initial.to_vec();

    for iteration in 1..=24 {
        naive_mean_field_step(&mut expected, &growth, &matrix, cutoff, dt);
        if iteration == 1 {
            assert_eq!(
                expected[3], 0.0,
                "mean-field cutoff branch was not exercised"
            );
        }
        simulation.step().unwrap();
        assert_state(simulation.state(), &expected, None, 1.0, iteration, dt);
    }
}

#[test]
fn demographic_mean_field_matches_naive_seeded_solver_at_every_step() {
    let initial = [0.43, 0.34, 0.23];
    let growth = [0.12, -0.04, 0.07];
    let matrix = [-0.2, 0.16, 0.08, -0.11, -0.17, 0.21, 0.05, -0.09, -0.14];
    let cutoff = 0.025;
    let sigma = 0.08;
    let dt = 0.025;
    let time_step = TimeStep::new(dt).unwrap();
    let state = MeanFieldReplicator::new(
        tensor(&initial),
        interaction(3, &matrix),
        MeanFieldReplicatorConfig::new(tensor(&growth), cutoff, time_step),
    )
    .unwrap()
    .into_state();
    let noise = DemographicGaussian::new(
        sigma,
        RngConfig::new(Some(8_675_309), None),
        NoiseDomain::aggregate(3).unwrap(),
    )
    .unwrap();
    let mut normal_filler = TensorRandFiller::try_new(
        RandType::Normal {
            mean: 0.0,
            std: 1.0,
        },
        noise.rng_config(),
    )
    .unwrap();
    let mut simulation = MeanFieldReplicator::from_plugins(
        state,
        Kernel::new(
            KernelCore::new(interaction(3, &matrix)),
            MeanFieldReplicatorRk4::new(tensor(&growth)).unwrap(),
        ),
        Noise::new(noise),
        FrequencyInvariant::new(3, cutoff).unwrap(),
        time_step,
    )
    .unwrap();
    let mut expected = initial.to_vec();
    let mut normal = vec![0.0; initial.len()];

    for iteration in 1..=20 {
        naive_mean_field_step(&mut expected, &growth, &matrix, cutoff, dt);
        let deterministic = expected.clone();
        normal_filler.try_fill_slice(&mut normal).unwrap();
        apply_naive_demographic_noise(&mut expected, &mut normal, sigma, dt);
        enforce_frequency(&mut expected, cutoff);
        assert_ne!(expected, deterministic, "demographic noise was not active");
        simulation.step().unwrap();
        assert_state(simulation.state(), &expected, None, 1.0, iteration, dt);
    }
}

#[test]
fn spatial_replicator_matches_naive_periodic_solver_at_every_step() {
    let shape = [2, 3];
    let species = 3;
    let full_shape = [2, 3, species];
    let spacing = [0.75, 1.4];
    let initial = [
        0.70, 0.28, 0.02, 0.18, 0.51, 0.31, 0.44, 0.13, 0.43, 0.09, 0.58, 0.33, 0.36, 0.39, 0.25,
        0.27, 0.68, 0.05,
    ];
    let growth = [0.20, -0.10, 0.05];
    let matrix = [-0.18, 0.23, -0.07, -0.12, -0.15, 0.19, 0.08, -0.21, -0.11];
    let diffusion = [0.08, 0.035, 0.06];
    let cutoff = 0.04;
    let dt = 0.07;
    let lattice =
        SquareLatticeConfig::try_new(&shape, BoundaryCondition::Periodic, Some(&spacing)).unwrap();
    let mut simulation = SpatialReplicator::new(
        Tensor::from_vec(&full_shape, initial.to_vec()),
        interaction(species, &matrix),
        SpatialReplicatorConfig::new(
            tensor(&growth),
            Diffusion::new(tensor(&diffusion), lattice).unwrap(),
            cutoff,
            TimeStep::new(dt).unwrap(),
        ),
    )
    .unwrap();
    let mut expected_space = initial.to_vec();

    for iteration in 1..=18 {
        expected_space = naive_spatial_rk2(
            &expected_space,
            &shape,
            &spacing,
            NaiveBoundary::Periodic,
            &growth,
            &matrix,
            &diffusion,
            NaiveDynamics::Replicator,
            dt,
        );
        let expected_abundance = enforce_local_frequency(&mut expected_space, species, cutoff);
        if iteration == 1 {
            assert_eq!(
                expected_space[2], 0.0,
                "local cutoff branch was not exercised"
            );
        }
        simulation.step().unwrap();
        assert_state(
            simulation.state(),
            &expected_abundance,
            Some(&expected_space),
            1.0,
            iteration,
            dt,
        );
    }
}

#[test]
fn spatial_glv_matches_naive_neumann_solver_at_every_step() {
    let shape = [3, 2];
    let species = 2;
    let full_shape = [3, 2, species];
    let spacing = [1.2, 0.85];
    let initial = [0.04, 1.0, 0.7, 0.5, 1.2, 0.3, 0.6, 0.9, 0.8, 0.2, 0.5, 0.7];
    let growth = [0.90, 0.55];
    let matrix = [-0.12, 0.04, 0.02, -0.10];
    let diffusion = [0.03, 0.06];
    let cutoff = 0.05;
    let carrying_capacity_value = 7.6;
    let carrying_capacity = Some(carrying_capacity_value);
    let dt = 0.08;
    let lattice =
        SquareLatticeConfig::try_new(&shape, BoundaryCondition::Neumann, Some(&spacing)).unwrap();
    let mut simulation = SpatialGeneralLotkaVolterra::new(
        Tensor::from_vec(&full_shape, initial.to_vec()),
        interaction(species, &matrix),
        SpatialGeneralLotkaVolterraConfig::new(
            tensor(&growth),
            Diffusion::new(tensor(&diffusion), lattice).unwrap(),
            cutoff,
            carrying_capacity,
            TimeStep::new(dt).unwrap(),
        ),
    )
    .unwrap();
    let mut expected_space = initial.to_vec();
    let mut capacity_was_active = false;

    for iteration in 1..=20 {
        expected_space = naive_spatial_rk2(
            &expected_space,
            &shape,
            &spacing,
            NaiveBoundary::Neumann,
            &growth,
            &matrix,
            &diffusion,
            NaiveDynamics::Glv,
            dt,
        );
        let (expected_abundance, expected_total) =
            enforce_population(&mut expected_space, species, cutoff, carrying_capacity);
        capacity_was_active |=
            (expected_abundance.iter().sum::<f64>() - carrying_capacity_value).abs() <= 2.0e-12;
        simulation.step().unwrap();
        assert_state(
            simulation.state(),
            &expected_abundance,
            Some(&expected_space),
            expected_total,
            iteration,
            dt,
        );
    }
    assert!(
        capacity_was_active,
        "carrying-capacity branch was not exercised"
    );
}
