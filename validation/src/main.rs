use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use current_glv::invariant::FrequencyInvariant;
use current_glv::kernel::{
    Boundary as CurrentBoundary, Diffusion as CurrentDiffusion, InMemorySource, InteractionSource,
    Kernel, KernelCore, MeanFieldReplicatorRk4,
};
use current_glv::noise::{DemographicGaussian, Noise as CurrentNoise, NoiseDomain};
use current_glv::{
    ABUNDANCE_FIELD, AbundanceRepresentation, AggregateAbundance, MeanFieldReplicator,
    MeanFieldReplicatorConfig, SPACE_FIELD, SpatialAbundance, SpatialGeneralLotkaVolterra,
    SpatialGeneralLotkaVolterraConfig, SpatialReplicator, SpatialReplicatorConfig, TimeStep,
};
use legacy_glv::prelude as legacy;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde_json::{Value, json};

const ABSOLUTE_TOLERANCE: f64 = 1e-12;
const RELATIVE_TOLERANCE: f64 = 1e-12;

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("validation output directory argument is required")?;
    fs::create_dir_all(&output)?;

    let mut cases = Vec::new();
    cases.push(compare_case(
        "mean_field_replicator",
        &current_mean_field()?,
        &legacy_mean_field(&output)?,
    )?);
    cases.push(compare_case(
        "spatial_replicator",
        &current_spatial_replicator()?,
        &legacy_spatial_replicator(&output)?,
    )?);
    cases.push(compare_case(
        "spatial_general_lotka_volterra",
        &current_spatial_glv()?,
        &legacy_spatial_glv(&output)?,
    )?);
    for seed in 0..8 {
        cases.push(compare_case(
            &format!("demographic_gaussian_seed_{seed}"),
            &current_demographic(seed)?,
            &legacy_demographic(seed),
        )?);
    }

    let report = json!({
        "legacy_commit": "5ad7cad1ade361e4ee40e540db72d602565e15e8",
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "status": "passed",
        "cases": cases,
    });
    fs::write(
        output.join("legacy-comparison.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn current_mean_field() -> Result<Vec<f64>, Box<dyn Error>> {
    let mut simulation = MeanFieldReplicator::new(
        Array1::from_vec(vec![0.2, 0.3, 0.5]),
        current_interaction(3, mean_field_interaction())?,
        MeanFieldReplicatorConfig::new(
            Array1::from_vec(vec![0.02, -0.01, 0.03]),
            1e-12,
            TimeStep::new(0.01)?,
        ),
    )?;
    for _ in 0..7 {
        simulation.step()?;
    }
    Ok(simulation
        .state()
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)?
        .to_vec())
}

fn legacy_mean_field(output: &Path) -> Result<Vec<f64>, Box<dyn Error>> {
    let directory = output.join("legacy-mean-field");
    let state = legacy::SystemState::from_arrays(
        legacy::Mode::Frequency {
            cutoff: Some(1e-12),
        },
        0,
        Array1::from_vec(vec![0.2, 0.3, 0.5]),
        None,
    );
    let outcome = legacy::solve(
        state,
        &Array2::from_shape_vec((3, 3), mean_field_interaction())?,
        Some(&Array1::from_vec(vec![0.02, -0.01, 0.03])),
        legacy::SolveConfig {
            dynamics: legacy::Dynamics::Replicator,
            space: legacy::Space::None,
            noise: legacy::Noise::none(),
            dt: 0.01,
            num_steps: 7,
            save_signal_interval: 7,
            output_path: &directory,
            termination: legacy::TerminationConfig::disabled(),
        },
        None,
    )?;
    Ok(outcome.final_state.state.to_vec())
}

fn current_spatial_replicator() -> Result<Vec<f64>, Box<dyn Error>> {
    let shape = vec![2, 2, 3];
    let mut simulation = SpatialReplicator::new(
        ArrayD::from_shape_vec(IxDyn(&shape), spatial_replicator_initial())?,
        current_interaction(3, mean_field_interaction())?,
        SpatialReplicatorConfig::new(
            shape,
            Array1::from_vec(vec![0.01, -0.02, 0.015]),
            CurrentDiffusion::new(
                Array1::from_vec(vec![0.03, 0.02, 0.01]),
                vec![1.0, 1.0],
                CurrentBoundary::Periodic,
            )?,
            1e-12,
            TimeStep::new(0.005)?,
        ),
    )?;
    for _ in 0..5 {
        simulation.step()?;
    }
    Ok(simulation
        .state()
        .payload::<SpatialAbundance>(SPACE_FIELD)?
        .as_ref()
        .expect("spatial simulation retains space")
        .iter()
        .copied()
        .collect())
}

fn legacy_spatial_replicator(output: &Path) -> Result<Vec<f64>, Box<dyn Error>> {
    let directory = output.join("legacy-spatial-replicator");
    let shape = [2, 2, 3];
    let state = legacy::SystemState::from_arrays(
        legacy::Mode::Frequency {
            cutoff: Some(1e-12),
        },
        0,
        Array1::zeros(3),
        Some(ArrayD::from_shape_vec(
            IxDyn(&shape),
            spatial_replicator_initial(),
        )?),
    );
    let diffusion = legacy::Diffusion {
        coefficients: Array1::from_vec(vec![0.03, 0.02, 0.01]),
        spacing: vec![1.0, 1.0],
        boundary: legacy::Boundary::Periodic,
    };
    legacy::solve(
        state,
        &Array2::from_shape_vec((3, 3), mean_field_interaction())?,
        Some(&Array1::from_vec(vec![0.01, -0.02, 0.015])),
        legacy::SolveConfig {
            dynamics: legacy::Dynamics::Replicator,
            space: legacy::Space::spatial(&diffusion, 5),
            noise: legacy::Noise::none(),
            dt: 0.005,
            num_steps: 5,
            save_signal_interval: 5,
            output_path: &directory,
            termination: legacy::TerminationConfig::disabled(),
        },
        None,
    )?;
    legacy_final_space(&directory)
}

fn current_spatial_glv() -> Result<Vec<f64>, Box<dyn Error>> {
    let shape = vec![2, 2, 2];
    let mut simulation = SpatialGeneralLotkaVolterra::new(
        ArrayD::from_shape_vec(IxDyn(&shape), spatial_glv_initial())?,
        current_interaction(2, spatial_glv_interaction())?,
        SpatialGeneralLotkaVolterraConfig::new(
            shape,
            Array1::from_vec(vec![0.6, 0.5]),
            CurrentDiffusion::new(
                Array1::from_vec(vec![0.02, 0.03]),
                vec![1.0, 1.0],
                CurrentBoundary::Periodic,
            )?,
            1e-12,
            None,
            TimeStep::new(0.01)?,
        ),
    )?;
    for _ in 0..5 {
        simulation.step()?;
    }
    Ok(simulation
        .state()
        .payload::<SpatialAbundance>(SPACE_FIELD)?
        .as_ref()
        .expect("spatial simulation retains space")
        .iter()
        .copied()
        .collect())
}

fn legacy_spatial_glv(output: &Path) -> Result<Vec<f64>, Box<dyn Error>> {
    let directory = output.join("legacy-spatial-glv");
    let shape = [2, 2, 2];
    let state = legacy::SystemState::from_arrays(
        legacy::Mode::Population {
            cutoff: Some(1e-12),
            carrying_capacity: None,
        },
        0,
        Array1::zeros(2),
        Some(ArrayD::from_shape_vec(
            IxDyn(&shape),
            spatial_glv_initial(),
        )?),
    );
    let diffusion = legacy::Diffusion {
        coefficients: Array1::from_vec(vec![0.02, 0.03]),
        spacing: vec![1.0, 1.0],
        boundary: legacy::Boundary::Periodic,
    };
    legacy::solve(
        state,
        &Array2::from_shape_vec((2, 2), spatial_glv_interaction())?,
        Some(&Array1::from_vec(vec![0.6, 0.5])),
        legacy::SolveConfig {
            dynamics: legacy::Dynamics::GlvPopulation,
            space: legacy::Space::spatial(&diffusion, 5),
            noise: legacy::Noise::none(),
            dt: 0.01,
            num_steps: 5,
            save_signal_interval: 5,
            output_path: &directory,
            termination: legacy::TerminationConfig::disabled(),
        },
        None,
    )?;
    legacy_final_space(&directory)
}

fn current_demographic(seed: u64) -> Result<Vec<f64>, Box<dyn Error>> {
    let species = 3;
    let time_step = TimeStep::new(0.005)?;
    let interaction = current_interaction(species, vec![0.0; species * species])?;
    let initial_state = MeanFieldReplicator::new(
        Array1::from_vec(vec![0.5, 0.3, 0.2]),
        interaction.clone(),
        MeanFieldReplicatorConfig::new(Array1::zeros(species), 0.0, time_step),
    )?
    .into_state();
    let mut simulation = MeanFieldReplicator::from_plugins(
        initial_state,
        AbundanceRepresentation::RelativeFrequency,
        Kernel::new(
            KernelCore::new(interaction),
            MeanFieldReplicatorRk4::new(Array1::zeros(species))?,
        ),
        CurrentNoise::new(DemographicGaussian::new(
            0.1,
            seed,
            NoiseDomain::aggregate(species)?,
        )?),
        FrequencyInvariant::new(species, 0.0)?,
        time_step,
    )?;
    simulation.step()?;
    Ok(simulation
        .state()
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)?
        .to_vec())
}

fn legacy_demographic(seed: u64) -> Vec<f64> {
    let mut state = legacy::SystemState::from_arrays(
        legacy::Mode::Frequency { cutoff: Some(0.0) },
        0,
        Array1::from_vec(vec![0.5, 0.3, 0.2]),
        None,
    );
    let mut context = legacy::NoiseContext::new(3);
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    legacy_glv::solvers::non_spatial::noise::apply_noise_inplace(
        &mut state,
        legacy::Noise::demographic_gaussian(0.1),
        0.005,
        &mut context,
        &mut rng,
    );
    state.state.to_vec()
}

fn current_interaction(
    species: usize,
    values: Vec<f64>,
) -> Result<current_glv::kernel::InteractionMatrix, Box<dyn Error>> {
    Ok(
        InMemorySource::new(Array2::from_shape_vec((species, species), values)?)
            .resolve(species)?,
    )
}

fn legacy_final_space(directory: &Path) -> Result<Vec<f64>, Box<dyn Error>> {
    let series = legacy::load_space_series(&directory.join("space/1.json"))?;
    Ok(series
        .samples
        .last()
        .expect("legacy final spatial sample exists")
        .space
        .iter()
        .copied()
        .collect())
}

fn compare_case(name: &str, current: &[f64], legacy: &[f64]) -> Result<Value, Box<dyn Error>> {
    if current.len() != legacy.len() {
        return Err(format!("{name}: length {} != {}", current.len(), legacy.len()).into());
    }
    let mut maximum_absolute_error = 0.0_f64;
    let mut maximum_relative_error = 0.0_f64;
    for (&current, &legacy) in current.iter().zip(legacy) {
        let absolute = (current - legacy).abs();
        let relative = absolute / legacy.abs().max(ABSOLUTE_TOLERANCE);
        maximum_absolute_error = maximum_absolute_error.max(absolute);
        maximum_relative_error = maximum_relative_error.max(relative);
        let tolerance = ABSOLUTE_TOLERANCE.max(RELATIVE_TOLERANCE * legacy.abs());
        if absolute > tolerance {
            return Err(format!(
                "{name}: {current:.17e} != {legacy:.17e}; error {absolute:.3e} > {tolerance:.3e}"
            )
            .into());
        }
    }
    Ok(json!({
        "name": name,
        "values": current.len(),
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
    }))
}

fn mean_field_interaction() -> Vec<f64> {
    vec![-0.2, 0.3, -0.1, 0.1, -0.15, 0.2, 0.25, -0.2, -0.1]
}

fn spatial_replicator_initial() -> Vec<f64> {
    vec![0.7, 0.2, 0.1, 0.1, 0.7, 0.2, 0.2, 0.2, 0.6, 0.4, 0.3, 0.3]
}

fn spatial_glv_interaction() -> Vec<f64> {
    vec![-0.4, 0.1, 0.05, -0.3]
}

fn spatial_glv_initial() -> Vec<f64> {
    vec![1.0, 0.5, 0.8, 1.2, 1.1, 0.7, 0.6, 1.4]
}
