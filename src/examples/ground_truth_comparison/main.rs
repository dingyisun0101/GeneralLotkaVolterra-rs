/*!
Cross-solver validation against SciPy.

Purpose:
    Runs small deterministic systems through this crate's public API and
    compares the final states against SciPy `solve_ivp` references computed by
    `reference_scipy.py`.

Run:
    cargo run --example ground_truth_comparison

Requirements:
    Python with numpy and scipy available on PATH as `python` or `python3`.
*/

use std::fs;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;
use std::process::Command;

use general_lotka_volterra_rs::io::space::load_space_series;
use general_lotka_volterra_rs::solvers::noise::Noise;
use general_lotka_volterra_rs::solvers::spatial::rk2::{Boundary, Diffusion};
use general_lotka_volterra_rs::solvers::termination::TerminationConfig;
use general_lotka_volterra_rs::solvers::{Dynamics, SolveConfig, Space, solve};
use general_lotka_volterra_rs::{Mode, SystemState};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use serde::Deserialize;

mod constants;
use constants::*;

#[derive(Deserialize)]
struct References {
    well_mixed_replicator: Vec<f64>,
    spatial_glv_no_diffusion: Vec<f64>,
    spatial_glv_periodic_diffusion: Vec<f64>,
}

fn main() -> Result<()> {
    let output_root = Path::new(OUTPUT);
    let _ = fs::remove_dir_all(output_root);
    fs::create_dir_all(output_root)?;

    let rust_replicator = run_well_mixed_replicator(output_root)?;
    let rust_glv_no_diffusion = run_spatial_glv_no_diffusion(output_root)?;
    let rust_glv_periodic_diffusion = run_spatial_glv_periodic_diffusion(output_root)?;
    check_unsupported_well_mixed_glv(output_root)?;

    let references = run_scipy_reference()?;

    assert_close(
        "well_mixed_replicator_vs_scipy",
        &rust_replicator,
        &references.well_mixed_replicator,
        REPLICATOR_TOLERANCE,
    )?;
    assert_close(
        "spatial_glv_no_diffusion_vs_scipy",
        &rust_glv_no_diffusion,
        &references.spatial_glv_no_diffusion,
        SPATIAL_GLV_TOLERANCE,
    )?;
    assert_close(
        "spatial_glv_periodic_diffusion_vs_scipy",
        &rust_glv_periodic_diffusion,
        &references.spatial_glv_periodic_diffusion,
        SPATIAL_GLV_TOLERANCE,
    )?;

    println!("ok unsupported_well_mixed_glv");
    Ok(())
}

fn run_well_mixed_replicator(output_root: &Path) -> Result<Vec<f64>> {
    let output_path = output_root.join("well_mixed_replicator");
    let interaction_matrix = Array2::from_shape_vec(
        (NUM_REPLICATOR_SPECIES, NUM_REPLICATOR_SPECIES),
        vec![0.0, 0.3, -0.2, -0.1, 0.0, 0.25, 0.15, -0.35, 0.0],
    )
    .expect("valid interaction shape");
    let growth_vector = Array1::from_vec(vec![0.04, -0.02, 0.01]);
    let initial_state = SystemState::from_arrays(
        Mode::Frequency { cutoff: None },
        0,
        Array1::from_vec(vec![0.2, 0.5, 0.3]),
        None,
    );

    let outcome = solve(
        initial_state,
        &interaction_matrix,
        Some(&growth_vector),
        SolveConfig {
            dynamics: Dynamics::Replicator,
            space: Space::None,
            noise: Noise::none(),
            dt: DT,
            num_steps: NUM_STEPS,
            save_signal_interval: SAVE_INTERVAL,
            output_path: &output_path,
            termination: TerminationConfig::disabled(),
        },
        None,
    )?;

    Ok(outcome.final_state.state.to_vec())
}

fn run_spatial_glv_no_diffusion(output_root: &Path) -> Result<Vec<f64>> {
    let output_path = output_root.join("spatial_glv_no_diffusion");
    let interaction_matrix = Array2::from_shape_vec(
        (NUM_GLV_SPECIES, NUM_GLV_SPECIES),
        vec![-0.40, 0.08, -0.05, -0.30],
    )
    .expect("valid interaction shape");
    let growth_vector = Array1::from_vec(vec![0.20, 0.10]);
    let diffusion = Diffusion::unit_spacing(Array1::zeros(NUM_GLV_SPECIES), 1, Boundary::Periodic);
    let initial_space = ArrayD::from_shape_vec(
        IxDyn(&[NUM_SPATIAL_CELLS, NUM_GLV_SPECIES]),
        vec![0.40, 0.20, 0.15, 0.50, 0.30, 0.25, 0.10, 0.35],
    )
    .expect("valid space shape");
    let initial_state = SystemState::from_arrays(
        Mode::Population {
            cutoff: None,
            carrying_capacity: None,
        },
        0,
        Array1::zeros(NUM_GLV_SPECIES),
        Some(initial_space),
    );

    solve(
        initial_state,
        &interaction_matrix,
        Some(&growth_vector),
        SolveConfig {
            dynamics: Dynamics::GlvPopulation,
            space: Space::spatial(&diffusion, SAVE_INTERVAL),
            noise: Noise::none(),
            dt: DT,
            num_steps: NUM_STEPS,
            save_signal_interval: SAVE_INTERVAL,
            output_path: &output_path,
            termination: TerminationConfig::disabled(),
        },
        None,
    )?;

    final_space_from_output(&output_path)
}

fn run_spatial_glv_periodic_diffusion(output_root: &Path) -> Result<Vec<f64>> {
    let output_path = output_root.join("spatial_glv_periodic_diffusion");
    let interaction_matrix = Array2::from_shape_vec(
        (NUM_GLV_SPECIES, NUM_GLV_SPECIES),
        vec![-0.30, 0.04, -0.02, -0.25],
    )
    .expect("valid interaction shape");
    let growth_vector = Array1::from_vec(vec![0.05, 0.02]);
    let diffusion =
        Diffusion::unit_spacing(Array1::from_vec(vec![0.01, 0.02]), 1, Boundary::Periodic);
    let initial_space = ArrayD::from_shape_vec(
        IxDyn(&[NUM_SPATIAL_CELLS, NUM_GLV_SPECIES]),
        vec![0.30, 0.10, 0.25, 0.20, 0.10, 0.40, 0.15, 0.30],
    )
    .expect("valid space shape");
    let initial_state = SystemState::from_arrays(
        Mode::Population {
            cutoff: None,
            carrying_capacity: None,
        },
        0,
        Array1::zeros(NUM_GLV_SPECIES),
        Some(initial_space),
    );

    solve(
        initial_state,
        &interaction_matrix,
        Some(&growth_vector),
        SolveConfig {
            dynamics: Dynamics::GlvPopulation,
            space: Space::spatial(&diffusion, SAVE_INTERVAL),
            noise: Noise::none(),
            dt: DT,
            num_steps: NUM_STEPS,
            save_signal_interval: SAVE_INTERVAL,
            output_path: &output_path,
            termination: TerminationConfig::disabled(),
        },
        None,
    )?;

    final_space_from_output(&output_path)
}

fn check_unsupported_well_mixed_glv(output_root: &Path) -> Result<()> {
    let interaction_matrix = Array2::zeros((NUM_GLV_SPECIES, NUM_GLV_SPECIES));
    let initial_state = SystemState::from_arrays(
        Mode::Population {
            cutoff: None,
            carrying_capacity: None,
        },
        0,
        Array1::from_vec(vec![0.5, 0.25]),
        None,
    );
    let output_path = output_root.join("unsupported_well_mixed_glv");

    match solve(
        initial_state,
        &interaction_matrix,
        None,
        SolveConfig {
            dynamics: Dynamics::GlvPopulation,
            space: Space::None,
            noise: Noise::none(),
            dt: DT,
            num_steps: NUM_STEPS,
            save_signal_interval: SAVE_INTERVAL,
            output_path: &output_path,
            termination: TerminationConfig::disabled(),
        },
        None,
    ) {
        Err(error) if error.kind() == ErrorKind::Unsupported => Ok(()),
        Err(error) => Err(Error::new(
            error.kind(),
            format!("expected Unsupported error, got {error}"),
        )),
        Ok(_) => Err(Error::other(
            "expected Unsupported error for well-mixed GLV, got success",
        )),
    }
}

fn final_space_from_output(output_path: &Path) -> Result<Vec<f64>> {
    let series = load_space_series(&output_path.join("space/1.json"))?;
    let final_sample = series
        .samples
        .last()
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "space output has no samples"))?;
    Ok(final_sample
        .space
        .as_slice_memory_order()
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "space output is not contiguous"))?
        .to_vec())
}

fn run_scipy_reference() -> Result<References> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script_path = manifest_dir.join("examples/ground_truth_comparison/reference_scipy.py");
    let status = run_python_script(manifest_dir, &script_path)?;
    if !status.success() {
        return Err(Error::other(format!(
            "SciPy reference script failed with status {status}; install numpy and scipy"
        )));
    }

    let reference_path = manifest_dir.join(OUTPUT).join("scipy_reference.json");
    let raw = fs::read_to_string(&reference_path).map_err(|error| {
        Error::new(
            error.kind(),
            format!("read SciPy reference {}: {error}", reference_path.display()),
        )
    })?;
    serde_json::from_str(&raw).map_err(|error| {
        Error::new(
            ErrorKind::InvalidData,
            format!(
                "parse SciPy reference {}: {error}",
                reference_path.display()
            ),
        )
    })
}

fn run_python_script(manifest_dir: &Path, script_path: &Path) -> Result<std::process::ExitStatus> {
    let mut last_not_found = None;
    for interpreter in ["python", "python3"] {
        match Command::new(interpreter)
            .current_dir(manifest_dir)
            .arg(script_path)
            .status()
        {
            Ok(status) => return Ok(status),
            Err(error) if error.kind() == ErrorKind::NotFound => last_not_found = Some(error),
            Err(error) => return Err(error),
        }
    }

    Err(Error::new(
        ErrorKind::NotFound,
        format!(
            "could not find python or python3 to run {}; last error: {}",
            script_path.display(),
            last_not_found
                .map(|error| error.to_string())
                .unwrap_or_else(|| "interpreter not found".to_string())
        ),
    ))
}

fn assert_close(name: &str, actual: &[f64], expected: &[f64], tolerance: f64) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "{name}: length mismatch: actual {}, expected {}",
                actual.len(),
                expected.len()
            ),
        ));
    }

    let max_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f64, f64::max);

    if max_error > tolerance {
        return Err(Error::other(format!(
            "{name}: max abs error {max_error:e} exceeds tolerance {tolerance:e}\nactual:   {actual:?}\nexpected: {expected:?}"
        )));
    }

    println!("ok {name}: max_abs_error={max_error:e}");
    Ok(())
}
