//src/tests/non_spatial_no_noise.rs

use std::fs::create_dir_all;
use std::path::PathBuf;

use ndarray::{Array1, Array2};

use crate::noise::Noise;
use crate::solver::Solver;

#[test]
fn no_noise() {
    //solver will write into {dir_output}/table/*.json
    let dir_output = PathBuf::from("tests/outputs/non_spatial/no_noise");
    let _ = create_dir_all(&dir_output);

    // Small deterministic toy model (d taxa)
    let d = 4usize;

    // Interaction matrix V (d×d)
    let v = Array2::<f64>::from_shape_vec(
        (d, d),
        vec![
            0.0, 1.0, -0.5, 0.2,
            -0.3, 0.0, 0.8, -0.1,
            0.4, -0.6, 0.0, 0.9,
            -0.2, 0.1, -0.7, 0.0,
        ],
    )
    .expect("shape");

    // Selection/payoff vector s (length d)
    let s = Array1::<f64>::from_vec(vec![0.1, 0.0, -0.05, 0.02]);

    // Initial simplex point (optional; if None, solver uses uniform)
    let nu_init = vec![0.7, 0.1, 0.1, 0.1];

    // Integration parameters
    let dt = 1e-3;
    let max_steps = 50_000usize;
    let save_interval = 10_000usize;
    let cutoff = 1e-15;

    // No noise
    let noise = Noise::none();

    // Run the solver (writes JSON outputs; no additional checks requested)
    let _final_table = Solver::new(
        &dir_output,
        &v,
        Some(&s),
        Some(&nu_init),
        dt,
        max_steps,
        noise,
        save_interval,
        cutoff,
    )
    .run();
}

#[test]
fn proportional_gaussian() {
    let dir_output = PathBuf::from("tests/outputs/non_spatial/proportional_gaussian");
    let _ = create_dir_all(&dir_output);

    let d = 4usize;

    let v = Array2::<f64>::from_shape_vec(
        (d, d),
        vec![
            0.0, 1.0, -0.5, 0.2,
            -0.3, 0.0, 0.8, -0.1,
            0.4, -0.6, 0.0, 0.9,
            -0.2, 0.1, -0.7, 0.0,
        ],
    )
    .expect("shape");

    let s = Array1::<f64>::from_vec(vec![0.1, 0.0, -0.05, 0.02]);
    let nu_init = vec![0.7, 0.1, 0.1, 0.1];

    let dt = 1e-3;
    let max_steps = 50_000usize;
    let save_interval = 10_000usize;
    let cutoff = 1e-15;

    // Proportional Gaussian noise
    let sigma = 0.05;
    let noise = Noise::proportional_gaussian(sigma);

    let _final_table = Solver::new(
        &dir_output,
        &v,
        Some(&s),
        Some(&nu_init),
        dt,
        max_steps,
        noise,
        save_interval,
        cutoff,
    )
    .run();
}


#[test]
fn demographic_gaussian() {
    let dir_output = PathBuf::from("tests/outputs/non_spatial/demographic_gaussian");
    let _ = create_dir_all(&dir_output);

    let d = 4usize;

    let v = Array2::<f64>::from_shape_vec(
        (d, d),
        vec![
            0.0, 1.0, -0.5, 0.2,
            -0.3, 0.0, 0.8, -0.1,
            0.4, -0.6, 0.0, 0.9,
            -0.2, 0.1, -0.7, 0.0,
        ],
    )
    .expect("shape");

    let s = Array1::<f64>::from_vec(vec![0.1, 0.0, -0.05, 0.02]);
    let nu_init = vec![0.7, 0.1, 0.1, 0.1];

    let dt = 1e-3;
    let max_steps = 50_000usize;
    let save_interval = 10_000usize;
    let cutoff = 1e-15;

    // Demographic Gaussian noise
    let sigma = 0.05;
    let noise = Noise::demographic_gaussian(sigma);

    let _final_table = Solver::new(
        &dir_output,
        &v,
        Some(&s),
        Some(&nu_init),
        dt,
        max_steps,
        noise,
        save_interval,
        cutoff,
    )
    .run();
}