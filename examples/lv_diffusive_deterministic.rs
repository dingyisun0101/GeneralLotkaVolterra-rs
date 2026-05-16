/*!
Deterministic diffusive GLV Cargo example.

Purpose:
    Builds a fixed GLV system on a larger two-dimensional grid and runs a
    longer diffusive deterministic GLV task into
    `output/lv_diffusive_deterministic`.
*/

fn main() {
    use general_lotka_volterra_rs::solvers::spatial::rk4::{Boundary, Diffusion};
    use ndarray::{Array1, Array2};

    let interaction_matrix = Array2::from_shape_vec(
        (3, 3),
        vec![
            -0.70, -0.20, 0.10, //
            0.05, -0.60, -0.15, //
            -0.10, 0.10, -0.50,
        ],
    )
    .expect("valid interaction matrix");
    let growth_vector = Array1::from_vec(vec![0.35, 0.25, 0.20]);

    let spatial_shape = [128, 128];
    let diffusion = Diffusion::unit_spacing(
        Array1::from_vec(vec![0.025, 0.015, 0.010]),
        spatial_shape.len(),
        Boundary::Neumann,
    );

    let output_path = std::path::Path::new("output/lv_diffusive_deterministic");
    let cutoff = 1e-9;
    let carrying_capacity = Some(40_000.0);
    let initial_population = 0.25;
    let dt = 0.02;
    let epoch_len = 5_000;
    let save_interval = 100;
    let num_epochs = 4;

    if let Err(err) = general_lotka_volterra_rs::tasks::lv_diffusive_deterministic::run(
        &interaction_matrix,
        Some(&growth_vector),
        cutoff,
        carrying_capacity,
        &spatial_shape,
        initial_population,
        &diffusion,
        dt,
        epoch_len,
        save_interval,
        num_epochs,
        output_path,
        None,
    ) {
        eprintln!("lv_diffusive_deterministic failed: {err}");
        std::process::exit(1);
    }
}
