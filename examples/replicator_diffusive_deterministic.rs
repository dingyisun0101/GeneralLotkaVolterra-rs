/*!
Deterministic diffusive replicator Cargo example.

Purpose:
    Builds a fixed three-strategy interaction system on a larger
    two-dimensional grid and runs a longer diffusive deterministic replicator
    task into
    `output/replicator_diffusive_deterministic`.
*/

mod common;

fn main() {
    use general_lotka_volterra_rs::solvers::spatial::rk4::{Boundary, Diffusion};
    use ndarray::{Array1, Array2};

    let interaction_matrix = Array2::from_shape_vec(
        (3, 3),
        vec![
            0.0, -1.0, 1.0, //
            1.0, 0.0, -1.0, //
            -1.0, 1.0, 0.0,
        ],
    )
    .expect("valid interaction matrix");
    let growth_vector = Array1::from_vec(vec![0.02, 0.00, -0.01]);

    let spatial_shape = [128, 128];
    let diffusion = Diffusion::unit_spacing(
        Array1::from_vec(vec![0.020, 0.015, 0.010]),
        spatial_shape.len(),
        Boundary::Periodic,
    );

    let output_path = std::path::Path::new("output/replicator_diffusive_deterministic");
    let cutoff = 1e-9;
    let dt = 0.02;
    let epoch_len = 5_000;
    let save_interval = 100;
    let num_epochs = 4;
    let progress =
        common::ExampleProgress::start("replicator_diffusive_deterministic", epoch_len, num_epochs);

    if let Err(err) = general_lotka_volterra_rs::tasks::replicator_diffusive_deterministic::run(
        &interaction_matrix,
        Some(&growth_vector),
        cutoff,
        &spatial_shape,
        &diffusion,
        dt,
        epoch_len,
        save_interval,
        num_epochs,
        output_path,
        Some(progress.counter.as_ref()),
    ) {
        eprintln!("replicator_diffusive_deterministic failed: {err}");
        std::process::exit(1);
    }

    progress.finish();
}
