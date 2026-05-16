/*!
Demographic-noise replicator Cargo example.

Purpose:
    Builds a larger random interaction matrix and runs a longer
    demographic-noise trajectory into `output/replicator_demographic`.
*/

mod common;

fn main() {
    use ndarray::Array2;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    const D: usize = 64;
    let mut rng = SmallRng::from_rng(&mut rand::rng());

    let interaction_matrix = Array2::from_shape_fn((D, D), |_| rng.random_range(-0.5..=0.5));
    let growth_vector = None;

    let output_path = std::path::Path::new("output/replicator_demographic");
    let cutoff = 1e-5;
    let sigma = 0.1;
    let dt = 0.005;
    let epoch_len = 50_000;
    let save_interval = 500;
    let num_epochs = 4;
    let progress = common::ExampleProgress::start("replicator_demographic", epoch_len, num_epochs);

    if let Err(err) = general_lotka_volterra_rs::tasks::replicator_demographic::run(
        &interaction_matrix,
        growth_vector,
        cutoff,
        sigma,
        dt,
        epoch_len,
        save_interval,
        num_epochs,
        output_path,
        Some(progress.counter.as_ref()),
    ) {
        eprintln!("replicator_demographic failed: {err}");
        std::process::exit(1);
    }

    progress.finish();
}
