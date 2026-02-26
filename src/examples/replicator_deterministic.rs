pub fn run() {
    use ndarray::{Array2};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    const D: usize = 10;
    let mut rng = SmallRng::from_rng(&mut rand::rng());

    let interaction_matrix = Array2::from_shape_fn((D, D), |_| rng.random_range(-0.5..=0.5));
    let growth_vector = None;

    let output_path = std::path::Path::new("output/replicator_deterministic");
    let cutoff = 1e-5;
    let dt = 0.01;
    let epoch_len = 1000;
    let save_interval = 10;
    let num_epochs = 10;

    if let Err(err) = crate::tasks::replicator_deterministic::run(
        &interaction_matrix,
        growth_vector,
        cutoff,
        dt,
        epoch_len,
        save_interval,
        num_epochs,
        output_path,
        None,
    ) {
        eprintln!("replicator_deterministic failed: {err}");
        std::process::exit(1);
    }
}
