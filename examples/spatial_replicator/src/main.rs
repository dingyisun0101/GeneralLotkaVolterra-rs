//! Configuration-only spatial replicator project.

use general_lotka_volterra_rs as _;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;
    scientific_workflow::run(std::path::Path::new(env!("CARGO_MANIFEST_DIR")))?;
    Ok(())
}
