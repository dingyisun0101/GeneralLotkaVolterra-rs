//! Spatial local-frequency replicator project.

use std::error::Error;
use std::path::PathBuf;

use general_lotka_volterra_rs::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let config = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config"));
    run(GlvTemplate::SpatialReplicator, config)?;
    Ok(())
}
