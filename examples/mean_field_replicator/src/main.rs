//! Complete configuration-only GLV Workflow project.

use general_lotka_volterra_rs as _;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The checked-in artifact references are relative to the project root.
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;
    // No state-schema setup belongs here: GlvUnit supplies Eco Core's schema.
    scientific_workflow::run(std::path::Path::new(env!("CARGO_MANIFEST_DIR")))?;
    Ok(())
}
