/*!
Demographic-noise replicator Cargo example.

Purpose:
    Builds a larger random interaction matrix and runs a longer
    demographic-noise trajectory into `output/replicator_demographic`.
*/

mod common;
#[path = "common/constants.rs"]
mod constants;

fn main() {
    common::run_and_render(
        constants::REPLICATOR_DEMOGRAPHIC_LABEL,
        constants::TOTAL_STEPS,
        constants::replicator_demographic_output_path(),
        constants::run_replicator_demographic,
    );
}
