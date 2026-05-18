/*!
Deterministic replicator Cargo example.

Purpose:
    Builds a larger random interaction matrix and runs a longer deterministic
    trajectory into `output/replicator_deterministic`.
*/

mod common;
#[path = "common/constants.rs"]
mod constants;

fn main() {
    common::run_and_render(
        constants::REPLICATOR_DETERMINISTIC_LABEL,
        constants::TOTAL_STEPS,
        constants::replicator_deterministic_output_path(),
        constants::run_replicator_deterministic,
    );
}
