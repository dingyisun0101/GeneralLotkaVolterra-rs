/*!
Deterministic diffusive replicator Cargo example.

Purpose:
    Builds a fixed ten-strategy interaction system on a larger
    two-dimensional grid and runs a longer diffusive deterministic replicator
    task into
    `output/replicator_diffusive_deterministic`.
*/

mod common;
#[path = "common/constants.rs"]
mod constants;

fn main() {
    common::run_and_render(
        constants::REPLICATOR_DIFFUSIVE_DETERMINISTIC_LABEL,
        constants::TOTAL_STEPS,
        constants::replicator_diffusive_deterministic_output_path(),
        constants::run_replicator_diffusive_deterministic,
    );
}
