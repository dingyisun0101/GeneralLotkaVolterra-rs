/*!
Deterministic diffusive GLV Cargo example.

Purpose:
    Builds a fixed ten-strain GLV system on a larger two-dimensional grid and runs a
    longer diffusive deterministic GLV task into
    `output/lv_diffusive_deterministic`.
*/

mod common;
#[path = "common/constants.rs"]
mod constants;

fn main() {
    common::run_and_render(
        constants::LV_DIFFUSIVE_DETERMINISTIC_LABEL,
        constants::EPOCH_LEN,
        constants::NUM_EPOCHS,
        constants::lv_diffusive_deterministic_output_path(),
        constants::run_lv_diffusive_deterministic,
    );
}
