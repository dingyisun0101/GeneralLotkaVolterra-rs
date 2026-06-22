/*!
Deterministic spatial GLV reaction-diffusion example.

Purpose:
    Demonstrates direct use of the unified solver API with a species-last
    spatial population field and deterministic GLV dynamics.
*/

use std::io::{Error, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

use general_lotka_volterra_rs::solvers::noise::Noise;
use general_lotka_volterra_rs::solvers::spatial::rk4::{Boundary, Diffusion};
use general_lotka_volterra_rs::solvers::termination::TerminationConfig;
use general_lotka_volterra_rs::solvers::{Dynamics, SolveConfig, Space, solve};
use general_lotka_volterra_rs::tasks::metadata::{
    TaskOutcome, output_label, prepare_output_dir, save_metadata,
};
use general_lotka_volterra_rs::utils::create_uniform_spatial_population_gs;
use ndarray::{Array1, Array2};

mod constants;
use constants::*;

fn main() -> Result<()> {
    // Choose where solver output should be written. Spatial examples write an
    // aggregate `signal/` stream and a full-field `space/` stream.
    let output_path = Path::new(OUTPUT);

    // Start from a clean output directory. This removes stale `signal/`,
    // `space/`, and `metadata.json` entries from previous runs.
    prepare_output_dir(output_path)?;

    // V controls local GLV interactions inside each spatial cell.
    let interaction_matrix = interaction_matrix();

    // The growth vector g sets the intrinsic growth rate for each species.
    let growth_vector = growth_vector();

    // Diffusion controls movement/smoothing across the spatial axes. The
    // species axis is not diffused; it is always the final axis.
    let diffusion = diffusion();

    // The spatial GLV state is a species-last population field:
    // space[x, y, species]. Values are absolute densities, not frequencies.
    let initial_state = create_uniform_spatial_population_gs(
        Some(CUTOFF),
        CARRYING_CAPACITY,
        &SPATIAL_SHAPE,
        NUM_STRAINS,
        INITIAL_POPULATION,
    );

    // Termination is explicit. This checks for monoculture at the same cadence
    // as saved samples. Use `TerminationConfig::disabled()` to force all steps.
    let termination = TerminationConfig::monoculture_only(SAVE_INTERVAL);

    // `solve` is the main user-facing API. Spatial and non-spatial runs use the
    // same function; the `Space` config decides which backend is used.
    let outcome = solve(
        initial_state,
        &interaction_matrix,
        Some(&growth_vector),
        // `SolveConfig` selects GLV population dynamics, spatial domain,
        // deterministic noise policy, time step, save cadence, and termination.
        SolveConfig {
            dynamics: Dynamics::GlvPopulation,
            // `SAVE_INTERVAL` is used for both aggregate signal snapshots and
            // full spatial field snapshots in this example.
            space: Space::spatial(&diffusion, SAVE_INTERVAL),
            noise: Noise::none(),
            dt: DT,
            num_steps: TOTAL_STEPS,
            save_signal_interval: SAVE_INTERVAL,
            output_path,
            termination,
        },
        None,
    )?;

    // The solver returns separate writer stats for `signal/` and `space/`.
    // `TaskOutcome::spatial` stores those stats plus model dimensions,
    // carrying-capacity settings, and termination metadata in `metadata.json`.
    let task_outcome = TaskOutcome::spatial(
        LABEL,
        "spatial_glv",
        &output_label(output_path),
        TOTAL_STEPS,
        DT,
        SAVE_INTERVAL,
        outcome.steps_run,
        outcome.reason,
        outcome.signal_stats,
        outcome.space_stats.unwrap_or_default(),
        NUM_STRAINS,
        &SPATIAL_SHAPE,
        Some(CUTOFF),
        CARRYING_CAPACITY,
        termination.survivor_tolerance,
    );
    save_metadata(output_path, &task_outcome)?;
    print_summary(output_path, &task_outcome);

    // Plotting is intentionally outside the solver. The Rust simulation has
    // already completed by this point; the Python renderer reads saved JSON.
    let plot_path = render_output_plot(output_path, LABEL)?;
    println!("plot: {}", plot_path.display());
    Ok(())
}

/// Build the GLV interaction matrix V.
///
/// The diagonal is negative self-limitation. Off-diagonal entries implement a
/// small cyclic pairwise interaction whose strength decays with index distance.
fn interaction_matrix() -> Array2<f64> {
    Array2::from_shape_fn((NUM_STRAINS, NUM_STRAINS), |(i, j)| {
        if i == j {
            SELF_LIMITATION
        } else {
            let direction = if (j + NUM_STRAINS - i) % NUM_STRAINS <= NUM_STRAINS / 2 {
                -1.0
            } else {
                1.0
            };
            direction * PAIR_INTERACTION / (1.0 + i.abs_diff(j) as f64)
        }
    })
}

/// Build a simple descending growth vector g.
fn growth_vector() -> Array1<f64> {
    Array1::from_shape_fn(NUM_STRAINS, |i| GROWTH_BASE - GROWTH_STEP * i as f64)
}

/// Build per-species diffusion coefficients and choose boundary behavior.
///
/// `Boundary::Neumann` reuses the edge value beyond each boundary, producing a
/// zero-flux condition. Use `Boundary::Periodic` for wrapped domains.
fn diffusion() -> Diffusion {
    Diffusion::unit_spacing(
        Array1::from_shape_fn(NUM_STRAINS, |i| {
            (DIFFUSION_BASE - DIFFUSION_STEP * i as f64).max(0.001)
        }),
        SPATIAL_SHAPE.len(),
        Boundary::Neumann,
    )
}

/// Print a compact run summary after metadata is saved.
fn print_summary(output_path: &Path, outcome: &TaskOutcome) {
    println!(
        "steps: {} / {}; reason: {:?}; signal files: {}; space files: {}; metadata: {}",
        outcome.steps_run,
        outcome.requested_steps,
        outcome.termination_reason,
        outcome.signal.files,
        outcome.space.as_ref().map(|space| space.files).unwrap_or(0),
        output_path.join("metadata.json").display()
    );
}

/// Render `output/<example>/signal/*.json` into `plot/plot.png`.
///
/// The current renderer plots aggregate signal history. Full spatial snapshots
/// remain available under `output/<example>/space/` for custom analysis.
fn render_output_plot(output_path: &Path, title: &str) -> Result<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output_path = output_path.canonicalize()?;
    let outdir = output_path.join("plot");
    let status = Command::new("python")
        .current_dir(manifest_dir)
        .args([
            "-m",
            "examples.plotting.render_from_output",
            output_path
                .to_str()
                .ok_or_else(|| Error::other("output path is not valid UTF-8"))?,
            "--outdir",
            outdir
                .to_str()
                .ok_or_else(|| Error::other("plot output path is not valid UTF-8"))?,
            "--title",
            title,
        ])
        .status()?;

    if !status.success() {
        return Err(Error::other(format!(
            "plot renderer failed with status {status}; activate a Python environment with numpy and matplotlib"
        )));
    }

    Ok(outdir.join("plot.png"))
}
