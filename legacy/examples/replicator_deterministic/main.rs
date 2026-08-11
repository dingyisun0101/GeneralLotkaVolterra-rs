/*!
Deterministic well-mixed replicator example.

Purpose:
    Demonstrates direct use of the unified solver API with no spatial field and
    no stochastic update.
*/

use std::io::{Error, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

use general_lotka_volterra_rs::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

mod constants;
use constants::*;

fn main() -> Result<()> {
    // Choose where solver output should be written. The output path is passed
    // to the signal writer inside `solve`, and later to the Python renderer.
    let output_path = Path::new(OUTPUT);

    // Start from a clean output directory. This removes stale `signal/`,
    // `space/`, and `metadata.json` entries from previous runs.
    prepare_output_dir(output_path)?;

    // The interaction matrix V defines pairwise effects between species. This
    // example uses a random dense matrix to keep the setup compact.
    let interaction_matrix = interaction_matrix();

    // A well-mixed replicator state is represented as a global frequency vector
    // with no spatial field. `Mode::Frequency` tells the solver to keep the
    // vector on the simplex, and `cutoff` prunes tiny frequencies to zero.
    let initial_state = create_well_mixed_gs(
        Mode::Frequency {
            cutoff: Some(CUTOFF),
        },
        NUM_STRAINS,
        None,
    );

    // Termination is explicit. This example checks for monoculture only at the
    // same cadence as saved samples. Use `TerminationConfig::disabled()` to run
    // exactly to `TOTAL_STEPS`.
    let termination = TerminationConfig::monoculture_only(SAVE_INTERVAL);

    // `solve` is the main user-facing API. The first arguments are the initial
    // state, interaction matrix, and optional growth vector. Replicator examples
    // can omit growth by passing `None`, which means g = 0.
    let outcome = solve(
        initial_state,
        &interaction_matrix,
        None,
        // `SolveConfig` selects the dynamics, spatial domain, noise policy,
        // time step, save cadence, output path, and termination behavior.
        SolveConfig {
            dynamics: Dynamics::Replicator,
            // `Space::None` means well-mixed / non-spatial.
            space: Space::None,
            // `Noise::none()` makes this a deterministic run.
            noise: Noise::none(),
            dt: DT,
            num_steps: TOTAL_STEPS,
            save_signal_interval: SAVE_INTERVAL,
            output_path,
            termination,
        },
        None,
    )?;

    // The solver returns raw writer stats and final state data. `TaskOutcome`
    // packages that into a stable metadata shape for downstream scripts.
    let task_outcome = TaskOutcome::non_spatial(
        LABEL,
        "well_mixed_replicator",
        &output_label(output_path),
        TOTAL_STEPS,
        DT,
        SAVE_INTERVAL,
        outcome.steps_run,
        outcome.reason,
        outcome.signal_stats,
        NUM_STRAINS,
        Some(CUTOFF),
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

/// Build the pairwise interaction matrix V used by the replicator equation:
/// dnu_i/dt = nu_i * (g_i + (V nu)_i - upsilon).
fn interaction_matrix() -> Array2<f64> {
    let mut rng = SmallRng::from_rng(&mut rand::rng());
    Array2::from_shape_fn((NUM_STRAINS, NUM_STRAINS), |_| {
        rng.random_range(RANDOM_INTERACTION_MIN..=RANDOM_INTERACTION_MAX)
    })
}

/// Print a compact run summary after metadata is saved.
fn print_summary(output_path: &Path, outcome: &TaskOutcome) {
    println!(
        "steps: {} / {}; reason: {:?}; signal files: {}; metadata: {}",
        outcome.steps_run,
        outcome.requested_steps,
        outcome.termination_reason,
        outcome.signal.files,
        output_path.join("metadata.json").display()
    );
}

/// Render `output/<example>/signal/*.json` into `plot/plot.png`.
///
/// This helper shells out to the bundled Python renderer. Users embedding the
/// crate in their own application can skip this entirely and consume JSON
/// outputs directly.
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
