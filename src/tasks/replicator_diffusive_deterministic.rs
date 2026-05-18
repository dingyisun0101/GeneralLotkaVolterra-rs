/*!
Deterministic diffusive replicator task.

Purpose:
    This task wires an arbitrary-dimensional local-frequency field into the
    spatial RK4 replicator reaction-diffusion solver and writes automatically
    sized JSON time-series chunks.
*/

use std::io::Result;
use std::path::Path;
use std::sync::atomic::AtomicUsize;

use ndarray::{Array1, Array2};

use crate::solvers::spatial::rk4::{Diffusion, solve_replicator_with_termination};
use crate::solvers::termination::TerminationConfig;
use crate::tasks::metadata::{TaskOutcome, output_label, prepare_output_dir, save_metadata};
use crate::utils::create_uniform_spatial_frequency_gs;

/// Run one spatial replicator trajectory and let signal/space writers chunk output files.
///
/// Details:
/// - Purpose: Runs deterministic local-simplex replicator reaction-diffusion
///   for `total_steps`.
/// - Parameters:
///   - `interaction_matrix`: Square interaction matrix `V`.
///   - `growth_vector`: Optional growth vector `g`; defaults to zero.
///   - `cutoff`: Frequency cutoff used by local-cell sanitization.
///   - `spatial_shape`: Arbitrary-dimensional grid shape, excluding species.
///   - `diffusion`: Per-species diffusion coefficients and grid metadata.
///   - `dt`: Step size.
///   - `total_steps`: Total solver steps to execute.
///   - `save_interval`: Save state and full spatial field every Nth step.
///   - `output_path`: Root output directory.
///   - `progress_counter`: Optional shared progress counter.
///   - `termination`: Explicit early-termination behavior.
pub fn run(
    interaction_matrix: &Array2<f64>,       // V
    growth_vector: Option<&Array1<f64>>,    // g override
    cutoff: f64,                            // cutoff
    spatial_shape: &[usize],                // grid shape without species axis
    diffusion: &Diffusion,                  // diffusion configuration
    dt: f64,                                // step size
    total_steps: usize,                     // total solver steps
    save_interval: usize,                   // save state and space every N steps
    output_path: &Path,                     // root output dir
    progress_counter: Option<&AtomicUsize>, // optional progress counter
    termination: TerminationConfig,         // explicit termination behavior
) -> Result<TaskOutcome> {
    let d = interaction_matrix.nrows();
    debug_assert_eq!(
        interaction_matrix.ncols(),
        d,
        "interaction_matrix must be square"
    );
    if let Some(g) = growth_vector {
        debug_assert_eq!(g.len(), d, "growth_vector length must match V");
    }
    debug_assert_eq!(
        diffusion.coefficients.len(),
        d,
        "diffusion coefficients must match V"
    );
    debug_assert_eq!(
        diffusion.spacing.len(),
        spatial_shape.len(),
        "diffusion spacing must match spatial_shape"
    );

    let gs = create_uniform_spatial_frequency_gs(Some(cutoff), spatial_shape, d);
    prepare_output_dir(output_path)?;

    let outcome = solve_replicator_with_termination(
        gs,                 // initial state
        interaction_matrix, // V
        growth_vector,      // g
        diffusion,          // diffusion configuration
        dt,                 // step size
        total_steps,        // steps
        save_interval,      // save aggregate state every N steps
        save_interval,      // include full spatial field every N steps
        output_path,        // output target
        progress_counter,
        termination,
    )?;

    let task_outcome = TaskOutcome::spatial(
        "replicator_diffusive_deterministic",
        "spatial_replicator",
        &output_label(output_path),
        total_steps,
        dt,
        save_interval,
        outcome.steps_run,
        outcome.reason,
        outcome.signal_stats,
        outcome.space_stats.unwrap_or_default(),
        d,
        spatial_shape,
        Some(cutoff),
        None,
        termination.survivor_tolerance,
    );
    save_metadata(output_path, &task_outcome)?;

    Ok(task_outcome)
}
