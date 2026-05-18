/*!
Deterministic replicator task.

Purpose:
    This task wires a well-mixed frequency initial condition into the RK4
    replicator solver and writes automatically sized JSON time-series chunks.
*/

use std::io::Result;
use std::path::Path;
use std::sync::atomic::AtomicUsize;

use ndarray::{Array1, Array2};

use crate::Mode;
use crate::solvers::non_spatial::noise::Noise;
use crate::solvers::non_spatial::rk4::solve_with_termination;
use crate::solvers::termination::TerminationConfig;
use crate::tasks::metadata::{TaskOutcome, output_label, prepare_output_dir, save_metadata};
use crate::utils::create_well_mixed_gs;

/// Run one trajectory and let the signal writer chunk output files by size.
///
/// Details:
/// - Purpose: Runs deterministic replicator dynamics for `total_steps`.
/// - Parameters:
///   - `interaction_matrix`: Square interaction matrix `V`.
///   - `growth_vector`: Optional growth vector `g`; defaults to zero.
///   - `cutoff`: Frequency cutoff used by state sanitization.
///   - `dt`: Step size.
///   - `total_steps`: Total solver steps to execute.
///   - `save_interval`: Save every Nth step.
///   - `output_path`: Root output directory.
///   - `progress_counter`: Optional shared progress counter.
///   - `termination`: Explicit early-termination behavior.
pub fn run(
    interaction_matrix: &Array2<f64>,       // V
    growth_vector: Option<&Array1<f64>>,    // g override
    cutoff: f64,                            // cutoff
    dt: f64,                                // step size
    total_steps: usize,                     // total solver steps
    save_interval: usize,                   // save every N steps
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

    // Initial condition: well-mixed uniform simplex (ν_i = 1/d).
    let mode = Mode::Frequency {
        cutoff: Some(cutoff),
    };
    let gs = create_well_mixed_gs(mode, d, None);
    prepare_output_dir(output_path)?;

    let outcome = solve_with_termination(
        gs,                 // initial state
        interaction_matrix, // V
        growth_vector,      // g
        Noise::none(),      // deterministic run
        dt,                 // step size
        total_steps,        // steps
        save_interval,      // save every N steps
        output_path,        // output target
        progress_counter,
        termination,
    )?;

    let task_outcome = TaskOutcome::non_spatial(
        "replicator_deterministic",
        "well_mixed_replicator",
        &output_label(output_path),
        total_steps,
        dt,
        save_interval,
        outcome.steps_run,
        outcome.reason,
        outcome.signal_stats,
        d,
        Some(cutoff),
        termination.survivor_tolerance,
    );
    save_metadata(output_path, &task_outcome)?;

    Ok(task_outcome)
}
