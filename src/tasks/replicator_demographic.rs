/*!
Demographic-noise replicator task.

Purpose:
    This task wires a well-mixed frequency initial condition into the RK4
    replicator solver with demographic Gaussian noise after each deterministic
    step.
*/

use std::io::Result;
use std::path::Path;
use std::sync::atomic::AtomicUsize;

use ndarray::{Array1, Array2};

use crate::solvers::non_spatial::noise::Noise;
use crate::solvers::non_spatial::rk4::solve;
use crate::state::Mode;
use crate::utils::create_well_mixed_gs;

/// Run `num_epochs` trajectories back-to-back, persisting each epoch to disk.
///
/// Details:
/// - Purpose: Runs demographic-noise replicator dynamics over sequential
///   epochs.
/// - Parameters:
///   - `interaction_matrix`: Square interaction matrix `V`.
///   - `growth_vector`: Optional growth vector `g`; defaults to zero.
///   - `cutoff`: Frequency cutoff used by state sanitization.
///   - `sigma`: Demographic noise strength.
///   - `dt`: Step size.
///   - `epoch_len`: Steps per epoch.
///   - `save_interval`: Save every Nth step.
///   - `num_epochs`: Number of epochs to execute.
///   - `output_path`: Root output directory.
///   - `progress_counter`: Optional shared progress counter.
pub fn run(
    interaction_matrix: &Array2<f64>,       // V
    growth_vector: Option<&Array1<f64>>,    // g override
    cutoff: f64,                            // cutoff
    sigma: f64,                             // demographic noise strength
    dt: f64,                                // step size
    epoch_len: usize,                       // steps per epoch
    save_interval: usize,                   // save every N steps
    num_epochs: usize,                      // number of epochs
    output_path: &Path,                     // root output dir
    progress_counter: Option<&AtomicUsize>, // optional progress counter
) -> Result<()> {
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
    let mut gs = create_well_mixed_gs(mode, d, None);

    // Sequential epochs: carry final state from epoch k into epoch k+1.
    for epoch in 1..=num_epochs {
        gs = solve(
            epoch,                              // current epoch
            gs,                                 // initial state for this epoch
            interaction_matrix,                 // V
            growth_vector,                      // g
            Noise::demographic_gaussian(sigma), // demographic noise
            dt,                                 // step size
            epoch_len,                          // steps
            save_interval,                      // save every N steps
            &output_path,                       // output target for this epoch
            progress_counter,
        )?;
    }

    Ok(())
}
