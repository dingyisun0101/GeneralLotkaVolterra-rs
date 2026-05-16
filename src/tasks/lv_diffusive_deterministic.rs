/*!
Deterministic diffusive GLV task.

Purpose:
    This task wires an arbitrary-dimensional population-density field into the
    spatial RK4 GLV reaction-diffusion solver and writes one JSON time series
    per epoch.
*/

use std::io::Result;
use std::path::Path;
use std::sync::atomic::AtomicUsize;

use ndarray::{Array1, Array2, ArrayD, IxDyn};

use crate::solvers::spatial::rk4::{Diffusion, solve};
use crate::state::{Mode, SystemState};

/// Run `num_epochs` spatial GLV trajectories back-to-back.
///
/// Details:
/// - Purpose: Runs deterministic GLV reaction-diffusion over sequential epochs.
/// - Parameters:
///   - `interaction_matrix`: Square interaction matrix `V`.
///   - `growth_vector`: Optional growth vector `g`; defaults to zero.
///   - `cutoff`: Population cutoff used by spatial sanitization.
///   - `carrying_capacity`: Optional global population cap.
///   - `spatial_shape`: Arbitrary-dimensional grid shape, excluding species.
///   - `initial_population`: Initial value per cell per species.
///   - `diffusion`: Per-species diffusion coefficients and grid metadata.
///   - `dt`: Step size.
///   - `epoch_len`: Steps per epoch.
///   - `save_signal_interval`: Save aggregate state every Nth step.
///   - `save_space_interval`: Include full spatial field every Nth step.
///   - `num_epochs`: Number of epochs to execute.
///   - `output_path`: Root output directory.
///   - `progress_counter`: Optional shared progress counter.
pub fn run(
    interaction_matrix: &Array2<f64>,       // V
    growth_vector: Option<&Array1<f64>>,    // g override
    cutoff: f64,                            // cutoff
    carrying_capacity: Option<f64>,         // optional global cap
    spatial_shape: &[usize],                // grid shape without species axis
    initial_population: f64,                // initial density per cell/species
    diffusion: &Diffusion,                  // diffusion configuration
    dt: f64,                                // step size
    epoch_len: usize,                       // steps per epoch
    save_signal_interval: usize,            // save aggregate state every N steps
    save_space_interval: usize,             // include full spatial field every N steps
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

    let mode = Mode::Population {
        cutoff: Some(cutoff),
        carrying_capacity,
    };
    let mut shape = spatial_shape.to_vec();
    shape.push(d);
    let space = ArrayD::from_elem(IxDyn(&shape), initial_population);
    let mut gs = SystemState::from_arrays(mode, 0, Array1::zeros(d), Some(space));

    // Sequential epochs: carry final state from epoch k into epoch k+1.
    for epoch in 1..=num_epochs {
        gs = solve(
            epoch,                // current epoch
            gs,                   // initial state for this epoch
            interaction_matrix,   // V
            growth_vector,        // g
            diffusion,            // diffusion configuration
            dt,                   // step size
            epoch_len,            // steps
            save_signal_interval, // save aggregate state every N steps
            save_space_interval,  // include full spatial field every N steps
            output_path,          // output target for this epoch
            progress_counter,
        )?;
    }

    Ok(())
}
