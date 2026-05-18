/*!
General utility helpers.

Purpose:
    Utility functions provide shared construction helpers that are not owned by
    a specific task or solver module.
*/

use crate::{Mode, Scalar, SystemState};
use ndarray::{Array1, ArrayD, IxDyn};

/// Create a well-mixed (spatially uniform / no-grid) `SystemState<T>` at time 0.
///
/// Details:
/// - Purpose: Creates the common no-grid initial condition used by task
///   runners.
/// - Parameters:
///   - `mode`: Frequency or population interpretation.
///   - `num_taxa`: Number of taxa in the global state vector.
///   - `population_i`: Optional per-taxon count for population mode.
pub fn create_well_mixed_gs<T>(
    mode: Mode<T>,           // representation convention
    num_taxa: usize,         // dimensionality d
    population_i: Option<T>, // per-taxon population for Population mode
) -> SystemState<T>
where
    T: Scalar,
{
    let zero = T::zero();
    let mut state = Array1::from_elem(num_taxa, zero);

    if num_taxa > 0 {
        match &mode {
            Mode::Frequency { .. } => {
                // Uniform simplex point: ν_i = 1/d.
                let v = T::one() / T::from(num_taxa).unwrap();
                state.iter_mut().for_each(|x| *x = v);
            }
            Mode::Population { .. } => {
                // Equal counts per taxon: n_i = population_i (default 1).
                let per_taxon = population_i.unwrap_or_else(T::one);
                state.iter_mut().for_each(|x| *x = per_taxon);
            }
        }
    }

    // time = 0, space = None (well-mixed)
    SystemState::from_arrays(mode, 0, state, None)
}

/// Create a uniform local-simplex spatial `SystemState<f64>` at time 0.
///
/// Details:
/// - Purpose: Builds the canonical spatial replicator initial condition used
///   by tasks and examples.
/// - Parameters:
///   - `cutoff`: Optional local-frequency cutoff for sanitization.
///   - `spatial_shape`: Grid shape excluding the species axis.
///   - `num_taxa`: Number of species on the final axis.
pub fn create_uniform_spatial_frequency_gs(
    cutoff: Option<f64>,
    spatial_shape: &[usize],
    num_taxa: usize,
) -> SystemState<f64> {
    let mode = Mode::Frequency { cutoff };
    let mut shape = spatial_shape.to_vec();
    shape.push(num_taxa);

    let initial_frequency = if num_taxa > 0 {
        1.0 / num_taxa as f64
    } else {
        0.0
    };
    let space = ArrayD::from_elem(IxDyn(&shape), initial_frequency);

    SystemState::from_arrays(mode, 0, Array1::zeros(num_taxa), Some(space))
}

/// Create a uniform spatial population `SystemState<f64>` at time 0.
///
/// Details:
/// - Purpose: Builds the canonical spatial GLV initial condition used by tasks
///   and examples.
/// - Parameters:
///   - `cutoff`: Optional population cutoff for sanitization.
///   - `carrying_capacity`: Optional global population cap.
///   - `spatial_shape`: Grid shape excluding the species axis.
///   - `num_taxa`: Number of species on the final axis.
///   - `initial_population`: Initial density per cell and species.
pub fn create_uniform_spatial_population_gs(
    cutoff: Option<f64>,
    carrying_capacity: Option<f64>,
    spatial_shape: &[usize],
    num_taxa: usize,
    initial_population: f64,
) -> SystemState<f64> {
    let mode = Mode::Population {
        cutoff,
        carrying_capacity,
    };
    let mut shape = spatial_shape.to_vec();
    shape.push(num_taxa);
    let space = ArrayD::from_elem(IxDyn(&shape), initial_population);

    SystemState::from_arrays(mode, 0, Array1::zeros(num_taxa), Some(space))
}
