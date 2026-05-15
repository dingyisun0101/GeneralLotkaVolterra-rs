/*!
General utility helpers.

Purpose:
    Utility functions provide shared construction helpers that are not owned by
    a specific task or solver module.
*/

use crate::state::{Mode, Scalar, SystemState};
use ndarray::Array1;

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
