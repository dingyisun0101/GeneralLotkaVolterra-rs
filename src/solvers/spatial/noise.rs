/*!
Spatial stochastic updates.

Purpose:
    This module mirrors `non_spatial::noise`: it preserves a spatial-specific
    noise application boundary while delegating the public noise model and raw
    slice kernels to `crate::solvers::noise`.
*/

use std::io::{Error, ErrorKind, Result};

use rand::Rng;

use crate::SystemState;
use crate::solvers::noise::apply_noise_to_slice;

use super::rk4::{
    Dynamics, SpatialLayout, sanitize_local_simplex_space_and_refresh_state,
    sanitize_space_and_refresh_state,
};

pub use crate::solvers::noise::{Noise, NoiseContext, NoiseKind};

/// Apply noise in-place to each spatial cell's species vector and restore the
/// spatial solver's invariant boundary.
#[inline]
pub(super) fn apply_noise_inplace(
    state: &mut SystemState<f64>,
    noise: Noise,
    dt: f64,
    layout: &SpatialLayout,
    dynamics: Dynamics,
    ctx: &mut NoiseContext,
    rng: &mut impl Rng,
) -> Result<()> {
    if dt == 0.0 || noise.is_none_or_zero() {
        return Ok(());
    }

    let Some(space) = state.space.as_mut() else {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "spatial RK4 requires SystemState.space",
        ));
    };
    let u = space.as_slice_memory_order_mut().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidInput,
            "spatial state must use standard contiguous memory layout",
        )
    })?;

    for cell in 0..layout.num_cells {
        let base = cell * layout.num_species;
        apply_noise_to_slice(&mut u[base..base + layout.num_species], noise, dt, ctx, rng);
    }

    match dynamics {
        Dynamics::GlvPopulation => sanitize_space_and_refresh_state(state, layout),
        Dynamics::LocalReplicatorFrequency => {
            sanitize_local_simplex_space_and_refresh_state(state, layout)
        }
    }
}
