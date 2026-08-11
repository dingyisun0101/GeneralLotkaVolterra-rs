/*!
Non-spatial stochastic updates.

Purpose:
    This module preserves the historical non-spatial noise import path while
    delegating the public noise model and raw update kernels to
    `crate::solvers::noise`.
*/

use rand::Rng;

use crate::SystemState;
use crate::solvers::noise::apply_noise_to_slice;

pub use crate::solvers::noise::{Noise, NoiseContext, NoiseKind};

/// Apply noise in-place to `state.state` and restore feasibility via
/// `SystemState::sanitize`.
#[inline]
pub fn apply_noise_inplace(
    state: &mut SystemState<f64>,
    noise: Noise,
    dt: f64,
    ctx: &mut NoiseContext,
    rng_local: &mut impl Rng,
) {
    if dt == 0.0 || noise.is_none_or_zero() {
        return;
    }

    apply_noise_to_slice(
        state
            .state
            .as_slice_mut()
            .expect("SystemState.state is contiguous"),
        noise,
        dt,
        ctx,
        rng_local,
    );
    state.sanitize();
}
