/*!
Non-spatial stochastic updates.

Purpose:
    This module defines optional stochasticity applied after a deterministic
    solver step. Noise mutates `SystemState<f64>` in place and finishes by
    calling `SystemState::sanitize`.

Runtime model:
    `Noise` is the user-facing configuration. `NoiseContext` owns reusable
    buffers and distribution objects so hot solver loops do not allocate random
    scratch space every step.
*/
#![allow(dead_code)]

use crate::state::SystemState;
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Copy, Debug)]
pub enum NoiseKind {
    /// No noise.
    None,

    /// Multiplicative Gaussian noise on ν with an approximate mass-preserving
    ///     projection (prior to sanitize):
    ///     ν_i < ν_i [1 + σ(η_i - \bar{η}) sqrt(dt)]
    ///     where η_i ~ N(0,1) and \bar{η} = Σ_j ν_j η_j.
    ProportionalGaussian { sigma: f64 },

    /// Demographic noise:
    ///     Gaussian fluctuations proportional to sqrt(ν_i).
    ///     ν_i < ν_i + σ sqrt(ν_i) (η_i - \bar{η}_{sqrt(ν)}) sqrt(dt)
    ///     where \bar{η}_{sqrt(ν)} = (Σ_j sqrt(ν_j) η_j) / (Σ_j sqrt(ν_j)).
    DemographicGaussian { sigma: f64 },
}

/// Noise configuration wrapper (public API).
#[derive(Clone, Copy, Debug)]
pub struct Noise {
    pub kind: NoiseKind,
}

impl Noise {
    #[inline]
    pub fn none() -> Self {
        Self {
            kind: NoiseKind::None,
        }
    }

    #[inline]
    pub fn proportional_gaussian(sigma: f64) -> Self {
        Self {
            kind: NoiseKind::ProportionalGaussian { sigma },
        }
    }

    #[inline]
    pub fn demographic_gaussian(sigma: f64) -> Self {
        Self {
            kind: NoiseKind::DemographicGaussian { sigma },
        }
    }
}

/// Reusable buffers and distribution objects for noise sampling.
///
/// Details:
/// - Purpose: Owns the standard-normal draw buffer and distribution object
///   reused across solver steps.
/// - Parameters:
///   - (none): Construct with `NoiseContext::new`.
pub struct NoiseContext {
    /// Standard normal draws, length d.
    eta: Vec<f64>,

    /// Standard normal distribution object.
    normal: Normal<f64>,
}

impl NoiseContext {
    /// Create a new context for simplex dimension `d`.
    ///
    /// Details:
    /// - Purpose: Allocates reusable noise scratch space.
    /// - Parameters:
    ///   - `d`: State dimension.
    #[inline]
    pub fn new(d: usize) -> Self {
        assert!(d > 0, "NoiseContext::new: d must be > 0");
        Self {
            eta: vec![0.0; d],
            normal: Normal::<f64>::new(0.0, 1.0).expect("Normal(0,1) ctor"),
        }
    }

    /// Ensure the buffer matches a desired dimension (useful if d is dynamic).
    ///
    /// Details:
    /// - Purpose: Keeps the context reusable if a caller changes state
    ///   dimension between runs.
    /// - Parameters:
    ///   - `d`: Desired state dimension.
    #[inline]
    pub fn resize_if_needed(&mut self, d: usize) {
        if self.eta.len() != d {
            self.eta.resize(d, 0.0);
        }
    }
}

/// Apply noise in-place to `state` and then restore feasibility via `state.sanitize()`.
///
/// Details:
/// - Purpose: Applies the configured stochastic update after a deterministic
///   integration step.
/// - Parameters:
///   - `state`: State to mutate.
///   - `noise`: Stochastic update policy.
///   - `dt`: Step size used for noise scaling.
///   - `ctx`: Reusable noise scratch context.
///   - `rng_local`: Random-number generator.
#[inline]
pub fn apply_noise_inplace(
    state: &mut SystemState<f64>,
    noise: Noise,
    dt: f64,
    ctx: &mut NoiseContext,
    rng_local: &mut impl Rng,
) {
    // Fast exit: avoid doing anything when timestep is degenerate.
    if dt == 0.0 {
        return;
    }

    let d = state.state.len();
    ctx.resize_if_needed(d);

    match noise.kind {
        NoiseKind::None => {}

        NoiseKind::ProportionalGaussian { sigma } => {
            if sigma == 0.0 {
                return;
            }

            // ----------------------------------------------------------------------------------
            // (1) Sample eta_i ~ N(0,1)
            // ----------------------------------------------------------------------------------
            for e in ctx.eta.iter_mut() {
                *e = ctx.normal.sample(rng_local);
            }

            // ----------------------------------------------------------------------------------
            // (2) eta_bar = Σ nu_i eta_i
            // ----------------------------------------------------------------------------------
            let nu_arr = &state.state;
            let mut eta_bar = 0.0;
            for i in 0..d {
                eta_bar += nu_arr[i] * ctx.eta[i];
            }

            // ----------------------------------------------------------------------------------
            // (3) Multiplicative update
            // ----------------------------------------------------------------------------------
            let scale = sigma * dt.sqrt();
            let nu_mut = &mut state.state;
            for i in 0..d {
                let val = nu_mut[i] * (1.0 + scale * (ctx.eta[i] - eta_bar));
                nu_mut[i] = if val.is_finite() && val > 0.0 {
                    val
                } else {
                    0.0
                };
            }

            // ----------------------------------------------------------------------------------
            // (4) Restore simplex constraints
            // ----------------------------------------------------------------------------------
            state.sanitize();
        }

        NoiseKind::DemographicGaussian { sigma } => {
            if sigma == 0.0 {
                return;
            }

            // ----------------------------------------------------------------------------------
            // (1) Sample eta_i ~ N(0,1)
            // ----------------------------------------------------------------------------------
            for e in ctx.eta.iter_mut() {
                *e = ctx.normal.sample(rng_local);
            }

            // ----------------------------------------------------------------------------------
            // (2) eta_bar_sqrt = (Σ sqrt(nu_i) eta_i) / (Σ sqrt(nu_i))
            // ----------------------------------------------------------------------------------
            let nu_arr = &state.state;
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..d {
                let xi = if nu_arr[i] > 0.0 { nu_arr[i] } else { 0.0 };
                let sxi = xi.sqrt();
                num += sxi * ctx.eta[i];
                den += sxi;
            }
            let eta_bar_sqrt = if den > 0.0 { num / den } else { 0.0 };

            // ----------------------------------------------------------------------------------
            // (3) Additive update
            // ----------------------------------------------------------------------------------
            let scale = sigma * dt.sqrt();
            let nu_mut = &mut state.state;
            for i in 0..d {
                let xi = if nu_mut[i] > 0.0 { nu_mut[i] } else { 0.0 };
                let incr = scale * xi.sqrt() * (ctx.eta[i] - eta_bar_sqrt);
                let val = xi + incr;
                nu_mut[i] = if val.is_finite() && val > 0.0 {
                    val
                } else {
                    0.0
                };
            }

            // ----------------------------------------------------------------------------------
            // (4) Restore simplex constraints
            // ----------------------------------------------------------------------------------
            state.sanitize();
        }
    }
}
