/*!
Shared stochastic update machinery.

Purpose:
    This module owns the public noise configuration and reusable Gaussian
    sampling context. Solvers decide which storage is the source of truth and
    when to restore representation invariants after a raw noise update.
*/

use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Copy, Debug)]
pub enum NoiseKind {
    /// No stochastic update.
    None,

    /// Multiplicative Gaussian noise with an approximate local mass projection.
    ProportionalGaussian { sigma: f64 },

    /// Additive Gaussian fluctuations scaled by sqrt(x_i).
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

    #[inline]
    pub fn is_none_or_zero(self) -> bool {
        match self.kind {
            NoiseKind::None => true,
            NoiseKind::ProportionalGaussian { sigma }
            | NoiseKind::DemographicGaussian { sigma } => sigma == 0.0,
        }
    }
}

/// Reusable buffers and distribution objects for noise sampling.
pub struct NoiseContext {
    eta: Vec<f64>,
    normal: Normal<f64>,
}

impl NoiseContext {
    #[inline]
    pub fn new(d: usize) -> Self {
        assert!(d > 0, "NoiseContext::new: d must be > 0");
        Self {
            eta: vec![0.0; d],
            normal: Normal::<f64>::new(0.0, 1.0).expect("Normal(0,1) ctor"),
        }
    }

    #[inline]
    pub fn resize_if_needed(&mut self, d: usize) {
        if self.eta.len() != d {
            self.eta.resize(d, 0.0);
        }
    }

    #[inline]
    fn sample_standard_normals(&mut self, d: usize, rng: &mut impl Rng) -> &[f64] {
        self.resize_if_needed(d);
        for eta in self.eta[..d].iter_mut() {
            *eta = self.normal.sample(rng);
        }
        &self.eta[..d]
    }
}

/// Apply raw noise to one species vector.
///
/// The caller owns post-update sanitization. This lets well-mixed frequency,
/// spatial local-frequency, and spatial population solvers share the update
/// kernels while preserving their different invariant boundaries.
#[inline]
pub fn apply_noise_to_slice(
    values: &mut [f64],
    noise: Noise,
    dt: f64,
    ctx: &mut NoiseContext,
    rng: &mut impl Rng,
) {
    if dt == 0.0 || values.is_empty() {
        return;
    }

    match noise.kind {
        NoiseKind::None => {}
        NoiseKind::ProportionalGaussian { sigma } => {
            if sigma != 0.0 {
                apply_proportional_gaussian_slice(values, sigma, dt, ctx, rng);
            }
        }
        NoiseKind::DemographicGaussian { sigma } => {
            if sigma != 0.0 {
                apply_demographic_gaussian_slice(values, sigma, dt, ctx, rng);
            }
        }
    }
}

#[inline]
fn apply_proportional_gaussian_slice(
    values: &mut [f64],
    sigma: f64,
    dt: f64,
    ctx: &mut NoiseContext,
    rng: &mut impl Rng,
) {
    let d = values.len();
    let eta = ctx.sample_standard_normals(d, rng);

    let total: f64 = values
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .sum();

    let mut eta_bar = 0.0;
    if total > 0.0 {
        let inv_total = 1.0 / total;
        for i in 0..d {
            let x = if values[i].is_finite() && values[i] > 0.0 {
                values[i]
            } else {
                0.0
            };
            eta_bar += x * inv_total * eta[i];
        }
    }

    let scale = sigma * dt.sqrt();
    for i in 0..d {
        let x = if values[i].is_finite() && values[i] > 0.0 {
            values[i]
        } else {
            0.0
        };
        let value = x * (1.0 + scale * (eta[i] - eta_bar));
        values[i] = if value.is_finite() && value > 0.0 {
            value
        } else {
            0.0
        };
    }
}

#[inline]
fn apply_demographic_gaussian_slice(
    values: &mut [f64],
    sigma: f64,
    dt: f64,
    ctx: &mut NoiseContext,
    rng: &mut impl Rng,
) {
    let d = values.len();
    let eta = ctx.sample_standard_normals(d, rng);

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for i in 0..d {
        let x = if values[i].is_finite() && values[i] > 0.0 {
            values[i]
        } else {
            0.0
        };
        let sqrt_x = x.sqrt();
        numerator += sqrt_x * eta[i];
        denominator += sqrt_x;
    }
    let eta_bar_sqrt = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    };

    let scale = sigma * dt.sqrt();
    for i in 0..d {
        let x = if values[i].is_finite() && values[i] > 0.0 {
            values[i]
        } else {
            0.0
        };
        let value = x + scale * x.sqrt() * (eta[i] - eta_bar_sqrt);
        values[i] = if value.is_finite() && value > 0.0 {
            value
        } else {
            0.0
        };
    }
}
