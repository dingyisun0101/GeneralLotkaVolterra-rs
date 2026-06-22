/*!
Solver module surface.

Purpose:
    `solvers` groups numerical evolution backends and exposes a single public
    solve dispatcher over dynamics, space, noise, and termination settings.
*/

use std::io::{Error, ErrorKind, Result};
use std::path::Path;
use std::sync::atomic::AtomicUsize;

use ndarray::{Array1, Array2};

pub mod noise;
pub mod non_spatial;
pub mod spatial;
pub mod termination;

use crate::SystemState;
use crate::solvers::noise::Noise;
use crate::solvers::spatial::rk2::Diffusion;
use crate::solvers::termination::{SolveOutcome, TerminationConfig};

/// Deterministic dynamics family to integrate.
#[derive(Clone, Copy, Debug)]
pub enum Dynamics {
    /// GLV population dynamics.
    GlvPopulation,

    /// Replicator frequency dynamics.
    Replicator,
}

/// Spatial domain configuration.
#[derive(Clone, Copy, Debug)]
pub enum Space<'a> {
    /// Well-mixed, non-spatial state.
    None,

    /// Species-last spatial field with diffusion and full-space save cadence.
    Spatial {
        diffusion: &'a Diffusion,
        save_space_interval: usize,
    },
}

impl<'a> Space<'a> {
    #[inline]
    pub fn none() -> Self {
        Self::None
    }

    #[inline]
    pub fn spatial(diffusion: &'a Diffusion, save_space_interval: usize) -> Self {
        Self::Spatial {
            diffusion,
            save_space_interval,
        }
    }
}

/// Unified solver configuration.
#[derive(Clone, Copy, Debug)]
pub struct SolveConfig<'a> {
    pub dynamics: Dynamics,
    pub space: Space<'a>,
    pub noise: Noise,
    pub dt: f64,
    pub num_steps: usize,
    pub save_signal_interval: usize,
    pub output_path: &'a Path,
    pub termination: TerminationConfig,
}

/// Integrate one trajectory using the selected dynamics, spatial domain, and
/// noise policy.
pub fn solve(
    gs_i: SystemState<f64>,
    interaction_matrix: &Array2<f64>,
    growth_vector: Option<&Array1<f64>>,
    config: SolveConfig<'_>,
    progress_counter: Option<&AtomicUsize>,
) -> Result<SolveOutcome> {
    match (config.dynamics, config.space) {
        (Dynamics::Replicator, Space::None) => non_spatial::rk4::solve_with_termination(
            gs_i,
            interaction_matrix,
            growth_vector,
            config.noise,
            config.dt,
            config.num_steps,
            config.save_signal_interval,
            config.output_path,
            progress_counter,
            config.termination,
        ),
        (
            Dynamics::GlvPopulation,
            Space::Spatial {
                diffusion,
                save_space_interval,
            },
        ) => spatial::rk2::solve_with_termination(
            gs_i,
            interaction_matrix,
            growth_vector,
            diffusion,
            config.noise,
            config.dt,
            config.num_steps,
            config.save_signal_interval,
            save_space_interval,
            config.output_path,
            progress_counter,
            config.termination,
        ),
        (
            Dynamics::Replicator,
            Space::Spatial {
                diffusion,
                save_space_interval,
            },
        ) => spatial::rk2::solve_replicator_with_termination(
            gs_i,
            interaction_matrix,
            growth_vector,
            diffusion,
            config.noise,
            config.dt,
            config.num_steps,
            config.save_signal_interval,
            save_space_interval,
            config.output_path,
            progress_counter,
            config.termination,
        ),
        (Dynamics::GlvPopulation, Space::None) => Err(Error::new(
            ErrorKind::Unsupported,
            "well-mixed GLV population dynamics are not implemented",
        )),
    }
}
