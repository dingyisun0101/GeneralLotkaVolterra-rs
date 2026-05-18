#![allow(dead_code)]

//! Central settings for Cargo examples.
//!
//! Constants are grouped from shared knobs to example-specific knobs. Each
//! example imports this module and ignores settings that do not apply to it.

use std::io::Result;
use std::path::Path;
use std::sync::atomic::AtomicUsize;

use general_lotka_volterra_rs::solvers::spatial::rk4::{Boundary, Diffusion};
use general_lotka_volterra_rs::solvers::termination::TerminationConfig;
use ndarray::{Array1, Array2};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

// ---------------------------------------------------------------------------
// Shared settings
// ---------------------------------------------------------------------------

/// Number of strains/species used by both well-mixed and spatial examples.
pub const NUM_STRAINS: usize = 10;

/// Total solver steps for every example. Output files are chunked automatically.
pub const TOTAL_STEPS: usize = 10_000;

// ---------------------------------------------------------------------------
// Shared Non-spatial settings
// ---------------------------------------------------------------------------

/// Minimum random interaction coefficient for random well-mixed examples.
pub const RANDOM_INTERACTION_MIN: f64 = -0.5;

/// Maximum random interaction coefficient for random well-mixed examples.
pub const RANDOM_INTERACTION_MAX: f64 = 0.5;

/// Frequency cutoff for well-mixed simplex examples.
pub const WELL_MIXED_CUTOFF: f64 = 1e-5;

/// Integration step size for well-mixed examples.
pub const WELL_MIXED_DT: f64 = 0.005;

/// Save one state sample every N solver steps for non-spatial examples.
pub const NON_SPATIAL_SAVE_INTERVAL: usize = 500;

// ---------------------------------------------------------------------------
// Shared Spatial settings
// ---------------------------------------------------------------------------

/// Spatial grid shape shared by diffusive examples, excluding species axis.
pub const SPATIAL_SHAPE: [usize; 2] = [128, 128];

/// Cutoff applied by spatial examples during sanitization.
pub const SPATIAL_CUTOFF: f64 = 1e-9;

/// Integration step size for diffusive spatial examples.
pub const SPATIAL_DT: f64 = 0.003;

/// Save one state and full spatial field sample every N solver steps.
pub const SPATIAL_SAVE_INTERVAL: usize = 500;

// ---------------------------------------------------------------------------
// replicator_deterministic
// ---------------------------------------------------------------------------

/// Output directory for the deterministic well-mixed replicator example.
pub const REPLICATOR_DETERMINISTIC_OUTPUT: &str = "output/replicator_deterministic";

/// Plot title and progress label for the deterministic well-mixed example.
pub const REPLICATOR_DETERMINISTIC_LABEL: &str = "replicator_deterministic";

// ---------------------------------------------------------------------------
// replicator_demographic
// ---------------------------------------------------------------------------

/// Output directory for the demographic-noise well-mixed replicator example.
pub const REPLICATOR_DEMOGRAPHIC_OUTPUT: &str = "output/replicator_demographic";

/// Plot title and progress label for the demographic-noise example.
pub const REPLICATOR_DEMOGRAPHIC_LABEL: &str = "replicator_demographic";

/// Demographic Gaussian noise strength used after each deterministic step.
pub const REPLICATOR_DEMOGRAPHIC_SIGMA: f64 = 0.1;

// ---------------------------------------------------------------------------
// replicator_diffusive_deterministic
// ---------------------------------------------------------------------------

/// Output directory for the diffusive deterministic replicator example.
pub const REPLICATOR_DIFFUSIVE_DETERMINISTIC_OUTPUT: &str =
    "output/replicator_diffusive_deterministic";

/// Plot title and progress label for the diffusive deterministic replicator.
pub const REPLICATOR_DIFFUSIVE_DETERMINISTIC_LABEL: &str = "replicator_diffusive_deterministic";

/// Base diffusion coefficient for spatial replicator species 0.
pub const REPLICATOR_DIFFUSIVE_DIFFUSION_BASE: f64 = 0.020;

/// Per-species diffusion decrement for the spatial replicator example.
pub const REPLICATOR_DIFFUSIVE_DIFFUSION_STEP: f64 = 0.001;

/// Base growth value for spatial replicator species 0.
pub const REPLICATOR_DIFFUSIVE_GROWTH_BASE: f64 = 0.02;

/// Per-species growth decrement for the spatial replicator example.
pub const REPLICATOR_DIFFUSIVE_GROWTH_STEP: f64 = 0.004;

/// Cyclic interaction strength for the spatial replicator example.
pub const REPLICATOR_DIFFUSIVE_INTERACTION_STRENGTH: f64 = 1.0;

// ---------------------------------------------------------------------------
// lv_diffusive_deterministic
// ---------------------------------------------------------------------------

/// Output directory for the diffusive deterministic GLV example.
pub const LV_DIFFUSIVE_DETERMINISTIC_OUTPUT: &str = "output/lv_diffusive_deterministic";

/// Plot title and progress label for the diffusive deterministic GLV example.
pub const LV_DIFFUSIVE_DETERMINISTIC_LABEL: &str = "lv_diffusive_deterministic";

/// Base diffusion coefficient for spatial GLV species 0.
pub const LV_DIFFUSIVE_DIFFUSION_BASE: f64 = 0.025;

/// Per-species diffusion decrement for the spatial GLV example.
pub const LV_DIFFUSIVE_DIFFUSION_STEP: f64 = 0.0015;

/// Base growth value for spatial GLV species 0.
pub const LV_DIFFUSIVE_GROWTH_BASE: f64 = 0.35;

/// Per-species growth decrement for the spatial GLV example.
pub const LV_DIFFUSIVE_GROWTH_STEP: f64 = 0.015;

/// Self-limitation coefficient on the GLV interaction matrix diagonal.
pub const LV_DIFFUSIVE_SELF_LIMITATION: f64 = -0.70;

/// Off-diagonal GLV interaction scale for cyclic pair effects.
pub const LV_DIFFUSIVE_PAIR_INTERACTION: f64 = 0.08;

/// Optional global carrying capacity for the GLV population field.
pub const LV_DIFFUSIVE_CARRYING_CAPACITY: Option<f64> = Some(40_000.0);

/// Initial population density per spatial cell and species.
pub const LV_DIFFUSIVE_INITIAL_POPULATION: f64 = 0.25;

pub fn replicator_deterministic_output_path() -> &'static Path {
    Path::new(REPLICATOR_DETERMINISTIC_OUTPUT)
}

pub fn replicator_demographic_output_path() -> &'static Path {
    Path::new(REPLICATOR_DEMOGRAPHIC_OUTPUT)
}

pub fn replicator_diffusive_deterministic_output_path() -> &'static Path {
    Path::new(REPLICATOR_DIFFUSIVE_DETERMINISTIC_OUTPUT)
}

pub fn lv_diffusive_deterministic_output_path() -> &'static Path {
    Path::new(LV_DIFFUSIVE_DETERMINISTIC_OUTPUT)
}

pub fn non_spatial_termination() -> TerminationConfig {
    TerminationConfig::monoculture_only(NON_SPATIAL_SAVE_INTERVAL)
}

pub fn spatial_termination() -> TerminationConfig {
    TerminationConfig::monoculture_only(SPATIAL_SAVE_INTERVAL)
}

pub fn run_replicator_deterministic(progress_counter: Option<&AtomicUsize>) -> Result<()> {
    general_lotka_volterra_rs::tasks::replicator_deterministic::run(
        &well_mixed_interaction_matrix(),
        None,
        WELL_MIXED_CUTOFF,
        WELL_MIXED_DT,
        TOTAL_STEPS,
        NON_SPATIAL_SAVE_INTERVAL,
        replicator_deterministic_output_path(),
        progress_counter,
        non_spatial_termination(),
    )
}

pub fn run_replicator_demographic(progress_counter: Option<&AtomicUsize>) -> Result<()> {
    general_lotka_volterra_rs::tasks::replicator_demographic::run(
        &well_mixed_interaction_matrix(),
        None,
        WELL_MIXED_CUTOFF,
        REPLICATOR_DEMOGRAPHIC_SIGMA,
        WELL_MIXED_DT,
        TOTAL_STEPS,
        NON_SPATIAL_SAVE_INTERVAL,
        replicator_demographic_output_path(),
        progress_counter,
        non_spatial_termination(),
    )
}

pub fn run_replicator_diffusive_deterministic(
    progress_counter: Option<&AtomicUsize>,
) -> Result<()> {
    general_lotka_volterra_rs::tasks::replicator_diffusive_deterministic::run(
        &replicator_diffusive_interaction_matrix(),
        Some(&replicator_diffusive_growth_vector()),
        SPATIAL_CUTOFF,
        &SPATIAL_SHAPE,
        &replicator_diffusive_diffusion(),
        SPATIAL_DT,
        TOTAL_STEPS,
        SPATIAL_SAVE_INTERVAL,
        replicator_diffusive_deterministic_output_path(),
        progress_counter,
        spatial_termination(),
    )
}

pub fn run_lv_diffusive_deterministic(progress_counter: Option<&AtomicUsize>) -> Result<()> {
    general_lotka_volterra_rs::tasks::lv_diffusive_deterministic::run(
        &lv_diffusive_interaction_matrix(),
        Some(&lv_diffusive_growth_vector()),
        SPATIAL_CUTOFF,
        LV_DIFFUSIVE_CARRYING_CAPACITY,
        &SPATIAL_SHAPE,
        LV_DIFFUSIVE_INITIAL_POPULATION,
        &lv_diffusive_diffusion(),
        SPATIAL_DT,
        TOTAL_STEPS,
        SPATIAL_SAVE_INTERVAL,
        lv_diffusive_deterministic_output_path(),
        progress_counter,
        spatial_termination(),
    )
}

fn well_mixed_interaction_matrix() -> Array2<f64> {
    let mut rng = SmallRng::from_rng(&mut rand::rng());

    Array2::from_shape_fn((NUM_STRAINS, NUM_STRAINS), |_| {
        rng.random_range(RANDOM_INTERACTION_MIN..=RANDOM_INTERACTION_MAX)
    })
}

fn replicator_diffusive_interaction_matrix() -> Array2<f64> {
    Array2::from_shape_fn((NUM_STRAINS, NUM_STRAINS), |(i, j)| {
        if i == j {
            0.0
        } else if (j + NUM_STRAINS - i) % NUM_STRAINS <= NUM_STRAINS / 2 {
            -REPLICATOR_DIFFUSIVE_INTERACTION_STRENGTH
        } else {
            REPLICATOR_DIFFUSIVE_INTERACTION_STRENGTH
        }
    })
}

fn replicator_diffusive_growth_vector() -> Array1<f64> {
    Array1::from_shape_fn(NUM_STRAINS, |i| {
        REPLICATOR_DIFFUSIVE_GROWTH_BASE - REPLICATOR_DIFFUSIVE_GROWTH_STEP * i as f64
    })
}

fn replicator_diffusive_diffusion() -> Diffusion {
    Diffusion::unit_spacing(
        Array1::from_shape_fn(NUM_STRAINS, |i| {
            (REPLICATOR_DIFFUSIVE_DIFFUSION_BASE - REPLICATOR_DIFFUSIVE_DIFFUSION_STEP * i as f64)
                .max(0.001)
        }),
        SPATIAL_SHAPE.len(),
        Boundary::Periodic,
    )
}

fn lv_diffusive_interaction_matrix() -> Array2<f64> {
    Array2::from_shape_fn((NUM_STRAINS, NUM_STRAINS), |(i, j)| {
        if i == j {
            LV_DIFFUSIVE_SELF_LIMITATION
        } else {
            let direction = if (j + NUM_STRAINS - i) % NUM_STRAINS <= NUM_STRAINS / 2 {
                -1.0
            } else {
                1.0
            };
            direction * LV_DIFFUSIVE_PAIR_INTERACTION / (1.0 + i.abs_diff(j) as f64)
        }
    })
}

fn lv_diffusive_growth_vector() -> Array1<f64> {
    Array1::from_shape_fn(NUM_STRAINS, |i| {
        LV_DIFFUSIVE_GROWTH_BASE - LV_DIFFUSIVE_GROWTH_STEP * i as f64
    })
}

fn lv_diffusive_diffusion() -> Diffusion {
    Diffusion::unit_spacing(
        Array1::from_shape_fn(NUM_STRAINS, |i| {
            (LV_DIFFUSIVE_DIFFUSION_BASE - LV_DIFFUSIVE_DIFFUSION_STEP * i as f64).max(0.001)
        }),
        SPATIAL_SHAPE.len(),
        Boundary::Neumann,
    )
}
