//! Swappable stochastic-update plugins.
//!
//! A [`Noise`] plugin receives temporary typed abundance borrows after a
//! deterministic kernel transition. It cannot own the Workflow state, update
//! total abundance, enforce final invariants, or advance simulation time.

mod algorithms;
pub mod core;

pub use algorithms::{
    DEMOGRAPHIC_GAUSSIAN_RNG_NAMESPACE, DemographicGaussian, NoNoise,
    PROPORTIONAL_GAUSSIAN_RNG_NAMESPACE, ProportionalGaussian,
};
pub use core::{Noise, NoiseAlgorithm, NoiseDomain, NoisePluginError, NoiseStepError};
