//! Swappable stochastic-update plugins.
//!
//! A [`Noise`] plugin receives temporary typed abundance borrows after a
//! deterministic kernel transition. It cannot own the Workflow state, update
//! total abundance, enforce final invariants, or advance simulation time.

pub mod core;
pub mod demographic_gaussian;
pub mod none;
pub mod proportional_gaussian;

pub use core::{Noise, NoiseAlgorithm, NoiseDomain, NoisePluginError, NoiseStepError};
pub use demographic_gaussian::DemographicGaussian;
pub use none::NoNoise;
pub use proportional_gaussian::ProportionalGaussian;
