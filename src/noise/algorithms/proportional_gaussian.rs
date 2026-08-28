//! Gaussian fluctuations proportional to local abundance.

use crate::{AggregateAbundance, SpatialAbundance, TimeStep};
use physics_in_parallel::prelude::basic::RngConfig;

use crate::noise::core::{
    GaussianKind, GaussianWorkspace, NoiseAlgorithm, NoiseDomain, NoisePluginError,
};

/// Workflow metadata namespace for proportional Gaussian RNG provenance.
pub const PROPORTIONAL_GAUSSIAN_RNG_NAMESPACE: &str = "glv.noise.proportional_gaussian";

/// Seeded proportional Gaussian noise with fixed reusable scratch.
#[derive(Debug)]
pub struct ProportionalGaussian {
    workspace: GaussianWorkspace,
}

impl ProportionalGaussian {
    /// Creates a seeded proportional plugin for one fixed payload domain.
    pub fn new(sigma: f64, rng: RngConfig, domain: NoiseDomain) -> Result<Self, NoisePluginError> {
        Ok(Self {
            workspace: GaussianWorkspace::new(
                sigma,
                rng,
                domain,
                PROPORTIONAL_GAUSSIAN_RNG_NAMESPACE,
            )?,
        })
    }

    /// Returns the configured Gaussian strength.
    pub const fn sigma(&self) -> f64 {
        self.workspace.sigma()
    }

    /// Returns the fully resolved PiP RNG configuration.
    pub fn rng_config(&self) -> RngConfig {
        self.workspace.rng_config()
    }

    /// Returns the maximum workers that this plugin's random filler may occupy.
    pub const fn max_threads(&self) -> usize {
        self.workspace.max_threads()
    }

    /// Sets the maximum workers that this plugin's random filler may occupy.
    pub fn set_max_threads(&mut self, max_threads: usize) -> Result<(), NoisePluginError> {
        self.workspace.set_max_threads(max_threads)
    }

    /// Returns this plugin with a new random-instance worker maximum.
    pub fn with_max_threads(mut self, max_threads: usize) -> Result<Self, NoisePluginError> {
        self.set_max_threads(max_threads)?;
        Ok(self)
    }

    /// Borrows the fixed aggregate or spatial domain.
    pub const fn domain(&self) -> &NoiseDomain {
        self.workspace.domain()
    }

    /// Returns the reusable proposal-buffer capacity.
    pub fn scratch_capacity(&self) -> usize {
        self.workspace.scratch_capacity()
    }
}

impl NoiseAlgorithm for ProportionalGaussian {
    type Error = NoisePluginError;

    fn is_noop(&self) -> bool {
        self.sigma() == 0.0
    }

    fn validate(
        &self,
        abundance: &AggregateAbundance,
        space: &SpatialAbundance,
    ) -> Result<(), Self::Error> {
        self.workspace
            .validate(abundance, space, GaussianKind::Proportional)
    }

    fn apply(
        &mut self,
        abundance: &mut AggregateAbundance,
        space: &mut SpatialAbundance,
        time_step: TimeStep,
    ) -> Result<(), Self::Error> {
        self.workspace
            .apply(abundance, space, time_step, GaussianKind::Proportional)
    }
}
