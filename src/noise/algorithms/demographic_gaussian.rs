//! Gaussian fluctuations scaled by the square root of local abundance.

use crate::{AggregateAbundance, SpatialAbundance, TimeStep};
use physics_in_parallel::prelude::basic::RngConfig;

use crate::noise::core::{
    GaussianKind, GaussianWorkspace, NoiseAlgorithm, NoiseDomain, NoisePluginError,
};
use scientific_workflow::prelude::basics::RngRecord;

/// Workflow metadata namespace for demographic Gaussian RNG provenance.
pub const DEMOGRAPHIC_GAUSSIAN_RNG_NAMESPACE: &str = "glv.noise.demographic_gaussian";

/// Seeded demographic Gaussian noise with fixed reusable scratch.
#[derive(Debug)]
pub struct DemographicGaussian {
    workspace: GaussianWorkspace,
}

impl DemographicGaussian {
    /// Creates a seeded demographic plugin for one fixed payload domain.
    pub fn new(sigma: f64, rng: RngConfig, domain: NoiseDomain) -> Result<Self, NoisePluginError> {
        Ok(Self {
            workspace: GaussianWorkspace::new(
                sigma,
                rng,
                domain,
                DEMOGRAPHIC_GAUSSIAN_RNG_NAMESPACE,
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

impl NoiseAlgorithm for DemographicGaussian {
    type Error = NoisePluginError;

    fn rng_record(&self) -> Option<&RngRecord> {
        Some(self.workspace.rng_record())
    }

    fn is_noop(&self) -> bool {
        self.sigma() == 0.0
    }

    fn validate(
        &self,
        abundance: &AggregateAbundance,
        space: &SpatialAbundance,
    ) -> Result<(), Self::Error> {
        self.workspace
            .validate(abundance, space, GaussianKind::Demographic)
    }

    fn apply(
        &mut self,
        abundance: &mut AggregateAbundance,
        space: &mut SpatialAbundance,
        time_step: TimeStep,
    ) -> Result<(), Self::Error> {
        self.workspace
            .apply(abundance, space, time_step, GaussianKind::Demographic)
    }
}
