//! Gaussian fluctuations scaled by the square root of local abundance.

use crate::{AggregateAbundance, SpatialAbundance, TimeStep};

use super::core::{GaussianKind, GaussianWorkspace, NoiseAlgorithm, NoiseDomain, NoisePluginError};

/// Seeded demographic Gaussian noise with fixed reusable scratch.
#[derive(Debug)]
pub struct DemographicGaussian {
    workspace: GaussianWorkspace,
}

impl DemographicGaussian {
    /// Creates a seeded demographic plugin for one fixed payload domain.
    pub fn new(sigma: f64, seed: u64, domain: NoiseDomain) -> Result<Self, NoisePluginError> {
        Ok(Self {
            workspace: GaussianWorkspace::new(sigma, seed, domain)?,
        })
    }

    /// Returns the configured Gaussian strength.
    pub const fn sigma(&self) -> f64 {
        self.workspace.sigma()
    }

    /// Returns the seed used to initialize the owned RNG.
    pub const fn seed(&self) -> u64 {
        self.workspace.seed()
    }

    /// Borrows the fixed aggregate or spatial domain.
    pub const fn domain(&self) -> &NoiseDomain {
        self.workspace.domain()
    }

    /// Returns the reusable normal-sample and proposal capacities.
    pub fn scratch_capacities(&self) -> (usize, usize) {
        self.workspace.scratch_capacities()
    }
}

impl NoiseAlgorithm for DemographicGaussian {
    type Error = NoisePluginError;

    fn validate(
        &self,
        abundance: &AggregateAbundance,
        space: &SpatialAbundance,
    ) -> Result<(), Self::Error> {
        self.workspace.validate(abundance, space)
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
