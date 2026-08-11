//! Gaussian fluctuations proportional to local abundance.

use crate::{AggregateAbundance, SpatialAbundance, TimeStep};

use super::core::{GaussianKind, GaussianWorkspace, NoiseAlgorithm, NoiseDomain, NoisePluginError};

/// Seeded proportional Gaussian noise with fixed reusable scratch.
#[derive(Debug)]
pub struct ProportionalGaussian {
    workspace: GaussianWorkspace,
}

impl ProportionalGaussian {
    /// Creates a seeded proportional plugin for one fixed payload domain.
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

impl NoiseAlgorithm for ProportionalGaussian {
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
            .apply(abundance, space, time_step, GaussianKind::Proportional)
    }
}
