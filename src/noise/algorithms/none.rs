//! Deterministic zero-noise plugin.

use std::convert::Infallible;

use crate::{AggregateAbundance, SpatialAbundance, TimeStep};

use crate::noise::core::NoiseAlgorithm;

/// Zero-sized deterministic default that performs no stochastic update.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct NoNoise;

impl NoiseAlgorithm for NoNoise {
    type Error = Infallible;

    fn validate(
        &self,
        _abundance: &AggregateAbundance,
        _space: &SpatialAbundance,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn apply(
        &mut self,
        _abundance: &mut AggregateAbundance,
        _space: &mut SpatialAbundance,
        _time_step: TimeStep,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}
