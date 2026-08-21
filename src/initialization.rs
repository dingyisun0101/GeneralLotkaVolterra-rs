//! Shared categorical initial-state input and explicit GLV conversion.

use ecological_model_core::initial_state::{
    InitialState, InitialStateArtifactDescriptor, InitialStateError, InitialStateSource,
    PersistedInitialState, persist_initial_state,
};
use ndarray::{ArrayD, IxDyn};
use physics_in_parallel::prelude::basic::SquareLatticeConfig;
use scientific_workflow::prelude::basics::ExecutionScope;
use thiserror::Error;

/// Resolves, validates, and republishes one core categorical input.
pub fn resolve_spatial_initial_state(
    source: &InitialStateSource,
    scope: &ExecutionScope,
    lattice: SquareLatticeConfig,
    num_taxa: usize,
) -> Result<ResolvedSpatialInitialState, SpatialInitializationError> {
    let initial = source.resolve(lattice, num_taxa)?;
    let persisted = persist_initial_state(scope, &initial)?;
    Ok(ResolvedSpatialInitialState { initial, persisted })
}

/// One resolved shared input and its current-execution artifact identity.
pub struct ResolvedSpatialInitialState {
    initial: InitialState,
    persisted: PersistedInitialState,
}

impl ResolvedSpatialInitialState {
    pub const fn initial(&self) -> &InitialState {
        &self.initial
    }

    pub const fn descriptor(&self) -> &InitialStateArtifactDescriptor {
        self.persisted.descriptor()
    }

    pub fn into_initial(self) -> InitialState {
        self.initial
    }
}

/// Converts categorical sites to an explicit species-last one-hot field.
///
/// `site_abundance` controls the sole nonzero value at every site. Frequency
/// models use `1.0`; population models must supply their scientific scale.
pub fn categorical_to_species_field(
    initial: &InitialState,
    site_abundance: f64,
) -> Result<ArrayD<f64>, SpatialInitializationError> {
    if !site_abundance.is_finite() || site_abundance <= 0.0 {
        return Err(SpatialInitializationError::InvalidSiteAbundance {
            value: site_abundance,
        });
    }
    let mut shape = initial.space().config().shape().to_vec();
    shape.push(initial.num_taxa());
    let mut values = vec![0.0; initial.space().data().len() * initial.num_taxa()];
    for (site, &taxon) in initial.space().data().iter().enumerate() {
        values[site * initial.num_taxa() + taxon] = site_abundance;
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&shape), values)?)
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SpatialInitializationError {
    #[error(transparent)]
    InitialState(#[from] InitialStateError),
    #[error("site_abundance must be positive and finite, got {value}")]
    InvalidSiteAbundance { value: f64 },
    #[error(transparent)]
    Shape(#[from] ndarray::ShapeError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use ecological_model_core::initial_state::{DistributionSource, InitialStateRecipe};
    use physics_in_parallel::prelude::basic::RngConfig;

    #[test]
    fn categorical_conversion_is_species_last_and_explicitly_scaled() {
        let initial = InitialStateRecipe::CenteredSeed {
            distribution: DistributionSource::Inline {
                weights: vec![1.0, 0.0],
            },
            seed_taxon: 1,
            seed_radius: 0,
            rng: RngConfig::new(Some(9), None),
        }
        .create(SquareLatticeConfig::periodic(&[3]), 2)
        .unwrap();
        let field = categorical_to_species_field(&initial, 2.5).unwrap();
        assert_eq!(field.shape(), &[3, 2]);
        for (site, &taxon) in initial.space().data().iter().enumerate() {
            assert_eq!(field[[site, taxon]], 2.5);
            assert_eq!(field[[site, 1 - taxon]], 0.0);
        }
    }
}
