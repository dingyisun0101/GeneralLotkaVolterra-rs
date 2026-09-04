//! Shared categorical initial-state input and explicit GLV conversion.

use ecological_state_toolkit::initial_state::InitialState;
use physics_in_parallel::prelude::basic::{Backend, Tensor, TensorError};
use thiserror::Error;

/// Converts categorical sites to an explicit species-last one-hot field.
///
/// `site_abundance` controls the sole nonzero value at every site. Frequency
/// models use `1.0`; population models must supply their scientific scale.
pub fn categorical_to_species_field(
    initial: &InitialState,
    site_abundance: f64,
) -> Result<Tensor<f64>, SpatialInitializationError> {
    if !site_abundance.is_finite() || site_abundance <= 0.0 {
        return Err(SpatialInitializationError::InvalidSiteAbundance {
            value: site_abundance,
        });
    }
    let mut shape = initial.space().geometry().shape().to_vec();
    shape.push(initial.num_taxa());
    let mut values = vec![0.0; initial.space().data().len() * initial.num_taxa()];
    for (site, &taxon) in initial.space().data().iter().enumerate() {
        values[site * initial.num_taxa() + taxon] = site_abundance;
    }
    Ok(Tensor::from_values(&shape, Backend::Dense, values)?)
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SpatialInitializationError {
    #[error("site_abundance must be positive and finite, got {value}")]
    InvalidSiteAbundance { value: f64 },
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use ecological_state_toolkit::initial_state::{DistributionSource, InitialStateRecipe};
    use physics_in_parallel::prelude::basic::{ResolvedRng, RngMethod, SquareLatticeGeometry};

    #[test]
    fn categorical_conversion_is_species_last_and_explicitly_scaled() {
        let initial = InitialStateRecipe::CenteredSeed {
            distribution: DistributionSource::Inline {
                weights: vec![1.0, 0.0],
            },
            seed_taxon: 1,
            seed_radius: 0,
            rng: ResolvedRng::new(9, RngMethod::IndexedSplitMix64),
        }
        .create(SquareLatticeGeometry::periodic(&[3]).unwrap(), 2)
        .unwrap();
        let field = categorical_to_species_field(&initial, 2.5).unwrap();
        assert_eq!(field.shape(), &[3, 2]);
        for (site, &taxon) in initial.space().data().iter().enumerate() {
            assert_eq!(field.get(&[site, taxon]).unwrap(), 2.5);
            assert_eq!(field.get(&[site, 1 - taxon]).unwrap(), 0.0);
        }
    }
}
