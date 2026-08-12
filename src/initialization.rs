//! Shared categorical initial-state input and explicit GLV conversion.

use ecological_initial_state::{
    InitialState, InitialStateArtifactDescriptor, InitialStateConfig, InitialStateError,
    PersistedInitialState, load_verified_initial_state, persist_initial_state,
};
use ndarray::{ArrayD, IxDyn};
use physics_in_parallel::space::discrete::square_lattice::SquareLatticeConfig;
use scientific_workflow::configuration::{ConfigurationError, TaskConfig};
use scientific_workflow::execution::ExecutionScope;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Source of one categorical ecological lattice.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpatialInitialStateSource {
    /// Generate or load a strict document through the shared crate.
    Config { config: InitialStateConfig },
    /// Verify a content-addressed artifact produced by an earlier execution.
    VerifiedArtifact {
        execution_directory_path_key: String,
        descriptor: InitialStateArtifactDescriptor,
    },
}

impl SpatialInitialStateSource {
    /// Resolves, validates, and republishes a categorical input in this scope.
    pub fn resolve(
        &self,
        task: &TaskConfig,
        scope: &ExecutionScope,
        lattice: SquareLatticeConfig,
        num_taxa: usize,
    ) -> Result<ResolvedSpatialInitialState, SpatialInitializationError> {
        let initial = match self {
            Self::Config { config } => {
                let resolved = config
                    .path_key()
                    .map(|key| task.resolve_path(key))
                    .transpose()?;
                config
                    .clone()
                    .create(lattice, num_taxa, resolved.as_deref())?
            }
            Self::VerifiedArtifact {
                execution_directory_path_key,
                descriptor,
            } => {
                if execution_directory_path_key.trim().is_empty() {
                    return Err(SpatialInitializationError::EmptyExecutionDirectoryPathKey);
                }
                let initial = load_verified_initial_state(
                    task.resolve_path(execution_directory_path_key)?,
                    descriptor,
                )?;
                if initial.space().config() != &lattice {
                    return Err(SpatialInitializationError::LatticeMismatch);
                }
                if initial.num_taxa() != num_taxa {
                    return Err(SpatialInitializationError::TaxonDimensionMismatch {
                        expected: num_taxa,
                        actual: initial.num_taxa(),
                    });
                }
                initial
            }
        };
        let persisted = persist_initial_state(scope, &initial)?;
        Ok(ResolvedSpatialInitialState { initial, persisted })
    }
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
    Configuration(#[from] ConfigurationError),
    #[error(transparent)]
    InitialState(#[from] InitialStateError),
    #[error("verified initial-state execution-directory path key must not be empty")]
    EmptyExecutionDirectoryPathKey,
    #[error("verified initial-state lattice does not match the GLV task")]
    LatticeMismatch,
    #[error("verified initial state declares {actual} taxa, expected {expected}")]
    TaxonDimensionMismatch { expected: usize, actual: usize },
    #[error("site_abundance must be positive and finite, got {value}")]
    InvalidSiteAbundance { value: f64 },
    #[error(transparent)]
    Shape(#[from] ndarray::ShapeError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use ecological_initial_state::DistributionSource;
    use physics_in_parallel::rng::RngConfig;

    #[test]
    fn categorical_conversion_is_species_last_and_explicitly_scaled() {
        let initial = InitialStateConfig::CenteredSeed {
            distribution: DistributionSource::Inline {
                weights: vec![1.0, 0.0],
            },
            seed_taxon: 1,
            seed_radius: 0,
            rng: RngConfig::new(Some(9), None, None),
        }
        .create(SquareLatticeConfig::periodic(&[3]), 2, None)
        .unwrap();
        let field = categorical_to_species_field(&initial, 2.5).unwrap();
        assert_eq!(field.shape(), &[3, 2]);
        for (site, &taxon) in initial.space().data().iter().enumerate() {
            assert_eq!(field[[site, taxon]], 2.5);
            assert_eq!(field[[site, 1 - taxon]], 0.0);
        }
    }

    #[test]
    fn verified_artifact_source_decodes_the_shared_descriptor() {
        let source: SpatialInitialStateSource = serde_json::from_str(
            r#"{
                "source":"verified_artifact",
                "execution_directory_path_key":"prior_execution",
                "descriptor":{
                    "format":"ecological.initial-state.v1",
                    "num_taxa":2,
                    "lattice":{"shape":[3],"boundary":"periodic","spacing":[1.0]},
                    "method":"random",
                    "path":"inputs/initial-state-digest.json",
                    "sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                }
            }"#,
        )
        .unwrap();
        assert!(matches!(
            source,
            SpatialInitialStateSource::VerifiedArtifact { .. }
        ));
    }
}
