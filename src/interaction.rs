//! GLV adapter for shared ecological interaction matrices.

use std::path::Path;

use ecological_model_core::interaction as shared;
use ndarray::Array2;
use physics_in_parallel::math::prelude::{DenseMatrix, MatrixError};
use scientific_workflow::execution::ExecutionScope;
use scientific_workflow::rng_record::{RngRecord, RngRecordError};

pub use scientific_workflow::artifact::ArtifactDisposition;
pub use shared::{
    DiagonalPolicy, GeneratorProvenance, INTERACTION_GENERATOR_IDENTITY,
    INTERACTION_GENERATOR_RNG_NAMESPACE, INTERACTION_GENERATOR_VERSION, INTERACTION_MATRIX_FORMAT,
    INTERACTION_MATRIX_METADATA_KEY, InteractionArtifactDescriptor, InteractionArtifactError,
    InteractionArtifactLoadError, InteractionMatrixError, InteractionMatrixRecipe,
    InteractionProvenance, InteractionRecipeError, InteractionSourceKind,
    InteractionTransformation, MatrixNormalization, PersistedInteraction, SignStructure,
};

/// Shared core matrix with one GLV-local ndarray constructor.
#[derive(Clone, Debug)]
pub struct InteractionMatrix(shared::InteractionMatrix);

impl InteractionMatrix {
    pub fn from_array(values: Array2<f64>, species: usize) -> Result<Self, InteractionMatrixError> {
        let rows = values.nrows();
        let columns = values.ncols();
        let matrix = DenseMatrix::try_from_vec(rows, columns, values.iter().copied().collect())?;
        Ok(Self(shared::InteractionMatrix::from_matrix(
            matrix, species,
        )?))
    }

    pub fn from_matrix(
        values: DenseMatrix<f64>,
        species: usize,
    ) -> Result<Self, InteractionMatrixError> {
        Ok(Self(shared::InteractionMatrix::from_matrix(
            values, species,
        )?))
    }

    pub fn from_shared(
        values: std::sync::Arc<DenseMatrix<f64>>,
        species: usize,
    ) -> Result<Self, InteractionMatrixError> {
        Ok(Self(shared::InteractionMatrix::from_shared(
            values, species,
        )?))
    }

    pub fn from_labeled_matrix(
        values: DenseMatrix<f64>,
        species: usize,
        label: impl Into<String>,
    ) -> Result<Self, InteractionMatrixError> {
        Ok(Self(shared::InteractionMatrix::from_labeled_matrix(
            values, species, label,
        )?))
    }

    pub fn from_rows(rows: Vec<Vec<f64>>, species: usize) -> Result<Self, InteractionMatrixError> {
        Ok(Self(shared::InteractionMatrix::from_rows(rows, species)?))
    }

    pub fn load_json(
        path: impl Into<std::path::PathBuf>,
        species: usize,
    ) -> Result<Self, InteractionMatrixError> {
        Ok(Self(shared::InteractionMatrix::load_json(path, species)?))
    }

    pub fn from_generated(
        values: DenseMatrix<f64>,
        species: usize,
        generator: GeneratorProvenance,
    ) -> Result<Self, InteractionMatrixError> {
        Ok(Self(shared::InteractionMatrix::from_generated(
            values, species, generator,
        )?))
    }

    pub fn generate(
        species: usize,
        recipe: &InteractionMatrixRecipe,
    ) -> Result<Self, InteractionRecipeError> {
        Ok(Self(shared::InteractionMatrix::generate(species, recipe)?))
    }

    pub fn species(&self) -> usize {
        self.0.species()
    }
    pub fn values(&self) -> &DenseMatrix<f64> {
        self.0.values()
    }
    pub fn shared_values(&self) -> std::sync::Arc<DenseMatrix<f64>> {
        self.0.shared_values()
    }
    pub fn coefficient(&self, row: usize, column: usize) -> f64 {
        self.0.coefficient(row, column)
    }
    pub fn mul_vector_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), MatrixError> {
        self.0.mul_vector_into(input, output)
    }
    pub const fn provenance(&self) -> &InteractionProvenance {
        self.0.provenance()
    }
    pub fn antisymmetrize(&self) -> Result<Self, InteractionMatrixError> {
        Ok(Self(self.0.antisymmetrize()?))
    }
    pub fn scale(&self, scalar: f64) -> Result<Self, InteractionMatrixError> {
        Ok(Self(self.0.scale(scalar)?))
    }
    pub fn normalize(&self, threshold: f64) -> Result<Self, InteractionMatrixError> {
        Ok(Self(self.0.normalize(threshold)?))
    }
    pub fn generator_rng_record(&self) -> Result<Option<RngRecord>, RngRecordError> {
        self.0.generator_rng_record()
    }
    pub const fn as_core(&self) -> &shared::InteractionMatrix {
        &self.0
    }
    pub fn into_core(self) -> shared::InteractionMatrix {
        self.0
    }
}

impl From<shared::InteractionMatrix> for InteractionMatrix {
    fn from(value: shared::InteractionMatrix) -> Self {
        Self(value)
    }
}

pub fn persist_interaction_matrix(
    scope: &ExecutionScope,
    interaction: &InteractionMatrix,
) -> Result<PersistedInteraction, InteractionArtifactError> {
    shared::persist_interaction_matrix(scope, interaction.as_core())
}

pub fn load_verified_interaction_matrix(
    execution_directory: impl AsRef<Path>,
    descriptor: &InteractionArtifactDescriptor,
) -> Result<InteractionMatrix, InteractionArtifactLoadError> {
    shared::load_verified_interaction_matrix(execution_directory, descriptor).map(InteractionMatrix)
}
