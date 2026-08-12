//! GLV interaction matrices over PiP numerical storage and Workflow artifacts.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::Array2;
use physics_in_parallel::math::prelude::{DenseMatrix, MatrixError};
use physics_in_parallel::rng::RngConfig;
pub use scientific_workflow::artifact::ArtifactDisposition;
use scientific_workflow::artifact::{
    ArtifactDescriptor, ArtifactError, ArtifactLoadError, load_verified_artifact, persist_artifact,
};
use scientific_workflow::execution::ExecutionScope;
use scientific_workflow::rng_record::{RngRecord, RngRecordError};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

/// PiP's versioned dense-matrix document used by GLV interaction artifacts.
pub const INTERACTION_MATRIX_FORMAT: &str = "pip.matrix.v1";

/// Workflow metadata namespace for stochastic interaction generation.
pub const INTERACTION_GENERATOR_RNG_NAMESPACE: &str = "glv.interaction_generator";

/// Creation-metadata key holding the interaction artifact descriptor.
pub const INTERACTION_MATRIX_METADATA_KEY: &str = "interaction_matrix";

/// A validated immutable GLV interaction matrix and its source provenance.
#[derive(Clone, Debug)]
pub struct InteractionMatrix {
    values: Arc<DenseMatrix<f64>>,
    provenance: InteractionProvenance,
}

impl InteractionMatrix {
    /// Converts an ndarray into PiP storage and validates it for a GLV model.
    pub fn from_array(values: Array2<f64>, species: usize) -> Result<Self, InteractionMatrixError> {
        let rows = values.nrows();
        let columns = values.ncols();
        let values = DenseMatrix::try_from_vec(rows, columns, values.iter().copied().collect())?;
        Self::from_matrix(values, species)
    }

    /// Validates and takes ownership of a PiP dense matrix.
    pub fn from_matrix(
        values: DenseMatrix<f64>,
        species: usize,
    ) -> Result<Self, InteractionMatrixError> {
        Self::resolve(
            Arc::new(values),
            species,
            InteractionProvenance::InMemory { label: None },
        )
    }

    /// Validates and reuses an existing immutable PiP matrix allocation.
    pub fn from_shared(
        values: Arc<DenseMatrix<f64>>,
        species: usize,
    ) -> Result<Self, InteractionMatrixError> {
        Self::resolve(
            values,
            species,
            InteractionProvenance::InMemory { label: None },
        )
    }

    /// Validates a PiP matrix and retains a caller-owned provenance label.
    pub fn from_labeled_matrix(
        values: DenseMatrix<f64>,
        species: usize,
        label: impl Into<String>,
    ) -> Result<Self, InteractionMatrixError> {
        Self::resolve(
            Arc::new(values),
            species,
            InteractionProvenance::InMemory {
                label: Some(label.into()),
            },
        )
    }

    /// Constructs an interaction matrix from inline row values.
    pub fn from_rows(rows: Vec<Vec<f64>>, species: usize) -> Result<Self, InteractionMatrixError> {
        let row_count = rows.len();
        let column_count = rows.first().map_or(0, Vec::len);
        for (row, values) in rows.iter().enumerate() {
            if values.len() != column_count {
                return Err(InteractionMatrixError::RaggedRows {
                    row,
                    expected: column_count,
                    actual: values.len(),
                });
            }
        }
        let values = DenseMatrix::try_from_vec(
            row_count,
            column_count,
            rows.into_iter().flatten().collect(),
        )?;
        Self::resolve(Arc::new(values), species, InteractionProvenance::Inline)
    }

    /// Loads PiP's versioned dense-matrix JSON from an already-resolved path.
    pub fn load_json(
        path: impl Into<PathBuf>,
        species: usize,
    ) -> Result<Self, InteractionMatrixError> {
        let path = path.into();
        let bytes = fs::read(&path).map_err(|source| InteractionMatrixError::Io {
            path: path.clone(),
            source,
        })?;
        Self::from_json_bytes(bytes, path, species)
    }

    /// Attaches explicit generator provenance to an already-generated matrix.
    pub fn from_generated(
        values: DenseMatrix<f64>,
        species: usize,
        generator: GeneratorProvenance,
    ) -> Result<Self, InteractionMatrixError> {
        Self::resolve(
            Arc::new(values),
            species,
            InteractionProvenance::Generated { generator },
        )
    }

    /// Returns the validated species dimension.
    pub fn species(&self) -> usize {
        self.values.rows()
    }

    /// Borrows the PiP dense matrix.
    pub fn values(&self) -> &DenseMatrix<f64> {
        &self.values
    }

    /// Clones only the shared allocation handle.
    pub fn shared_values(&self) -> Arc<DenseMatrix<f64>> {
        Arc::clone(&self.values)
    }

    /// Returns one interaction coefficient.
    #[inline]
    pub fn coefficient(&self, row: usize, column: usize) -> f64 {
        self.values.get(row as isize, column as isize)
    }

    /// Computes `output = interaction * input` without allocating.
    #[inline]
    pub fn mul_vector_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), MatrixError> {
        self.values.mul_vector_into(input, output)
    }

    /// Borrows the complete source provenance.
    pub const fn provenance(&self) -> &InteractionProvenance {
        &self.provenance
    }

    /// Returns a Workflow RNG record when a stochastic generator was used.
    pub fn generator_rng_record(&self) -> Result<Option<RngRecord>, RngRecordError> {
        self.provenance.generator_rng_record()
    }

    fn from_json_bytes(
        bytes: Vec<u8>,
        path: PathBuf,
        species: usize,
    ) -> Result<Self, InteractionMatrixError> {
        let values =
            serde_json::from_slice(&bytes).map_err(|source| InteractionMatrixError::Json {
                path: path.clone(),
                source,
            })?;
        Self::resolve(
            Arc::new(values),
            species,
            InteractionProvenance::JsonFile { path },
        )
    }

    fn resolve(
        values: Arc<DenseMatrix<f64>>,
        species: usize,
        provenance: InteractionProvenance,
    ) -> Result<Self, InteractionMatrixError> {
        if species == 0 {
            return Err(InteractionMatrixError::EmptySpecies);
        }
        let rows = values.rows();
        let columns = values.cols();
        if rows != columns {
            return Err(InteractionMatrixError::NonSquare { rows, columns });
        }
        if rows != species {
            return Err(InteractionMatrixError::SpeciesMismatch {
                expected: species,
                actual: rows,
            });
        }
        for row in 0..rows {
            for column in 0..columns {
                let value = values.get(row as isize, column as isize);
                if !value.is_finite() {
                    return Err(InteractionMatrixError::NonFiniteEntry { row, column, value });
                }
            }
        }
        Ok(Self { values, provenance })
    }
}

/// Stable category retained in compact recording metadata.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionSourceKind {
    InMemory,
    Inline,
    JsonFile,
    Generated,
}

/// Complete provenance for a resolved GLV interaction matrix.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InteractionProvenance {
    InMemory { label: Option<String> },
    Inline,
    JsonFile { path: PathBuf },
    Generated { generator: GeneratorProvenance },
}

impl InteractionProvenance {
    pub const fn kind(&self) -> InteractionSourceKind {
        match self {
            Self::InMemory { .. } => InteractionSourceKind::InMemory,
            Self::Inline => InteractionSourceKind::Inline,
            Self::JsonFile { .. } => InteractionSourceKind::JsonFile,
            Self::Generated { .. } => InteractionSourceKind::Generated,
        }
    }

    pub const fn generator(&self) -> Option<&GeneratorProvenance> {
        match self {
            Self::Generated { generator } => Some(generator),
            _ => None,
        }
    }

    pub fn generator_rng_record(&self) -> Result<Option<RngRecord>, RngRecordError> {
        self.generator()
            .map_or(Ok(None), GeneratorProvenance::rng_record)
    }
}

/// Reproducibility metadata for an externally generated matrix.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GeneratorProvenance {
    identity: String,
    version: String,
    parameters: Value,
    rng_config: Option<RngConfig>,
}

impl GeneratorProvenance {
    pub fn new(
        identity: impl Into<String>,
        version: impl Into<String>,
        parameters: Value,
        rng_config: Option<RngConfig>,
    ) -> Result<Self, InteractionMatrixError> {
        let identity = identity.into();
        let version = version.into();
        if identity.trim().is_empty() {
            return Err(InteractionMatrixError::InvalidGeneratorLabel { field: "identity" });
        }
        if version.trim().is_empty() {
            return Err(InteractionMatrixError::InvalidGeneratorLabel { field: "version" });
        }
        if rng_config.is_some_and(|rng| rng.seed().is_none() || rng.method().is_none()) {
            return Err(InteractionMatrixError::UnresolvedGeneratorRngConfig { identity });
        }
        Ok(Self {
            identity,
            version,
            parameters,
            rng_config,
        })
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub const fn parameters(&self) -> &Value {
        &self.parameters
    }

    pub const fn rng_config(&self) -> Option<RngConfig> {
        self.rng_config
    }

    pub fn rng_record(&self) -> Result<Option<RngRecord>, RngRecordError> {
        let Some(rng) = self.rng_config else {
            return Ok(None);
        };
        let method = rng
            .method()
            .expect("generator RNG configuration is resolved");
        let mut parameters = Map::new();
        parameters.insert("generator_parameters".to_owned(), self.parameters.clone());
        if let Some(streams) = rng.parallel_streams() {
            parameters.insert("parallel_streams".to_owned(), Value::from(streams.get()));
        }
        Ok(Some(RngRecord::new(
            INTERACTION_GENERATOR_RNG_NAMESPACE,
            format!("{}+{}", self.identity, method.name()),
            format!("{}+{}", self.version, method.version()),
            method.seed_encoding(),
            rng.encode_seed().expect("generator RNG seed is resolved"),
            Some(parameters),
        )?))
    }
}

/// Compact identity for a content-addressed interaction artifact.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct InteractionArtifactDescriptor {
    format: String,
    rows: usize,
    columns: usize,
    #[serde(flatten)]
    artifact: ArtifactDescriptor,
    source_kind: InteractionSourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    generator: Option<GeneratorProvenance>,
}

impl InteractionArtifactDescriptor {
    pub fn format(&self) -> &str {
        &self.format
    }

    pub const fn shape(&self) -> [usize; 2] {
        [self.rows, self.columns]
    }

    pub fn sha256(&self) -> &str {
        self.artifact.sha256()
    }

    pub fn path(&self) -> &str {
        self.artifact.path()
    }

    pub const fn source_kind(&self) -> InteractionSourceKind {
        self.source_kind
    }

    pub const fn generator(&self) -> Option<&GeneratorProvenance> {
        self.generator.as_ref()
    }

    pub fn insert_into_metadata(&self, metadata: &mut Map<String, Value>) -> Option<Value> {
        metadata.insert(
            INTERACTION_MATRIX_METADATA_KEY.to_owned(),
            serde_json::to_value(self).expect("interaction descriptor is JSON-compatible"),
        )
    }
}

/// Result of publishing one resolved interaction through Workflow.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistedInteraction {
    descriptor: InteractionArtifactDescriptor,
    disposition: ArtifactDisposition,
}

impl PersistedInteraction {
    pub const fn descriptor(&self) -> &InteractionArtifactDescriptor {
        &self.descriptor
    }

    pub const fn disposition(&self) -> ArtifactDisposition {
        self.disposition
    }

    pub fn into_descriptor(self) -> InteractionArtifactDescriptor {
        self.descriptor
    }
}

/// Serializes through PiP and atomically publishes through Workflow.
pub fn persist_interaction_matrix(
    scope: &ExecutionScope,
    matrix: &InteractionMatrix,
) -> Result<PersistedInteraction, InteractionArtifactError> {
    let bytes = serde_json::to_vec(matrix.values())?;
    let persisted = persist_artifact(scope, "interaction", "json", &bytes)?;
    let species = matrix.species();
    Ok(PersistedInteraction {
        descriptor: InteractionArtifactDescriptor {
            format: INTERACTION_MATRIX_FORMAT.to_owned(),
            rows: species,
            columns: species,
            artifact: persisted.descriptor().clone(),
            source_kind: matrix.provenance().kind(),
            generator: matrix.provenance().generator().cloned(),
        },
        disposition: persisted.disposition(),
    })
}

/// Verifies exact artifact bytes before decoding PiP's matrix schema.
pub fn load_verified_interaction_matrix(
    execution_directory: impl AsRef<Path>,
    descriptor: &InteractionArtifactDescriptor,
) -> Result<InteractionMatrix, InteractionArtifactLoadError> {
    if descriptor.format != INTERACTION_MATRIX_FORMAT
        || descriptor.rows == 0
        || descriptor.rows != descriptor.columns
    {
        return Err(InteractionArtifactLoadError::InvalidDescriptor);
    }
    let verified = load_verified_artifact(execution_directory, &descriptor.artifact)?;
    let path = verified.path().to_path_buf();
    Ok(InteractionMatrix::from_json_bytes(
        verified.into_bytes(),
        path,
        descriptor.rows,
    )?)
}

/// Failure while constructing or loading a GLV interaction matrix.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InteractionMatrixError {
    #[error(transparent)]
    Matrix(#[from] MatrixError),
    #[error("interaction matrix species dimension must be greater than zero")]
    EmptySpecies,
    #[error("interaction matrix must be square, found {rows}x{columns}")]
    NonSquare { rows: usize, columns: usize },
    #[error("interaction matrix has {actual} species, expected {expected}")]
    SpeciesMismatch { expected: usize, actual: usize },
    #[error("interaction matrix row {row} has {actual} columns, expected {expected}")]
    RaggedRows {
        row: usize,
        expected: usize,
        actual: usize,
    },
    #[error("interaction matrix entry ({row}, {column}) is not finite: {value}")]
    NonFiniteEntry {
        row: usize,
        column: usize,
        value: f64,
    },
    #[error("failed to read interaction matrix at `{path}`")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid PiP matrix JSON at `{path}`")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("interaction generator {field} must not be empty")]
    InvalidGeneratorLabel { field: &'static str },
    #[error("interaction generator `{identity}` has unresolved RNG configuration")]
    UnresolvedGeneratorRngConfig { identity: String },
}

/// Failure while encoding or publishing an interaction artifact.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InteractionArtifactError {
    #[error("could not serialize the interaction matrix")]
    Serialize(#[from] serde_json::Error),
    #[error(transparent)]
    Workflow(#[from] ArtifactError),
}

/// Failure while verifying or reconstructing an interaction artifact.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InteractionArtifactLoadError {
    #[error("invalid interaction artifact descriptor")]
    InvalidDescriptor,
    #[error(transparent)]
    Workflow(#[from] ArtifactLoadError),
    #[error(transparent)]
    Matrix(#[from] InteractionMatrixError),
}
