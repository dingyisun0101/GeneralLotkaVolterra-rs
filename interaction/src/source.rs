//! Typed interaction-matrix sources and resolution provenance.

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::Array2;
use physics_in_parallel::rng::RngConfig;
use scientific_workflow::rng_record::{RngRecord, RngRecordError};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error as ThisError;

/// Stable format identifier for persisted interaction matrices.
pub const INTERACTION_MATRIX_FORMAT: &str = "glv.interaction-matrix.v1";

/// Stable logical layout of values in an interaction artifact.
pub const INTERACTION_MATRIX_LAYOUT: &str = "row_major";

/// Workflow metadata namespace for stochastic matrix generation.
pub const INTERACTION_GENERATOR_RNG_NAMESPACE: &str = "scientific_interaction.generator";

/// A validated immutable interaction matrix and its resolution provenance.
#[derive(Clone, Debug)]
pub struct InteractionMatrix {
    values: Arc<Array2<f64>>,
    provenance: InteractionProvenance,
}

impl InteractionMatrix {
    /// Returns the validated species dimension.
    pub fn species(&self) -> usize {
        self.values.nrows()
    }

    /// Borrows the exact resolved coefficients.
    pub fn values(&self) -> &Array2<f64> {
        &self.values
    }

    /// Clones only the shared allocation handle.
    pub fn shared_values(&self) -> Arc<Array2<f64>> {
        Arc::clone(&self.values)
    }

    /// Borrows the complete source provenance.
    pub const fn provenance(&self) -> &InteractionProvenance {
        &self.provenance
    }

    /// Returns a Workflow RNG record when this matrix came from a stochastic generator.
    pub fn generator_rng_record(&self) -> Result<Option<RngRecord>, RngRecordError> {
        self.provenance.generator_rng_record()
    }

    pub(crate) fn artifact_document(&self) -> InteractionMatrixDocument {
        let rows = self.values.nrows();
        let columns = self.values.ncols();
        let mut values = Vec::with_capacity(self.values.len());
        for row in 0..rows {
            for column in 0..columns {
                values.push(self.values[(row, column)]);
            }
        }
        InteractionMatrixDocument {
            format: INTERACTION_MATRIX_FORMAT.to_owned(),
            rows,
            columns,
            layout: INTERACTION_MATRIX_LAYOUT.to_owned(),
            values,
        }
    }
}

/// Stable category used in compact recording metadata.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionSourceKind {
    /// Matrix supplied directly by a Rust caller.
    InMemory,
    /// Matrix decoded from inline typed project configuration.
    InlineJson,
    /// Matrix read from an already-resolved JSON path.
    JsonFile,
    /// Matrix produced by a typed generator.
    Generated,
}

/// Complete provenance retained beside a resolved interaction matrix.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InteractionProvenance {
    /// Matrix supplied directly by a Rust caller.
    InMemory {
        /// Optional caller-owned identifier.
        label: Option<String>,
    },
    /// Values decoded by Workflow from inline project configuration.
    InlineJson,
    /// Versioned matrix document read from an already-resolved path.
    JsonFile {
        /// Exact path supplied after Workflow path resolution.
        path: PathBuf,
    },
    /// Matrix produced by a typed generator.
    Generated {
        /// Reproducibility information supplied by the generator type.
        generator: GeneratorProvenance,
    },
}

impl InteractionProvenance {
    /// Returns the compact source category.
    pub const fn kind(&self) -> InteractionSourceKind {
        match self {
            Self::InMemory { .. } => InteractionSourceKind::InMemory,
            Self::InlineJson => InteractionSourceKind::InlineJson,
            Self::JsonFile { .. } => InteractionSourceKind::JsonFile,
            Self::Generated { .. } => InteractionSourceKind::Generated,
        }
    }

    /// Returns generator provenance when this matrix was generated.
    pub const fn generator(&self) -> Option<&GeneratorProvenance> {
        match self {
            Self::Generated { generator } => Some(generator),
            _ => None,
        }
    }

    /// Returns a Workflow RNG record for stochastic generation only.
    pub fn generator_rng_record(&self) -> Result<Option<RngRecord>, RngRecordError> {
        let Some(generator) = self.generator() else {
            return Ok(None);
        };
        generator.rng_record()
    }
}

/// Reproducibility metadata for one generated matrix.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GeneratorProvenance {
    identity: String,
    version: String,
    parameters: Value,
    rng_config: Option<RngConfig>,
}

impl GeneratorProvenance {
    /// Returns the stable generator identity.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Returns the generator implementation version.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Borrows the exact JSON-compatible generator parameters.
    pub const fn parameters(&self) -> &Value {
        &self.parameters
    }

    /// Returns resolved RNG provenance, or `None` for deterministic generation.
    pub const fn rng_config(&self) -> Option<RngConfig> {
        self.rng_config
    }

    /// Converts resolved stochastic configuration into Workflow's lightweight provenance record.
    pub fn rng_record(&self) -> Result<Option<RngRecord>, RngRecordError> {
        let Some(rng) = self.rng_config else {
            return Ok(None);
        };
        let method = rng
            .method()
            .expect("generated sources validate resolved RNG methods");
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
            rng.encode_seed()
                .expect("generated sources validate resolved RNG seeds"),
            Some(parameters),
        )?))
    }
}

/// A consumed source that resolves one exact interaction matrix.
pub trait InteractionSource {
    /// Resolves and validates the matrix against the model species dimension.
    fn resolve(self, species: usize) -> Result<InteractionMatrix, InteractionSourceError>;
}

/// Checked source for direct programmatic callers and tests.
#[derive(Clone, Debug)]
pub struct InMemorySource {
    values: Arc<Array2<f64>>,
    label: Option<String>,
}

impl InMemorySource {
    /// Takes ownership of an ndarray without cloning its coefficients.
    pub fn new(values: Array2<f64>) -> Self {
        Self {
            values: Arc::new(values),
            label: None,
        }
    }

    /// Reuses an existing immutable allocation.
    pub fn from_shared(values: Arc<Array2<f64>>) -> Self {
        Self {
            values,
            label: None,
        }
    }

    /// Attaches an optional caller-owned provenance label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl InteractionSource for InMemorySource {
    fn resolve(self, species: usize) -> Result<InteractionMatrix, InteractionSourceError> {
        resolve_values(
            self.values,
            species,
            InteractionProvenance::InMemory { label: self.label },
        )
    }
}

/// JSON-backed source whose configuration has already been decoded by Workflow.
#[derive(Clone, Debug)]
pub enum JsonInteractionSource {
    /// Inline rows decoded from typed task configuration.
    Inline {
        /// Row-major nested values.
        rows: Vec<Vec<f64>>,
    },
    /// Versioned artifact at a path already resolved by `TaskConfig`.
    File {
        /// Exact resolved filesystem path.
        path: PathBuf,
    },
}

impl JsonInteractionSource {
    /// Creates an inline source from already-decoded rows.
    pub fn inline(rows: Vec<Vec<f64>>) -> Self {
        Self::Inline { rows }
    }

    /// Creates a file source from an already-resolved project path.
    pub fn resolved_file(path: impl Into<PathBuf>) -> Self {
        Self::File { path: path.into() }
    }
}

impl InteractionSource for JsonInteractionSource {
    fn resolve(self, species: usize) -> Result<InteractionMatrix, InteractionSourceError> {
        match self {
            Self::Inline { rows } => resolve_inline(rows, species),
            Self::File { path } => resolve_file(path, species),
        }
    }
}

/// Typed implementation contract for generated interaction matrices.
pub trait InteractionGenerator: Sized {
    /// Generator-specific failure.
    type Error: Error + Send + Sync + 'static;
    /// Typed parameter structure serialized into exact provenance.
    type Parameters: Serialize;

    /// Stable generator identity, independent of the Rust type name.
    const IDENTITY: &'static str;
    /// Stable implementation or algorithm version.
    const VERSION: &'static str;

    /// Borrows the exact parameters controlling generation.
    fn parameters(&self) -> &Self::Parameters;

    /// Returns the resolved universal RNG configuration, or `None` for deterministic generation.
    fn rng_config(&self) -> Option<RngConfig>;

    /// Consumes the configured generator and creates one matrix.
    fn generate(self, species: usize) -> Result<Array2<f64>, Self::Error>;
}

/// Source adapter for any typed [`InteractionGenerator`].
#[derive(Clone, Copy, Debug)]
pub struct GeneratedSource<G> {
    generator: G,
}

impl<G> GeneratedSource<G> {
    /// Wraps a configured generator for one-time resolution.
    pub const fn new(generator: G) -> Self {
        Self { generator }
    }
}

impl<G> InteractionSource for GeneratedSource<G>
where
    G: InteractionGenerator,
{
    fn resolve(self, species: usize) -> Result<InteractionMatrix, InteractionSourceError> {
        validate_generator_label("identity", G::IDENTITY)?;
        validate_generator_label("version", G::VERSION)?;
        let parameters = serde_json::to_value(self.generator.parameters()).map_err(|source| {
            InteractionSourceError::GeneratorParameters {
                identity: G::IDENTITY.to_owned(),
                source,
            }
        })?;
        let rng_config = self.generator.rng_config();
        if let Some(rng) = rng_config
            && (rng.seed().is_none() || rng.method().is_none())
        {
            return Err(InteractionSourceError::UnresolvedGeneratorRngConfig {
                identity: G::IDENTITY.to_owned(),
            });
        }
        let values = self.generator.generate(species).map_err(|source| {
            InteractionSourceError::Generator {
                identity: G::IDENTITY.to_owned(),
                source: Box::new(source),
            }
        })?;
        resolve_values(
            Arc::new(values),
            species,
            InteractionProvenance::Generated {
                generator: GeneratorProvenance {
                    identity: G::IDENTITY.to_owned(),
                    version: G::VERSION.to_owned(),
                    parameters,
                    rng_config,
                },
            },
        )
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InteractionMatrixDocument {
    pub(crate) format: String,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) layout: String,
    pub(crate) values: Vec<f64>,
}

fn resolve_inline(
    rows: Vec<Vec<f64>>,
    species: usize,
) -> Result<InteractionMatrix, InteractionSourceError> {
    let row_count = rows.len();
    let column_count = rows.first().map_or(0, Vec::len);
    for (row, values) in rows.iter().enumerate() {
        if values.len() != column_count {
            return Err(InteractionSourceError::RaggedRows {
                row,
                expected: column_count,
                actual: values.len(),
            });
        }
    }
    checked_element_count(row_count, column_count)?;
    let flattened = rows.into_iter().flatten().collect();
    let values =
        Array2::from_shape_vec((row_count, column_count), flattened).map_err(|source| {
            InteractionSourceError::InvalidShape {
                rows: row_count,
                columns: column_count,
                message: source.to_string(),
            }
        })?;
    resolve_values(Arc::new(values), species, InteractionProvenance::InlineJson)
}

fn resolve_file(
    path: PathBuf,
    species: usize,
) -> Result<InteractionMatrix, InteractionSourceError> {
    let bytes = fs::read(&path).map_err(|source| InteractionSourceError::Io {
        operation: "read interaction matrix",
        path: path.clone(),
        source,
    })?;
    let document: InteractionMatrixDocument =
        serde_json::from_slice(&bytes).map_err(|source| InteractionSourceError::Json {
            path: path.clone(),
            source,
        })?;
    if document.format != INTERACTION_MATRIX_FORMAT {
        return Err(InteractionSourceError::UnsupportedFormat {
            expected: INTERACTION_MATRIX_FORMAT,
            actual: document.format,
        });
    }
    if document.layout != INTERACTION_MATRIX_LAYOUT {
        return Err(InteractionSourceError::UnsupportedLayout {
            expected: INTERACTION_MATRIX_LAYOUT,
            actual: document.layout,
        });
    }
    let expected_values = checked_element_count(document.rows, document.columns)?;
    if document.values.len() != expected_values {
        return Err(InteractionSourceError::ElementCountMismatch {
            rows: document.rows,
            columns: document.columns,
            expected: expected_values,
            actual: document.values.len(),
        });
    }
    let values = Array2::from_shape_vec((document.rows, document.columns), document.values)
        .map_err(|source| InteractionSourceError::InvalidShape {
            rows: document.rows,
            columns: document.columns,
            message: source.to_string(),
        })?;
    resolve_values(
        Arc::new(values),
        species,
        InteractionProvenance::JsonFile { path },
    )
}

fn resolve_values(
    values: Arc<Array2<f64>>,
    species: usize,
    provenance: InteractionProvenance,
) -> Result<InteractionMatrix, InteractionSourceError> {
    if species == 0 {
        return Err(InteractionSourceError::EmptySpecies);
    }
    let rows = values.nrows();
    let columns = values.ncols();
    let expected_values = checked_element_count(rows, columns)?;
    if values.len() != expected_values {
        return Err(InteractionSourceError::ElementCountMismatch {
            rows,
            columns,
            expected: expected_values,
            actual: values.len(),
        });
    }
    if rows != columns {
        return Err(InteractionSourceError::NonSquare { rows, columns });
    }
    if rows != species {
        return Err(InteractionSourceError::SpeciesMismatch {
            expected: species,
            actual: rows,
        });
    }
    if let Some(((row, column), value)) =
        values.indexed_iter().find(|(_, value)| !value.is_finite())
    {
        return Err(InteractionSourceError::NonFiniteEntry {
            row,
            column,
            value: *value,
        });
    }
    Ok(InteractionMatrix { values, provenance })
}

fn checked_element_count(rows: usize, columns: usize) -> Result<usize, InteractionSourceError> {
    rows.checked_mul(columns)
        .ok_or(InteractionSourceError::ElementCountOverflow { rows, columns })
}

fn validate_generator_label(
    field: &'static str,
    value: &'static str,
) -> Result<(), InteractionSourceError> {
    if value.trim().is_empty() {
        Err(InteractionSourceError::InvalidGeneratorLabel { field })
    } else {
        Ok(())
    }
}

/// Failure while resolving or validating an interaction matrix.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum InteractionSourceError {
    /// Stochastic generator provenance must contain its actual seed and method.
    #[error("interaction generator `{identity}` returned unresolved RNG configuration")]
    UnresolvedGeneratorRngConfig {
        /// Generator identity.
        identity: String,
    },
    /// Models require at least one species.
    #[error("interaction matrix species dimension must be greater than zero")]
    EmptySpecies,
    /// A square scientific interaction matrix was not supplied.
    #[error("interaction matrix must be square, found {rows} rows and {columns} columns")]
    NonSquare {
        /// Row count.
        rows: usize,
        /// Column count.
        columns: usize,
    },
    /// The matrix dimension disagrees with the model domain.
    #[error("interaction matrix has {actual} species, expected {expected}")]
    SpeciesMismatch {
        /// Model species dimension.
        expected: usize,
        /// Matrix species dimension.
        actual: usize,
    },
    /// A shape cannot be represented as a platform-sized element count.
    #[error("interaction matrix shape {rows}x{columns} overflows its element count")]
    ElementCountOverflow {
        /// Declared rows.
        rows: usize,
        /// Declared columns.
        columns: usize,
    },
    /// A document's flattened values do not match its declared shape.
    #[error("interaction matrix shape {rows}x{columns} requires {expected} values, found {actual}")]
    ElementCountMismatch {
        /// Declared rows.
        rows: usize,
        /// Declared columns.
        columns: usize,
        /// Checked required count.
        expected: usize,
        /// Supplied count.
        actual: usize,
    },
    /// Inline rows do not share one column dimension.
    #[error("interaction matrix row {row} has {actual} columns, expected {expected}")]
    RaggedRows {
        /// Zero-based row index.
        row: usize,
        /// First-row column count.
        expected: usize,
        /// Rejected row column count.
        actual: usize,
    },
    /// Ndarray rejected a checked shape conversion.
    #[error("could not construct interaction matrix shape {rows}x{columns}: {message}")]
    InvalidShape {
        /// Row count.
        rows: usize,
        /// Column count.
        columns: usize,
        /// Ndarray diagnostic.
        message: String,
    },
    /// Every coefficient must be finite before evolution.
    #[error("interaction matrix entry ({row}, {column}) is not finite: {value}")]
    NonFiniteEntry {
        /// Zero-based row index.
        row: usize,
        /// Zero-based column index.
        column: usize,
        /// Rejected coefficient.
        value: f64,
    },
    /// A file used an unknown artifact format.
    #[error("unsupported interaction matrix format `{actual}`; expected `{expected}`")]
    UnsupportedFormat {
        /// Supported format.
        expected: &'static str,
        /// Rejected format.
        actual: String,
    },
    /// A file used an unknown logical value layout.
    #[error("unsupported interaction matrix layout `{actual}`; expected `{expected}`")]
    UnsupportedLayout {
        /// Supported layout.
        expected: &'static str,
        /// Rejected layout.
        actual: String,
    },
    /// Filesystem access failed.
    #[error("failed to {operation} at `{path}`")]
    Io {
        /// Stable operation label.
        operation: &'static str,
        /// Affected path.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// A versioned JSON document was malformed.
    #[error("invalid interaction matrix JSON at `{path}`")]
    Json {
        /// Source path.
        path: PathBuf,
        /// JSON failure.
        #[source]
        source: serde_json::Error,
    },
    /// A generator identity or version was empty.
    #[error("interaction generator {field} must not be empty")]
    InvalidGeneratorLabel {
        /// Rejected metadata field.
        field: &'static str,
    },
    /// Typed generator parameters could not be represented as JSON metadata.
    #[error("could not serialize parameters for interaction generator `{identity}`")]
    GeneratorParameters {
        /// Generator identity.
        identity: String,
        /// Serialization failure.
        #[source]
        source: serde_json::Error,
    },
    /// Typed generation failed.
    #[error("interaction generator `{identity}` failed")]
    Generator {
        /// Generator identity.
        identity: String,
        /// Generator-specific failure.
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },
}
