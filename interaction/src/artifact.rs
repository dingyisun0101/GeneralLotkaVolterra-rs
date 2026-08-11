//! Interaction-matrix metadata over Workflow's generic artifact mechanics.

use std::path::{Path, PathBuf};

pub use scientific_workflow::artifact::ArtifactDisposition;
use scientific_workflow::artifact::{
    ArtifactDescriptor, ArtifactError, ArtifactLoadError, load_verified_artifact, persist_artifact,
};
use scientific_workflow::execution::ExecutionScope;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use super::source::{
    GeneratorProvenance, INTERACTION_MATRIX_FORMAT, InteractionMatrix, InteractionSourceError,
    InteractionSourceKind, resolve_json_bytes,
};

/// Creation-metadata key holding the compact matrix descriptor.
pub const INTERACTION_MATRIX_METADATA_KEY: &str = "interaction_matrix";

/// Compact matrix identity stored once in each consuming task's metadata.
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
    /// Returns the stable interaction-document format.
    pub fn format(&self) -> &str {
        &self.format
    }

    /// Returns `[rows, columns]`.
    pub const fn shape(&self) -> [usize; 2] {
        [self.rows, self.columns]
    }

    /// Returns the lowercase SHA-256 digest of the exact document bytes.
    pub fn sha256(&self) -> &str {
        self.artifact.sha256()
    }

    /// Returns the artifact path relative to its execution scope.
    pub fn path(&self) -> &str {
        self.artifact.path()
    }

    /// Returns the scientific source category retained at resolution.
    pub const fn source_kind(&self) -> InteractionSourceKind {
        self.source_kind
    }

    /// Returns typed generator provenance when the source was generated.
    pub const fn generator(&self) -> Option<&GeneratorProvenance> {
        self.generator.as_ref()
    }

    /// Inserts the descriptor under [`INTERACTION_MATRIX_METADATA_KEY`].
    ///
    /// The previous value is returned so orchestration can reject collisions.
    pub fn insert_into_metadata(&self, metadata: &mut Map<String, Value>) -> Option<Value> {
        metadata.insert(
            INTERACTION_MATRIX_METADATA_KEY.to_owned(),
            serde_json::to_value(self)
                .expect("interaction artifact descriptors contain only JSON-compatible fields"),
        )
    }
}

/// Result of publishing one resolved matrix through Workflow's artifact API.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistedInteraction {
    descriptor: InteractionArtifactDescriptor,
    disposition: ArtifactDisposition,
}

impl PersistedInteraction {
    /// Borrows the compact recording descriptor.
    pub const fn descriptor(&self) -> &InteractionArtifactDescriptor {
        &self.descriptor
    }

    /// Reports whether exact bytes were created or reused.
    pub const fn disposition(&self) -> ArtifactDisposition {
        self.disposition
    }

    /// Transfers ownership of the compact descriptor.
    pub fn into_descriptor(self) -> InteractionArtifactDescriptor {
        self.descriptor
    }
}

/// Serializes and atomically publishes one immutable interaction matrix.
///
/// Workflow owns hashing, publication, and reuse. This crate owns the
/// canonical matrix document and scientific provenance fields.
pub fn persist_interaction_matrix(
    scope: &ExecutionScope,
    matrix: &InteractionMatrix,
) -> Result<PersistedInteraction, InteractionArtifactError> {
    let bytes = serde_json::to_vec(&matrix.artifact_document())
        .map_err(InteractionArtifactError::Serialize)?;
    let persisted =
        persist_artifact(scope, "interaction", "json", &bytes).map_err(map_artifact_error)?;
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

/// Verifies exact artifact bytes before reconstructing the matrix.
pub fn load_verified_interaction_matrix(
    execution_directory: impl AsRef<Path>,
    descriptor: &InteractionArtifactDescriptor,
) -> Result<InteractionMatrix, InteractionArtifactLoadError> {
    validate_descriptor(descriptor)?;
    let verified = load_verified_artifact(execution_directory, &descriptor.artifact)
        .map_err(map_artifact_load_error)?;
    let path = verified.path().to_path_buf();
    let matrix = resolve_json_bytes(verified.into_bytes(), path, descriptor.rows)?;
    if matrix.species() != descriptor.rows || descriptor.columns != descriptor.rows {
        return Err(InteractionArtifactLoadError::InvalidDescriptor {
            reason: format!(
                "descriptor shape {}x{} does not match a square matrix",
                descriptor.rows, descriptor.columns
            ),
        });
    }
    Ok(matrix)
}

fn validate_descriptor(
    descriptor: &InteractionArtifactDescriptor,
) -> Result<(), InteractionArtifactLoadError> {
    if descriptor.format != INTERACTION_MATRIX_FORMAT {
        return Err(InteractionArtifactLoadError::InvalidDescriptor {
            reason: format!(
                "artifact format `{}` is not `{INTERACTION_MATRIX_FORMAT}`",
                descriptor.format
            ),
        });
    }
    if descriptor.rows == 0 || descriptor.rows != descriptor.columns {
        return Err(InteractionArtifactLoadError::InvalidDescriptor {
            reason: format!(
                "artifact shape {}x{} must be nonempty and square",
                descriptor.rows, descriptor.columns
            ),
        });
    }
    Ok(())
}

fn map_artifact_error(error: ArtifactError) -> InteractionArtifactError {
    match error {
        ArtifactError::Io {
            operation,
            path,
            source,
        } => InteractionArtifactError::Io {
            operation,
            path,
            source,
        },
        ArtifactError::DigestCollision { digest, path } => {
            InteractionArtifactError::DigestCollision { digest, path }
        }
        ArtifactError::TemporaryIdentityExhausted { directory } => {
            InteractionArtifactError::TemporaryIdentityExhausted { directory }
        }
        other => InteractionArtifactError::Workflow(other),
    }
}

fn map_artifact_load_error(error: ArtifactLoadError) -> InteractionArtifactLoadError {
    match error {
        ArtifactLoadError::InvalidDescriptor { reason } => {
            InteractionArtifactLoadError::InvalidDescriptor { reason }
        }
        ArtifactLoadError::Io {
            operation,
            path,
            source,
        } => InteractionArtifactLoadError::Io {
            operation,
            path,
            source,
        },
        ArtifactLoadError::DigestMismatch {
            path,
            expected,
            actual,
        } => InteractionArtifactLoadError::DigestMismatch {
            path,
            expected,
            actual,
        },
        other => InteractionArtifactLoadError::Workflow(other),
    }
}

/// Failure while encoding or publishing an interaction artifact.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InteractionArtifactError {
    #[error("could not serialize the resolved interaction matrix")]
    Serialize(#[source] serde_json::Error),
    #[error("failed to {operation} at `{path}`")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("interaction artifact digest collision for `{digest}` at `{path}`")]
    DigestCollision { digest: String, path: PathBuf },
    #[error("could not allocate a temporary interaction artifact beneath `{directory}`")]
    TemporaryIdentityExhausted { directory: PathBuf },
    #[error(transparent)]
    Workflow(ArtifactError),
}

/// Failure while verifying or reconstructing an interaction artifact.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InteractionArtifactLoadError {
    #[error("invalid interaction artifact descriptor: {reason}")]
    InvalidDescriptor { reason: String },
    #[error("failed to {operation} at `{path}`")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "interaction artifact `{path}` has SHA-256 `{actual}`, but metadata declares `{expected}`"
    )]
    DigestMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error(transparent)]
    Matrix(#[from] InteractionSourceError),
    #[error(transparent)]
    Workflow(ArtifactLoadError),
}
