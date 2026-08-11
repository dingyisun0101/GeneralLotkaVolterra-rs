//! Content-addressed persistence for resolved interaction matrices.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use scientific_workflow::execution::ExecutionScope;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::source::{
    GeneratorProvenance, INTERACTION_MATRIX_FORMAT, InteractionMatrix, InteractionSource,
    InteractionSourceError, InteractionSourceKind, JsonInteractionSource,
};

/// Creation-metadata key holding the compact matrix descriptor.
pub const INTERACTION_MATRIX_METADATA_KEY: &str = "interaction_matrix";

static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Compact descriptor stored once in each task's creation metadata.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct InteractionArtifactDescriptor {
    format: String,
    rows: usize,
    columns: usize,
    sha256: String,
    path: String,
    source_kind: InteractionSourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    generator: Option<GeneratorProvenance>,
}

impl InteractionArtifactDescriptor {
    /// Returns the stable artifact format.
    pub fn format(&self) -> &str {
        &self.format
    }

    /// Returns the matrix shape.
    pub const fn shape(&self) -> [usize; 2] {
        [self.rows, self.columns]
    }

    /// Returns the lowercase SHA-256 digest of the exact artifact bytes.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Returns the execution-relative artifact path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the source category without embedding matrix values.
    pub const fn source_kind(&self) -> InteractionSourceKind {
        self.source_kind
    }

    /// Returns generator provenance when the source was generated.
    pub const fn generator(&self) -> Option<&GeneratorProvenance> {
        self.generator.as_ref()
    }

    /// Inserts this descriptor under the stable creation-metadata key.
    ///
    /// The previous value is returned so orchestration can reject accidental
    /// key collisions instead of silently losing task metadata.
    pub fn insert_into_metadata(&self, metadata: &mut Map<String, Value>) -> Option<Value> {
        metadata.insert(
            INTERACTION_MATRIX_METADATA_KEY.to_owned(),
            serde_json::to_value(self)
                .expect("interaction artifact descriptors contain only JSON-compatible fields"),
        )
    }
}

/// Whether persistence created a new artifact or reused identical bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ArtifactDisposition {
    /// This call atomically published the artifact.
    Created,
    /// An identical content-addressed artifact already existed.
    Reused,
}

/// Result of persisting one resolved interaction matrix.
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

    /// Returns whether bytes were created or safely reused.
    pub const fn disposition(&self) -> ArtifactDisposition {
        self.disposition
    }

    /// Transfers ownership of the descriptor.
    pub fn into_descriptor(self) -> InteractionArtifactDescriptor {
        self.descriptor
    }
}

/// Persists a resolved matrix once beneath an execution scope's `inputs/` directory.
///
/// Exact canonical JSON bytes are hashed before a temporary file is written.
/// A hard link publishes that complete file atomically without replacing an
/// existing digest path. Existing bytes must match exactly before reuse.
pub fn persist_interaction_matrix(
    scope: &ExecutionScope,
    matrix: &InteractionMatrix,
) -> Result<PersistedInteraction, InteractionArtifactError> {
    let document = matrix.artifact_document();
    let bytes = serde_json::to_vec(&document).map_err(InteractionArtifactError::Serialize)?;
    let digest = sha256_hex(&bytes);
    let file_name = format!("interaction-{digest}.json");
    let relative_path = format!("inputs/{file_name}");
    let inputs = scope.directory().join("inputs");
    fs::create_dir_all(&inputs).map_err(|source| InteractionArtifactError::Io {
        operation: "create interaction input directory",
        path: inputs.clone(),
        source,
    })?;
    let destination = inputs.join(&file_name);

    if destination.exists() {
        verify_existing(&destination, &bytes, &digest)?;
        return Ok(persisted_descriptor(
            matrix,
            digest,
            relative_path,
            ArtifactDisposition::Reused,
        ));
    }

    let temporary = create_complete_temporary(&inputs, &digest, &bytes)?;
    let disposition = match fs::hard_link(&temporary, &destination) {
        Ok(()) => ArtifactDisposition::Created,
        Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
            let verification = verify_existing(&destination, &bytes, &digest);
            remove_published_temporary(&temporary)?;
            verification?;
            ArtifactDisposition::Reused
        }
        Err(source) => {
            remove_temporary(&temporary);
            return Err(InteractionArtifactError::Io {
                operation: "publish interaction artifact",
                path: destination,
                source,
            });
        }
    };
    if temporary.exists() {
        remove_published_temporary(&temporary)?;
    }
    sync_directory(&inputs)?;

    Ok(persisted_descriptor(
        matrix,
        digest,
        relative_path,
        disposition,
    ))
}

/// Loads one recorded interaction artifact after verifying its exact identity.
///
/// `execution_directory` is the execution-scope root against which the
/// descriptor's relative path was recorded. Exact SHA-256 verification occurs
/// before JSON decoding or matrix construction.
pub fn load_verified_interaction_matrix(
    execution_directory: impl AsRef<Path>,
    descriptor: &InteractionArtifactDescriptor,
) -> Result<InteractionMatrix, InteractionArtifactLoadError> {
    validate_descriptor(descriptor)?;
    let execution_directory = fs::canonicalize(execution_directory.as_ref()).map_err(|source| {
        InteractionArtifactLoadError::Io {
            operation: "resolve execution directory",
            path: execution_directory.as_ref().to_path_buf(),
            source,
        }
    })?;
    let relative = Path::new(descriptor.path());
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(InteractionArtifactLoadError::InvalidDescriptor {
            reason: "artifact path must be a nonempty normalized relative path".to_owned(),
        });
    }
    let unresolved = execution_directory.join(relative);
    let path =
        fs::canonicalize(&unresolved).map_err(|source| InteractionArtifactLoadError::Io {
            operation: "resolve interaction artifact",
            path: unresolved,
            source,
        })?;
    if !path.starts_with(&execution_directory) {
        return Err(InteractionArtifactLoadError::InvalidDescriptor {
            reason: "artifact path resolves outside the execution directory".to_owned(),
        });
    }
    let bytes = fs::read(&path).map_err(|source| InteractionArtifactLoadError::Io {
        operation: "read interaction artifact",
        path: path.clone(),
        source,
    })?;
    let actual = sha256_hex(&bytes);
    if actual != descriptor.sha256 {
        return Err(InteractionArtifactLoadError::DigestMismatch {
            path,
            expected: descriptor.sha256.clone(),
            actual,
        });
    }
    let matrix = JsonInteractionSource::resolved_file(&path).resolve(descriptor.rows)?;
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
    if descriptor.sha256.len() != 64
        || !descriptor
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(InteractionArtifactLoadError::InvalidDescriptor {
            reason: "artifact SHA-256 must contain exactly 64 lowercase hexadecimal digits"
                .to_owned(),
        });
    }
    Ok(())
}

fn persisted_descriptor(
    matrix: &InteractionMatrix,
    digest: String,
    path: String,
    disposition: ArtifactDisposition,
) -> PersistedInteraction {
    let species = matrix.species();
    PersistedInteraction {
        descriptor: InteractionArtifactDescriptor {
            format: INTERACTION_MATRIX_FORMAT.to_owned(),
            rows: species,
            columns: species,
            sha256: digest,
            path,
            source_kind: matrix.provenance().kind(),
            generator: matrix.provenance().generator().cloned(),
        },
        disposition,
    }
}

fn create_complete_temporary(
    directory: &Path,
    digest: &str,
    bytes: &[u8],
) -> Result<PathBuf, InteractionArtifactError> {
    for _ in 0..1024 {
        let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(
            ".interaction-{digest}-{}-{sequence}.tmp",
            std::process::id()
        ));
        let mut file = match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => file,
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(InteractionArtifactError::Io {
                    operation: "create temporary interaction artifact",
                    path,
                    source,
                });
            }
        };
        if let Err(source) = file.write_all(bytes).and_then(|()| file.sync_all()) {
            drop(file);
            remove_temporary(&path);
            return Err(InteractionArtifactError::Io {
                operation: "write temporary interaction artifact",
                path,
                source,
            });
        }
        return Ok(path);
    }
    Err(InteractionArtifactError::TemporaryIdentityExhausted {
        directory: directory.to_path_buf(),
    })
}

fn verify_existing(
    path: &Path,
    expected: &[u8],
    digest: &str,
) -> Result<(), InteractionArtifactError> {
    let actual = fs::read(path).map_err(|source| InteractionArtifactError::Io {
        operation: "read existing interaction artifact",
        path: path.to_path_buf(),
        source,
    })?;
    if actual == expected {
        Ok(())
    } else {
        Err(InteractionArtifactError::DigestCollision {
            digest: digest.to_owned(),
            path: path.to_path_buf(),
        })
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(encoded, "{byte:02x}").expect("writing into a String cannot fail");
    }
    encoded
}

fn sync_directory(path: &Path) -> Result<(), InteractionArtifactError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| InteractionArtifactError::Io {
            operation: "synchronize interaction input directory",
            path: path.to_path_buf(),
            source,
        })
}

fn remove_temporary(path: &Path) {
    let _ = fs::remove_file(path);
}

fn remove_published_temporary(path: &Path) -> Result<(), InteractionArtifactError> {
    fs::remove_file(path).map_err(|source| InteractionArtifactError::Io {
        operation: "remove temporary interaction artifact",
        path: path.to_path_buf(),
        source,
    })
}

/// Failure while serializing or publishing an interaction artifact.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InteractionArtifactError {
    /// Validated coefficients unexpectedly failed JSON serialization.
    #[error("could not serialize the resolved interaction matrix")]
    Serialize(#[source] serde_json::Error),
    /// A filesystem operation failed.
    #[error("failed to {operation} at `{path}`")]
    Io {
        /// Stable operation label.
        operation: &'static str,
        /// Affected path.
        path: PathBuf,
        /// Underlying failure.
        #[source]
        source: std::io::Error,
    },
    /// A digest-named file exists with different exact bytes.
    #[error("interaction artifact digest collision for `{digest}` at `{path}`")]
    DigestCollision {
        /// Expected content digest.
        digest: String,
        /// Conflicting path.
        path: PathBuf,
    },
    /// Repeated process-local temporary names were already occupied.
    #[error("could not allocate a temporary interaction artifact beneath `{directory}`")]
    TemporaryIdentityExhausted {
        /// Input artifact directory.
        directory: PathBuf,
    },
}

/// Failure while verifying and reconstructing a recorded interaction artifact.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InteractionArtifactLoadError {
    /// The compact metadata descriptor violates the artifact contract.
    #[error("invalid interaction artifact descriptor: {reason}")]
    InvalidDescriptor {
        /// Concise rejected invariant.
        reason: String,
    },
    /// A filesystem operation failed.
    #[error("failed to {operation} at `{path}`")]
    Io {
        /// Stable operation label.
        operation: &'static str,
        /// Affected path.
        path: PathBuf,
        /// Underlying failure.
        #[source]
        source: std::io::Error,
    },
    /// Exact artifact bytes disagree with the recorded identity.
    #[error(
        "interaction artifact `{path}` has SHA-256 `{actual}`, but metadata declares `{expected}`"
    )]
    DigestMismatch {
        /// Verified artifact path.
        path: PathBuf,
        /// Metadata-declared lowercase digest.
        expected: String,
        /// Digest computed over exact current bytes.
        actual: String,
    },
    /// Verified artifact bytes do not encode a valid interaction matrix.
    #[error(transparent)]
    Matrix(#[from] InteractionSourceError),
}
