//! Shared interaction-matrix representation, sources, and artifact provenance.
//!
//! Model crates own interpretation and admissible coefficient ranges. This
//! crate owns only immutable representation, source resolution, generated
//! matrix provenance, and content-addressed persistence through Workflow
//! execution scopes.

mod artifact;
mod source;

pub use artifact::{
    ArtifactDisposition, INTERACTION_MATRIX_METADATA_KEY, InteractionArtifactDescriptor,
    InteractionArtifactError, InteractionArtifactLoadError, PersistedInteraction,
    load_verified_interaction_matrix, persist_interaction_matrix,
};
pub use source::{
    GeneratedSource, GeneratorProvenance, GeneratorRandomness, INTERACTION_GENERATOR_KEY_ENCODING,
    INTERACTION_GENERATOR_RNG_NAMESPACE, INTERACTION_MATRIX_FORMAT, INTERACTION_MATRIX_LAYOUT,
    InMemorySource, InteractionGenerator, InteractionMatrix, InteractionProvenance,
    InteractionSource, InteractionSourceError, InteractionSourceKind, JsonInteractionSource,
};
