//! Deterministic numerical-kernel plugins.
//!
//! A [`Kernel`] combines shared interaction behavior with one numerical
//! algorithm. Algorithms receive temporary typed borrows of the canonical
//! abundance payloads; they cannot replace the authoritative Workflow state or
//! advance its simulation time.

pub mod artifact;
pub mod core;
pub mod source;

pub use artifact::{
    ArtifactDisposition, INTERACTION_MATRIX_METADATA_KEY, InteractionArtifactDescriptor,
    InteractionArtifactError, PersistedInteraction, persist_interaction_matrix,
};
pub use core::{
    Kernel, KernelAlgorithm, KernelCore, KernelCoreError, KernelStateView, KernelStepError,
    KernelUpdate, KernelUpdateError,
};
pub use source::{
    GeneratedSource, GeneratorProvenance, GeneratorRandomness, INTERACTION_MATRIX_FORMAT,
    INTERACTION_MATRIX_LAYOUT, InMemorySource, InteractionGenerator, InteractionMatrix,
    InteractionProvenance, InteractionSource, InteractionSourceError, InteractionSourceKind,
    JsonInteractionSource,
};
