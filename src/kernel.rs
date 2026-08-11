//! Deterministic numerical-kernel plugins.
//!
//! A [`Kernel`] combines shared interaction behavior with one numerical
//! algorithm. Algorithms receive temporary typed borrows of the canonical
//! abundance payloads; they cannot replace the authoritative Workflow state or
//! advance its simulation time.

mod algorithms;
pub mod core;

pub use algorithms::{
    BoundaryCondition, Diffusion, KernelAlgorithmError, MeanFieldReplicatorRk4,
    SpatialGeneralLotkaVolterraRk2, SpatialReplicatorRk2,
};
pub use core::{
    Kernel, KernelAlgorithm, KernelCore, KernelCoreError, KernelStateView, KernelStepError,
    KernelUpdate, KernelUpdateError,
};
pub use scientific_interaction::{
    ArtifactDisposition, GeneratedSource, GeneratorProvenance, INTERACTION_GENERATOR_RNG_NAMESPACE,
    INTERACTION_MATRIX_FORMAT, INTERACTION_MATRIX_LAYOUT, INTERACTION_MATRIX_METADATA_KEY,
    InMemorySource, InteractionArtifactDescriptor, InteractionArtifactError,
    InteractionArtifactLoadError, InteractionGenerator, InteractionMatrix, InteractionProvenance,
    InteractionSource, InteractionSourceError, InteractionSourceKind, JsonInteractionSource,
    PersistedInteraction, load_verified_interaction_matrix, persist_interaction_matrix,
};
