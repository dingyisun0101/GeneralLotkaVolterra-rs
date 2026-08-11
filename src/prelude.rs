//! Convenient imports for applications that construct and run GLV models.
//!
//! This prelude combines the crate's supported public API with the Workflow
//! orchestration types and ndarray containers used at its boundaries:
//!
//! ```
//! use general_lotka_volterra_rs::prelude::*;
//!
//! let interaction = InMemorySource::new(arr2(&[[0.0, 0.2], [-0.1, 0.0]]))
//!     .resolve(2)?;
//! let config = MeanFieldReplicatorConfig::new(
//!     Array1::zeros(2),
//!     1.0e-12,
//!     TimeStep::new(0.01)?,
//! );
//! let mut simulation = MeanFieldReplicator::new(
//!     Array1::from_vec(vec![0.6, 0.4]),
//!     interaction,
//!     config,
//! )?;
//! simulation.step()?;
//!
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Standard-library facilities and application-specific serialization traits
//! remain explicit imports in downstream code.

pub use crate::core::{TimeStep, TimeStepError};
pub use crate::invariant::{
    FrequencyInvariant, INVARIANT_TOLERANCE, InvariantError, InvariantPolicy, InvariantPolicyError,
    LocalFrequencyInvariant, PopulationInvariant, enforce_state, validate_state,
};
pub use crate::kernel::{
    ArtifactDisposition, BoundaryCondition, Diffusion, GeneratedSource, GeneratorProvenance,
    INTERACTION_MATRIX_FORMAT, INTERACTION_MATRIX_LAYOUT, INTERACTION_MATRIX_METADATA_KEY,
    InMemorySource, InteractionArtifactDescriptor, InteractionArtifactError,
    InteractionArtifactLoadError, InteractionGenerator, InteractionMatrix, InteractionProvenance,
    InteractionSource, InteractionSourceError, InteractionSourceKind, JsonInteractionSource,
    Kernel, KernelAlgorithm, KernelAlgorithmError, KernelCore, KernelCoreError, KernelStateView,
    KernelStepError, KernelUpdate, KernelUpdateError, MeanFieldReplicatorRk4, PersistedInteraction,
    SpatialGeneralLotkaVolterraRk2, SpatialReplicatorRk2, load_verified_interaction_matrix,
    persist_interaction_matrix,
};
pub use crate::noise::{
    DEMOGRAPHIC_GAUSSIAN_RNG_NAMESPACE, DemographicGaussian, NoNoise, Noise, NoiseAlgorithm,
    NoiseDomain, NoisePluginError, NoiseStepError, PROPORTIONAL_GAUSSIAN_RNG_NAMESPACE,
    ProportionalGaussian,
};
pub use crate::project::{GlvProjectError, load_glv_project, validate_glv_project};
pub use crate::reading::{
    GlvCheckpointVerificationError, glv_json_decoders, open_completed_glv_recording,
    verify_completed_glv_checkpoint,
};
pub use crate::recording::{
    ABUNDANCE_REPRESENTATION_METADATA_KEY, COMPLETED_ITERATION_METADATA_KEY, GlvRecording,
    GlvRecordingError, GlvRecordingMetadata, MODEL_KIND_METADATA_KEY, RecordingMetadataError,
    TASK_ORDINAL_METADATA_KEY, TERMINATION_REASON_METADATA_KEY, TerminationReason,
};
pub use crate::simulation::{
    DefaultSimulationBuildError, MeanFieldReplicator, MeanFieldReplicatorConfig,
    SimulationBuildError, SimulationKind, SpatialGeneralLotkaVolterra,
    SpatialGeneralLotkaVolterraConfig, SpatialReplicator, SpatialReplicatorConfig,
    StateAssemblyError,
};
pub use crate::{
    ABUNDANCE_FIELD, AbundanceRepresentation, AggregateAbundance, CHECKPOINT_STREAM, SIGNAL_STREAM,
    SPACE_FIELD, SPACE_STREAM, SpatialAbundance, TOTAL_FIELD, TotalAbundance, load_state_schema,
    state_schema_path,
};

// These are the collection types that appear in constructors and state
// payloads, plus the common array literals used to define small matrices.
pub use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn, ShapeError, arr1, arr2};
pub use physics_in_parallel::rng::{RngConfig, RngMethod};
pub use physics_in_parallel::space::discrete::square_lattice::SquareLatticeConfig;

// GLV delegates orchestration, time, state, and storage to Scientific
// Workflow. Re-exporting its supported prelude lets applications opt into one
// coherent import instead of coordinating two preludes.
pub use scientific_workflow::prelude::*;
