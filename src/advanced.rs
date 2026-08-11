//! Curated API for authors of custom GLV project templates.
//!
//! Ordinary users should import [`crate::prelude`] instead. This layer exposes
//! the same model, plugin, interaction, and Workflow building blocks used by
//! the built-in [`crate::GlvTemplate`] implementations.

/// Imports for advanced users assembling a custom project template.
pub mod prelude {
    pub use crate::core::{TimeStep, TimeStepError};
    pub use crate::invariant::{
        FrequencyInvariant, INVARIANT_TOLERANCE, InvariantError, InvariantPolicy,
        InvariantPolicyError, LocalFrequencyInvariant, PopulationInvariant, enforce_state,
        validate_state,
    };
    pub use crate::kernel::{
        ArtifactDisposition, BoundaryCondition, Diffusion, GeneratedSource, GeneratorProvenance,
        INTERACTION_MATRIX_FORMAT, INTERACTION_MATRIX_LAYOUT, INTERACTION_MATRIX_METADATA_KEY,
        InMemorySource, InteractionArtifactDescriptor, InteractionArtifactError,
        InteractionArtifactLoadError, InteractionGenerator, InteractionMatrix,
        InteractionProvenance, InteractionSource, InteractionSourceError, InteractionSourceKind,
        JsonInteractionSource, Kernel, KernelAlgorithm, KernelAlgorithmError, KernelCore,
        KernelCoreError, KernelStateView, KernelStepError, KernelUpdate, KernelUpdateError,
        MeanFieldReplicatorRk4, PersistedInteraction, SpatialGeneralLotkaVolterraRk2,
        SpatialReplicatorRk2, load_verified_interaction_matrix, persist_interaction_matrix,
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
    pub use crate::template::{
        GlvProjectTemplate, GlvRunError, GlvTemplate, TemplateTaskError, run,
    };
    pub use crate::{
        ABUNDANCE_FIELD, AbundanceRepresentation, AggregateAbundance, CHECKPOINT_STREAM,
        SIGNAL_STREAM, SPACE_FIELD, SPACE_STREAM, SpatialAbundance, TOTAL_FIELD, TotalAbundance,
        load_state_schema, state_schema_path,
    };

    pub use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn, ShapeError, arr1, arr2};
    pub use physics_in_parallel::rng::{RngConfig, RngMethod};
    pub use physics_in_parallel::space::discrete::square_lattice::SquareLatticeConfig;
    pub use scientific_workflow::prelude::*;
}
