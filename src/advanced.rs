//! Curated API for authors of custom GLV scientific compositions.
//!
//! Ordinary users should import [`crate::prelude`] instead. This layer exposes
//! the same model, plugin, interaction, and Workflow building blocks used by
//! [`crate::GlvUnit`].

/// Imports for advanced users assembling a custom simulation.
pub mod prelude {
    pub use crate::core::{TimeStep, TimeStepError};
    pub use crate::initialization::{SpatialInitializationError, categorical_to_species_field};
    pub use crate::interaction::{
        GeneratorProvenance, INTERACTION_MATRIX_FORMAT, INTERACTION_MATRIX_METADATA_KEY,
        InteractionArtifactDescriptor, InteractionArtifactError, InteractionArtifactLoadError,
        InteractionMatrix, InteractionMatrixError, InteractionProvenance, InteractionSourceKind,
        PersistedInteraction, load_verified_interaction_matrix, persist_interaction_matrix,
    };
    pub use crate::invariant::{
        FrequencyInvariant, INVARIANT_TOLERANCE, InvariantError, InvariantPolicy,
        InvariantPolicyError, LocalFrequencyInvariant, PopulationInvariant, enforce_state,
        validate_state,
    };
    pub use crate::kernel::{
        BoundaryCondition, Diffusion, Kernel, KernelAlgorithm, KernelAlgorithmError, KernelCore,
        KernelCoreError, KernelResidual, KernelResidualError, KernelStateView, KernelStepError,
        KernelUpdate, KernelUpdateError, MeanFieldReplicatorRk4, SpatialGeneralLotkaVolterraRk2,
        SpatialReplicatorRk2,
    };
    pub use crate::noise::{
        DEMOGRAPHIC_GAUSSIAN_RNG_NAMESPACE, DemographicGaussian, NoNoise, Noise, NoiseAlgorithm,
        NoiseDomain, NoisePluginError, NoiseStepError, PROPORTIONAL_GAUSSIAN_RNG_NAMESPACE,
        ProportionalGaussian,
    };
    pub use crate::simulation::{
        DefaultSimulationBuildError, MeanFieldReplicator, MeanFieldReplicatorConfig,
        SimulationBuildError, SimulationKind, SpatialGeneralLotkaVolterra,
        SpatialGeneralLotkaVolterraConfig, SpatialReplicator, SpatialReplicatorConfig,
        StateAssemblyError,
    };
    pub use crate::workflow::{
        GlvConstants, GlvExecutionError, GlvModelConfig, GlvObservationConfig, GlvUnit,
        ObservationConfig, SpeciesValues,
    };
    pub use crate::{
        ABUNDANCE_FIELD, AbundanceRepresentation, AggregateAbundance, CHECKPOINT_STREAM,
        ComputePool, ComputePoolError, SIGNAL_STREAM, SPACE_FIELD, SPACE_STREAM, SpatialAbundance,
        TOTAL_FIELD, TotalAbundance, with_threads,
    };
    pub use ecological_model_core::state_schema::{
        ECOLOGICAL_STATE_SCHEMA_ID, ecological_state_schema,
    };
}
