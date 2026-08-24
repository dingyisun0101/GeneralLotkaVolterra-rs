//! Concrete simulation APIs for direct orchestration.
//!
//! These wrappers fix each model's abundance representation and invariant
//! policy while allowing compatible deterministic and stochastic plugins to
//! remain statically dispatched.

mod mean_field_replicator;
mod spatial_general_lotka_volterra;
mod spatial_replicator;

use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use physics_in_parallel::prelude::basic::Tensor;
use scientific_workflow::prelude::basics::{
    PayloadInsertError, SimulationTime, StateError, SystemState,
};
use serde::{Deserialize, Serialize};
use thiserror::Error as ThisError;

use crate::engine::EngineBuildError;
use crate::invariant::InvariantPolicyError;
use crate::kernel::KernelAlgorithmError;
use crate::{
    ABUNDANCE_FIELD, AbundanceRepresentation, AggregateAbundance, SPACE_FIELD, SpatialAbundance,
    TOTAL_FIELD, TotalAbundance, load_state_schema,
};

pub use mean_field_replicator::MeanFieldReplicator;
pub use mean_field_replicator::MeanFieldReplicatorConfig;
pub use spatial_general_lotka_volterra::{
    SpatialGeneralLotkaVolterra, SpatialGeneralLotkaVolterraConfig,
};
pub use spatial_replicator::{SpatialReplicator, SpatialReplicatorConfig};

/// Stable identity of a concrete GLV simulation family.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SimulationKind {
    /// Non-spatial mean-field replicator dynamics.
    MeanFieldReplicator,
    /// Species-last local-frequency replicator reaction-diffusion dynamics.
    SpatialReplicator,
    /// Species-last absolute-population GLV reaction-diffusion dynamics.
    SpatialGeneralLotkaVolterra,
}

impl SimulationKind {
    /// Returns the stable recording-metadata value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MeanFieldReplicator => "mean_field_replicator",
            Self::SpatialReplicator => "spatial_replicator",
            Self::SpatialGeneralLotkaVolterra => "spatial_general_lotka_volterra",
        }
    }

    /// Returns the abundance representation structurally required by this model.
    pub const fn abundance_representation(self) -> AbundanceRepresentation {
        match self {
            Self::MeanFieldReplicator | Self::SpatialReplicator => {
                AbundanceRepresentation::RelativeFrequency
            }
            Self::SpatialGeneralLotkaVolterra => AbundanceRepresentation::AbsoluteCount,
        }
    }
}

/// Rejection while assembling the canonical Workflow state from initial values.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum StateAssemblyError {
    /// The canonical zero iteration/physical-time coordinate was rejected.
    #[error("canonical initial simulation time is invalid")]
    InvalidInitialTime,
    /// The checked-in canonical schema could not be loaded.
    #[error("canonical state schema could not be loaded: {0}")]
    Schema(#[source] StateError),
    /// Workflow rejected the aggregate allocation and retained it in this error.
    #[error("canonical abundance insertion failed: {0}")]
    Abundance(#[source] Box<PayloadInsertError<AggregateAbundance>>),
    /// Workflow rejected the optional spatial allocation and retained it here.
    #[error("canonical space insertion failed: {0}")]
    Space(#[source] Box<PayloadInsertError<SpatialAbundance>>),
    /// Workflow rejected the total value and retained it in this error.
    #[error("canonical total insertion failed: {0}")]
    Total(#[source] Box<PayloadInsertError<TotalAbundance>>),
}

/// Construction failure shared by concrete simulations with custom plugins.
#[derive(Debug)]
#[non_exhaustive]
pub enum SimulationBuildError<KE, NE> {
    /// The shared engine rejected the state or plugin composition.
    Composition(EngineBuildError<KE, NE, InvariantPolicyError>),
}

impl<KE, NE> fmt::Display for SimulationBuildError<KE, NE>
where
    KE: fmt::Display,
    NE: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Composition(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl<KE, NE> Error for SimulationBuildError<KE, NE>
where
    KE: Error + 'static,
    NE: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Composition(error) => Some(error),
        }
    }
}

/// Construction failure for the built-in deterministic/no-noise composition.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum DefaultSimulationBuildError {
    /// Initial values could not be moved into the canonical state.
    #[error(transparent)]
    State(#[from] StateAssemblyError),
    /// Built-in deterministic configuration was invalid.
    #[error("deterministic kernel configuration failed: {0}")]
    Kernel(#[source] KernelAlgorithmError),
    /// Built-in invariant configuration was invalid.
    #[error("invariant configuration failed: {0}")]
    Invariant(#[source] InvariantPolicyError),
    /// An initial spatial allocation is not standard contiguous row-major data.
    #[error("initial space shape {shape:?} is not standard contiguous row-major storage")]
    NonStandardInitialSpace {
        /// Rejected species-last shape.
        shape: Vec<usize>,
    },
    /// Initial spatial storage disagrees with its typed model configuration.
    #[error("initial space shape {actual:?} does not match configured shape {expected:?}")]
    InitialSpaceShapeMismatch {
        /// Shape fixed by typed simulation configuration.
        expected: Vec<usize>,
        /// Shape carried by the supplied initial allocation.
        actual: Vec<usize>,
    },
    /// Reconstruction metadata or the completed default composition was invalid.
    #[error(transparent)]
    Composition(#[from] SimulationBuildError<KernelAlgorithmError, Infallible>),
}

pub(crate) fn assemble_initial_state(
    abundance: AggregateAbundance,
    space: SpatialAbundance,
    total: f64,
) -> Result<SystemState, StateAssemblyError> {
    let time = SimulationTime::from_iteration_and_physical_time(0, 0.0)
        .ok_or(StateAssemblyError::InvalidInitialTime)?;
    let mut state = load_state_schema()
        .map_err(StateAssemblyError::Schema)?
        .create_empty_state(time);
    drop(
        state
            .insert_payload(ABUNDANCE_FIELD, abundance)
            .map_err(|error| StateAssemblyError::Abundance(Box::new(error)))?,
    );
    drop(
        state
            .insert_payload(SPACE_FIELD, space)
            .map_err(|error| StateAssemblyError::Space(Box::new(error)))?,
    );
    let _ = state
        .insert_payload(TOTAL_FIELD, total)
        .map_err(|error| StateAssemblyError::Total(Box::new(error)))?;
    Ok(state)
}

pub(crate) fn aggregate_spatial(
    space: &Tensor<f64>,
    species: usize,
    average: bool,
) -> Result<AggregateAbundance, DefaultSimulationBuildError> {
    let values = space.as_slice();
    let cells = values.len() / species;
    let mut abundance = Tensor::zeros(&[species]);
    for cell in values.chunks_exact(species) {
        for (target, value) in abundance.as_mut_slice().iter_mut().zip(cell) {
            *target += *value;
        }
    }
    if average {
        abundance.map_in_place(|value| value / cells as f64);
    }
    Ok(abundance)
}

pub(crate) fn composition_error<KE, NE>(
    error: EngineBuildError<KE, NE, InvariantPolicyError>,
) -> SimulationBuildError<KE, NE> {
    SimulationBuildError::Composition(error)
}
