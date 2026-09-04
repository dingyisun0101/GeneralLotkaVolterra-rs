#![doc = include_str!("../README.md")]

use physics_in_parallel::prelude::basic::Tensor;
use serde::{Deserialize, Serialize};

pub mod advanced;
pub mod core;
mod engine;
pub mod initialization;
pub mod interaction;
pub mod invariant;
pub mod kernel;
pub mod noise;
pub mod prelude;
pub mod simulation;
mod tensor_compat;
pub mod workflow;

pub use ecological_state_toolkit::state_schema::{
    ECOLOGICAL_STATE_SCHEMA_ID, ecological_state_schema,
};

pub use workflow::{
    GlvConstants, GlvExecutionError, GlvModelConfig, GlvObservationConfig, GlvUnit,
    ObservationConfig, SpeciesValues,
};

/// Process-wide PiP worker participation controls.
pub use physics_in_parallel::prelude::basic::{ParallelismError, max_threads, set_max_threads};

pub use simulation::{
    MeanFieldReplicator, MeanFieldReplicatorConfig, SpatialGeneralLotkaVolterra,
    SpatialGeneralLotkaVolterraConfig, SpatialReplicator, SpatialReplicatorConfig,
};

pub use core::{TimeStep, TimeStepError};

/// Canonical field containing aggregate species abundance.
pub const ABUNDANCE_FIELD: &str = "abundance";

/// Canonical field containing optional spatial species abundance.
pub const SPACE_FIELD: &str = "space";

/// Canonical field containing total abundance.
pub const TOTAL_FIELD: &str = "total";

/// Canonical stream containing aggregate and total abundance.
pub const SIGNAL_STREAM: &str = "signal";

/// Canonical stream containing aggregate, spatial, and total abundance.
pub const SPACE_STREAM: &str = "space";

/// Canonical full-state stream used for deterministic continuation.
pub const CHECKPOINT_STREAM: &str = "checkpoint";

/// Aggregate species-abundance payload shared by every implemented model.
pub type AggregateAbundance = Tensor<f64>;

/// Spatial payload shared by every model.
///
/// A populated workflow slot contains `None` for a non-spatial model and
/// `Some(Tensor<f64>)` for a spatial model.
pub type SpatialAbundance = Option<Tensor<f64>>;

/// Total-abundance payload shared by every implemented model.
pub type TotalAbundance = f64;

/// Scientific interpretation of abundance values.
///
/// Representation is immutable model configuration and recording metadata. It
/// is deliberately not repeated as a payload in every evolving state.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AbundanceRepresentation {
    /// Abundances are normalized relative frequencies.
    RelativeFrequency,
    /// Abundances are absolute population counts.
    AbsoluteCount,
}

impl AbundanceRepresentation {
    /// Returns the stable recording-metadata value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RelativeFrequency => "relative_frequency",
            Self::AbsoluteCount => "absolute_count",
        }
    }
}
