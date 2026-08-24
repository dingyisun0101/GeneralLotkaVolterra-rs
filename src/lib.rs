#![doc = include_str!("../README.md")]

use std::path::Path;

use physics_in_parallel::prelude::basic::Tensor;
use scientific_workflow::prelude::basics::{StateError, SystemStateSchema};
use serde::{Deserialize, Serialize};

pub mod advanced;
pub mod core;
mod engine;
pub mod fixed_point;
pub mod initialization;
pub mod interaction;
pub mod invariant;
pub mod kernel;
pub mod noise;
pub mod prelude;
pub mod reading;
pub mod recording;
pub mod simulation;
pub mod study_inputs;
mod template;
pub mod terminal_state;
pub mod workload;

pub use study_inputs::{GlvConfiguration, GlvInputs, GlvInputsError};
pub use template::{GlvTemplate, INTERACTION_SOURCE_KEY, InteractionSource, TemplateTaskError};
pub use workload::{GlvWorkload, GlvWorkloadError};

/// Explicit reusable worker pool for callers that want to bound an entire GLV
/// computation. Plain GLV operations use the current Rayon pool and create no
/// pool of their own.
pub use physics_in_parallel::prelude::basic::{ComputePool, ComputePoolError, with_threads};

pub use simulation::{
    MeanFieldReplicator, MeanFieldReplicatorConfig, SpatialGeneralLotkaVolterra,
    SpatialGeneralLotkaVolterraConfig, SpatialReplicator, SpatialReplicatorConfig,
};

pub use core::{TimeStep, TimeStepError};
pub use fixed_point::{
    ACCEPTED_FIXED_POINT_FORMAT, AcceptedFixedPoint, AcceptedFixedPointError,
    open_accepted_fixed_point,
};
pub use terminal_state::{
    TERMINAL_STATE_FORMAT, TerminalState, TerminalStateOpenError, open_terminal_state,
};

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

/// Returns the checked-in canonical state-schema path.
pub fn state_schema_path() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/schemas/state.json"))
}

/// Loads and validates the canonical GLV state schema.
pub fn load_state_schema() -> Result<SystemStateSchema, StateError> {
    SystemStateSchema::load_json_template(state_schema_path())
}
