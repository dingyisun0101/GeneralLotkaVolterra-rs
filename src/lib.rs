#![doc = include_str!("../README.md")]

use std::path::Path;

use ndarray::{Array1, ArrayD};
use scientific_workflow::system_state::{StateError, SystemStateSchema};
use serde::{Deserialize, Serialize};

pub mod advanced;
pub mod core;
#[doc(hidden)]
pub mod engine;
pub mod initialization;
pub mod interaction;
pub mod invariant;
pub mod kernel;
pub mod noise;
pub mod prelude;
pub mod project;
pub mod reading;
pub mod recording;
pub mod simulation;
mod template;
pub mod termination;

pub use template::{GlvRunError, GlvTemplate, run};

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
pub type AggregateAbundance = Array1<f64>;

/// Spatial payload shared by every model.
///
/// A populated workflow slot contains `None` for a non-spatial model and
/// `Some(ArrayD<f64>)` for a spatial model.
pub type SpatialAbundance = Option<ArrayD<f64>>;

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
