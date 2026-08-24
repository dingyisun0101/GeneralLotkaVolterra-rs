//! Built-in deterministic numerical algorithms.

mod mean_field_replicator_rk4;
mod spatial;
mod spatial_general_lotka_volterra_rk2;
mod spatial_replicator_rk2;

use thiserror::Error;

use super::core::KernelCoreError;

pub use mean_field_replicator_rk4::MeanFieldReplicatorRk4;
pub use physics_in_parallel::prelude::basic::BoundaryCondition;
pub use spatial::Diffusion;
pub use spatial_general_lotka_volterra_rk2::SpatialGeneralLotkaVolterraRk2;
pub use spatial_replicator_rk2::SpatialReplicatorRk2;

/// Configuration, state-domain, or numerical failure in a built-in kernel.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum KernelAlgorithmError {
    /// Numerical algorithms require at least one species.
    #[error("kernel algorithm requires at least one species")]
    EmptySpecies,
    /// A coefficient tensor must be a vector.
    #[error("{field} tensor has rank {actual}, expected rank 1")]
    CoefficientRank {
        /// Configuration field name.
        field: &'static str,
        /// Rejected tensor rank.
        actual: usize,
    },
    /// Model and kernel facilities disagree about species count.
    #[error("kernel core has {actual} species, expected {expected}")]
    CoreSpeciesMismatch {
        /// Algorithm species dimension.
        expected: usize,
        /// Matrix species dimension.
        actual: usize,
    },
    /// Aggregate abundance has the wrong species dimension.
    #[error("abundance length {actual} does not match species count {expected}")]
    SpeciesMismatch {
        /// Algorithm species dimension.
        expected: usize,
        /// State dimension.
        actual: usize,
    },
    /// Growth coefficients must be finite.
    #[error("growth coefficient {index} is not finite: {value}")]
    NonFiniteGrowth {
        /// Species index.
        index: usize,
        /// Rejected coefficient.
        value: f64,
    },
    /// A mean-field algorithm received a spatial state.
    #[error("mean-field kernel requires `space = None`")]
    UnexpectedSpace,
    /// A spatial algorithm received no spatial payload.
    #[error("spatial kernel requires populated `space`")]
    SpaceRequired,
    /// Checked spatial element or stride arithmetic overflowed.
    #[error("spatial shape {shape:?} overflows its element count or strides")]
    SpatialShapeOverflow {
        /// Rejected shape.
        shape: Vec<usize>,
    },
    /// Spatial state shape differs from the fixed algorithm layout.
    #[error("space shape {actual:?} does not match kernel shape {expected:?}")]
    SpaceShapeMismatch {
        /// Configured shape.
        expected: Vec<usize>,
        /// State shape.
        actual: Vec<usize>,
    },
    /// Hot-loop flat indexing requires standard row-major storage.
    #[error("kernel state must use standard contiguous row-major storage")]
    NonStandardLayout,
    /// State values must be finite at the deterministic boundary.
    #[error("{field} value at linear index {linear_index} is not finite: {value}")]
    NonFiniteState {
        /// Canonical field name.
        field: &'static str,
        /// Row-major linear index.
        linear_index: usize,
        /// Rejected value.
        value: f64,
    },
    /// State values must be nonnegative at the deterministic boundary.
    #[error("{field} value at linear index {linear_index} is negative: {value}")]
    NegativeState {
        /// Canonical field name.
        field: &'static str,
        /// Row-major linear index.
        linear_index: usize,
        /// Rejected value.
        value: f64,
    },
    /// Diffusion coefficients have the wrong species dimension.
    #[error("diffusion length {actual} does not match species count {expected}")]
    DiffusionLength {
        /// Species dimension.
        expected: usize,
        /// Coefficient count.
        actual: usize,
    },
    /// A diffusion coefficient was negative or non-finite.
    #[error("diffusion coefficient {index} must be finite and nonnegative, found {value}")]
    InvalidDiffusion {
        /// Species index.
        index: usize,
        /// Rejected coefficient.
        value: f64,
    },
    /// PiP rejected lattice geometry or finite-difference layout.
    #[error("invalid lattice configuration: {0}")]
    SpaceConfig(#[from] physics_in_parallel::prelude::basic::SquareLatticeConfigError),
    /// Explicit diffusion would exceed the conservative stability limit.
    #[error("time step {actual} exceeds explicit diffusion stability limit {maximum}")]
    UnstableTimeStep {
        /// Configured engine increment.
        actual: f64,
        /// Conservative maximum.
        maximum: f64,
    },
    /// Shared interaction application rejected an internal vector.
    #[error(transparent)]
    Interaction(#[from] KernelCoreError),
}

pub(crate) fn validate_values(
    field: &'static str,
    values: impl Iterator<Item = f64>,
) -> Result<(), KernelAlgorithmError> {
    for (linear_index, value) in values.enumerate() {
        if !value.is_finite() {
            return Err(KernelAlgorithmError::NonFiniteState {
                field,
                linear_index,
                value,
            });
        }
        if value < 0.0 {
            return Err(KernelAlgorithmError::NegativeState {
                field,
                linear_index,
                value,
            });
        }
    }
    Ok(())
}
