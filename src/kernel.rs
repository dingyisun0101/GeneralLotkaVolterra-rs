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
    Kernel, KernelAlgorithm, KernelCore, KernelCoreError, KernelResidual, KernelResidualError,
    KernelStateView, KernelStepError, KernelUpdate, KernelUpdateError,
};
