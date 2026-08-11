//! Deterministic numerical-kernel plugins.
//!
//! A [`Kernel`] combines shared interaction behavior with one numerical
//! algorithm. Algorithms receive temporary typed borrows of the canonical
//! abundance payloads; they cannot replace the authoritative Workflow state or
//! advance its simulation time.

pub mod core;

pub use core::{Kernel, KernelAlgorithm, KernelCore, KernelCoreError, KernelStepError};
