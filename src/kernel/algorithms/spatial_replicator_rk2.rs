//! Allocation-free spatial replicator midpoint RK2 evolution.

use ndarray::Array1;

use crate::TimeStep;
use crate::kernel::core::{KernelAlgorithm, KernelCore, KernelStateView, KernelUpdate};

use super::KernelAlgorithmError;
use super::spatial::{Diffusion, SpatialDynamics, SpatialLayout, SpatialRk2};

/// Midpoint RK2 integration of local replicator reaction-diffusion dynamics.
#[derive(Debug)]
pub struct SpatialReplicatorRk2 {
    inner: SpatialRk2,
}

impl SpatialReplicatorRk2 {
    /// Creates fixed configuration and scratch for one species-last shape.
    pub fn new(
        shape: Vec<usize>,
        growth: Array1<f64>,
        diffusion: Diffusion,
    ) -> Result<Self, KernelAlgorithmError> {
        Ok(Self {
            inner: SpatialRk2::new(shape, growth, diffusion)?,
        })
    }

    /// Borrows the validated spatial layout.
    pub const fn layout(&self) -> &SpatialLayout {
        self.inner.layout()
    }

    /// Borrows the immutable growth vector.
    pub const fn growth(&self) -> &Array1<f64> {
        self.inner.growth()
    }

    /// Borrows the immutable diffusion configuration.
    pub const fn diffusion(&self) -> &Diffusion {
        self.inner.diffusion()
    }

    /// Returns every fixed spatial scratch length for allocation-reuse checks.
    pub fn scratch_lengths(&self) -> [usize; 3] {
        self.inner.scratch_lengths()
    }

    /// Checks the conservative explicit-diffusion stability bound.
    pub fn validate_time_step(&self, time_step: TimeStep) -> Result<(), KernelAlgorithmError> {
        self.inner.validate_time_step(time_step)
    }
}

impl KernelAlgorithm for SpatialReplicatorRk2 {
    type Error = KernelAlgorithmError;

    fn validate(&self, core: &KernelCore, state: KernelStateView<'_>) -> Result<(), Self::Error> {
        self.inner.validate(core, state)
    }

    fn compute<'algorithm>(
        &'algorithm mut self,
        core: &KernelCore,
        state: KernelStateView<'_>,
        time_step: TimeStep,
    ) -> Result<KernelUpdate<'algorithm>, Self::Error> {
        self.inner
            .compute(core, state, time_step, SpatialDynamics::Replicator)
            .map(KernelUpdate::space)
    }
}
