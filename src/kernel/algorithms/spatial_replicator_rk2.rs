//! Allocation-free spatial replicator midpoint RK2 evolution.

use physics_in_parallel::prelude::basic::Tensor;

use crate::TimeStep;
use crate::kernel::core::{
    KernelAlgorithm, KernelCore, KernelResidual, KernelStateView, KernelUpdate,
};

use super::KernelAlgorithmError;
use super::spatial::{Diffusion, SpatialDynamics, SpatialRk2};

/// Midpoint RK2 integration of local replicator reaction-diffusion dynamics.
#[derive(Debug)]
pub struct SpatialReplicatorRk2 {
    inner: SpatialRk2,
}

impl SpatialReplicatorRk2 {
    /// Creates fixed configuration and scratch from PiP lattice geometry.
    pub fn new(growth: Tensor<f64>, diffusion: Diffusion) -> Result<Self, KernelAlgorithmError> {
        Ok(Self {
            inner: SpatialRk2::new(growth, diffusion)?,
        })
    }

    /// Borrows the expected species-last state shape.
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    /// Returns the fixed species dimension.
    pub const fn species(&self) -> usize {
        self.inner.species()
    }

    /// Borrows the immutable growth vector.
    pub const fn growth(&self) -> &Tensor<f64> {
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

    fn residual<'algorithm>(
        &'algorithm mut self,
        core: &KernelCore,
        state: KernelStateView<'_>,
    ) -> Result<Option<KernelResidual<'algorithm>>, Self::Error> {
        self.inner
            .residual(core, state, SpatialDynamics::Replicator)
            .map(|values| Some(KernelResidual::Space(values)))
    }
}
