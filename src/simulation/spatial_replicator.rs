//! Concrete spatial replicator simulation.

use physics_in_parallel::prelude::basic::Tensor;
use scientific_workflow::prelude::basics::{RngRecord, SimulationTime, SystemState};

use crate::engine::{Engine, EngineStepError};
use crate::interaction::InteractionMatrix;
use crate::invariant::{InvariantPolicyError, LocalFrequencyInvariant};
use crate::kernel::{
    Diffusion, Kernel, KernelAlgorithm, KernelCore, KernelStepError, SpatialReplicatorRk2,
};
use crate::noise::{NoNoise, Noise, NoiseAlgorithm};
use crate::{AbundanceRepresentation, TimeStep};

use super::{
    DefaultSimulationBuildError, SimulationBuildError, SimulationKind, aggregate_spatial,
    assemble_initial_state, composition_error,
};

/// Immutable inputs that distinguish one spatial replicator simulation.
#[derive(Clone, Debug)]
pub struct SpatialReplicatorConfig {
    growth: Tensor<f64>,
    diffusion: Diffusion,
    cutoff: f64,
    time_step: TimeStep,
}

impl SpatialReplicatorConfig {
    /// Collects typed numerical configuration around PiP-owned lattice geometry.
    pub const fn new(
        growth: Tensor<f64>,
        diffusion: Diffusion,
        cutoff: f64,
        time_step: TimeStep,
    ) -> Self {
        Self {
            growth,
            diffusion,
            cutoff,
            time_step,
        }
    }

    /// Borrows intrinsic per-species growth rates.
    pub const fn growth(&self) -> &Tensor<f64> {
        &self.growth
    }

    /// Borrows finite-difference diffusion configuration.
    pub const fn diffusion(&self) -> &Diffusion {
        &self.diffusion
    }

    /// Returns the hard local-frequency cutoff.
    pub const fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Returns the physical-time increment.
    pub const fn time_step(&self) -> TimeStep {
        self.time_step
    }
}

/// Spatial local-frequency replicator simulation with static plugins.
#[derive(Debug)]
pub struct SpatialReplicator<A = SpatialReplicatorRk2, N = NoNoise> {
    engine: Engine<A, N, LocalFrequencyInvariant>,
}

impl SpatialReplicator {
    /// Builds a deterministic simulation at iteration zero from species-last cells.
    pub fn new(
        initial_space: Tensor<f64>,
        interaction: InteractionMatrix,
        config: SpatialReplicatorConfig,
    ) -> Result<Self, DefaultSimulationBuildError> {
        let SpatialReplicatorConfig {
            growth,
            diffusion,
            cutoff,
            time_step,
        } = config;
        let algorithm = SpatialReplicatorRk2::new(growth, diffusion)
            .map_err(DefaultSimulationBuildError::Kernel)?;
        if initial_space.shape() != algorithm.shape() {
            return Err(DefaultSimulationBuildError::InitialSpaceShapeMismatch {
                expected: algorithm.shape().to_vec(),
                actual: initial_space.shape().to_vec(),
            });
        }
        algorithm
            .validate_time_step(time_step)
            .map_err(DefaultSimulationBuildError::Kernel)?;
        let species = algorithm.species();
        let abundance = aggregate_spatial(&initial_space, species, true)?;
        let state = assemble_initial_state(abundance, Some(initial_space), 1.0)?;
        let invariant = LocalFrequencyInvariant::new(species, cutoff)
            .map_err(DefaultSimulationBuildError::Invariant)?;
        Self::from_plugins(
            state,
            Kernel::new(KernelCore::new(interaction), algorithm),
            Noise::new(NoNoise),
            invariant,
            time_step,
        )
        .map_err(DefaultSimulationBuildError::Composition)
    }

    /// Reconstructs the default deterministic composition around an existing state.
    pub fn from_state(
        state: SystemState,
        interaction: InteractionMatrix,
        config: SpatialReplicatorConfig,
    ) -> Result<Self, DefaultSimulationBuildError> {
        let SpatialReplicatorConfig {
            growth,
            diffusion,
            cutoff,
            time_step,
        } = config;
        let algorithm = SpatialReplicatorRk2::new(growth, diffusion)
            .map_err(DefaultSimulationBuildError::Kernel)?;
        algorithm
            .validate_time_step(time_step)
            .map_err(DefaultSimulationBuildError::Kernel)?;
        let invariant = LocalFrequencyInvariant::new(algorithm.species(), cutoff)
            .map_err(DefaultSimulationBuildError::Invariant)?;
        Self::from_plugins(
            state,
            Kernel::new(KernelCore::new(interaction), algorithm),
            Noise::new(NoNoise),
            invariant,
            time_step,
        )
        .map_err(DefaultSimulationBuildError::Composition)
    }
}

impl<A, N> SpatialReplicator<A, N>
where
    A: KernelAlgorithm,
    N: NoiseAlgorithm,
{
    /// Validates and owns a custom spatial-replicator kernel/noise composition.
    pub fn from_plugins(
        state: SystemState,
        kernel: Kernel<A>,
        noise: Noise<N>,
        invariant: LocalFrequencyInvariant,
        time_step: TimeStep,
    ) -> Result<Self, SimulationBuildError<A::Error, N::Error>> {
        let engine =
            Engine::new(state, kernel, noise, invariant, time_step).map_err(composition_error)?;
        Ok(Self { engine })
    }

    /// Returns the stable concrete-model identity.
    pub const fn kind(&self) -> SimulationKind {
        SimulationKind::SpatialReplicator
    }

    /// Returns the representation required by this concrete model.
    pub const fn abundance_representation(&self) -> AbundanceRepresentation {
        AbundanceRepresentation::RelativeFrequency
    }

    /// Borrows the sole authoritative Workflow state.
    pub const fn state(&self) -> &SystemState {
        self.engine.state()
    }

    /// Returns the fixed physical-time increment.
    pub const fn time_step(&self) -> TimeStep {
        self.engine.time_step()
    }

    /// Returns immutable RNG provenance declared by the selected noise plugin.
    pub fn rng_record(&self) -> Option<&RngRecord> {
        self.engine.rng_record()
    }

    /// Computes the maximum component-wise scaled deterministic RHS residual.
    pub fn maximum_scaled_residual(
        &mut self,
        absolute_tolerance: f64,
        relative_tolerance: f64,
    ) -> Result<Option<f64>, KernelStepError<A::Error>> {
        self.engine
            .maximum_scaled_residual(absolute_tolerance, relative_tolerance)
    }

    /// Performs one complete shared-engine step.
    pub fn step(
        &mut self,
    ) -> Result<SimulationTime, EngineStepError<A::Error, N::Error, InvariantPolicyError>> {
        self.engine.step()
    }

    /// Deliberately transfers ownership of the authoritative Workflow state.
    pub fn into_state(self) -> SystemState {
        self.engine.into_state()
    }
}
