//! Concrete spatial replicator simulation.

use ndarray::{Array1, ArrayD};
use scientific_workflow::system_state::{SimulationTime, SystemState};

use crate::engine::{Engine, EngineStepError};
use crate::invariant::{InvariantPolicyError, LocalFrequencyInvariant};
use crate::kernel::{
    Diffusion, InteractionMatrix, Kernel, KernelAlgorithm, KernelCore, SpatialReplicatorRk2,
};
use crate::noise::{NoNoise, Noise, NoiseAlgorithm};
use crate::{AbundanceRepresentation, TimeStep};

use super::{
    DefaultSimulationBuildError, SimulationBuildError, SimulationKind, aggregate_spatial,
    assemble_initial_state, composition_error, require_representation,
};

/// Immutable inputs that distinguish one spatial replicator simulation.
#[derive(Clone, Debug)]
pub struct SpatialReplicatorConfig {
    shape: Vec<usize>,
    growth: Array1<f64>,
    diffusion: Diffusion,
    cutoff: f64,
    time_step: TimeStep,
}

impl SpatialReplicatorConfig {
    /// Collects a species-last layout and typed numerical configuration.
    pub const fn new(
        shape: Vec<usize>,
        growth: Array1<f64>,
        diffusion: Diffusion,
        cutoff: f64,
        time_step: TimeStep,
    ) -> Self {
        Self {
            shape,
            growth,
            diffusion,
            cutoff,
            time_step,
        }
    }

    /// Borrows the exact species-last shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrows intrinsic per-species growth rates.
    pub const fn growth(&self) -> &Array1<f64> {
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
        initial_space: ArrayD<f64>,
        interaction: InteractionMatrix,
        config: SpatialReplicatorConfig,
    ) -> Result<Self, DefaultSimulationBuildError> {
        if initial_space.shape() != config.shape() {
            return Err(DefaultSimulationBuildError::InitialSpaceShapeMismatch {
                expected: config.shape().to_vec(),
                actual: initial_space.shape().to_vec(),
            });
        }
        let SpatialReplicatorConfig {
            shape,
            growth,
            diffusion,
            cutoff,
            time_step,
        } = config;
        let algorithm = SpatialReplicatorRk2::new(shape, growth, diffusion)
            .map_err(DefaultSimulationBuildError::Kernel)?;
        algorithm
            .validate_time_step(time_step)
            .map_err(DefaultSimulationBuildError::Kernel)?;
        let species = algorithm.layout().species();
        let abundance = aggregate_spatial(&initial_space, species, true)?;
        let state = assemble_initial_state(abundance, Some(initial_space), 1.0)?;
        let invariant = LocalFrequencyInvariant::new(species, cutoff)
            .map_err(DefaultSimulationBuildError::Invariant)?;
        Self::from_plugins(
            state,
            AbundanceRepresentation::RelativeFrequency,
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
        representation: AbundanceRepresentation,
        interaction: InteractionMatrix,
        config: SpatialReplicatorConfig,
    ) -> Result<Self, DefaultSimulationBuildError> {
        let SpatialReplicatorConfig {
            shape,
            growth,
            diffusion,
            cutoff,
            time_step,
        } = config;
        let algorithm = SpatialReplicatorRk2::new(shape, growth, diffusion)
            .map_err(DefaultSimulationBuildError::Kernel)?;
        algorithm
            .validate_time_step(time_step)
            .map_err(DefaultSimulationBuildError::Kernel)?;
        let invariant = LocalFrequencyInvariant::new(algorithm.layout().species(), cutoff)
            .map_err(DefaultSimulationBuildError::Invariant)?;
        Self::from_plugins(
            state,
            representation,
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
        representation: AbundanceRepresentation,
        kernel: Kernel<A>,
        noise: Noise<N>,
        invariant: LocalFrequencyInvariant,
        time_step: TimeStep,
    ) -> Result<Self, SimulationBuildError<A::Error, N::Error>> {
        require_representation(representation, AbundanceRepresentation::RelativeFrequency)?;
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
