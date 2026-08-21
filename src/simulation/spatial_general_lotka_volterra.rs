//! Concrete spatial General Lotka–Volterra population simulation.

use ndarray::{Array1, ArrayD};
use scientific_workflow::prelude::basics::{RngRecord, SimulationTime, SystemState};

use crate::engine::{Engine, EngineStepError};
use crate::interaction::InteractionMatrix;
use crate::invariant::{InvariantPolicyError, PopulationInvariant};
use crate::kernel::{
    Diffusion, Kernel, KernelAlgorithm, KernelCore, KernelStepError, SpatialGeneralLotkaVolterraRk2,
};
use crate::noise::{NoNoise, Noise, NoiseAlgorithm};
use crate::{AbundanceRepresentation, TimeStep};

use super::{
    DefaultSimulationBuildError, SimulationBuildError, SimulationKind, aggregate_spatial,
    assemble_initial_state, composition_error,
};

/// Immutable inputs that distinguish one spatial General Lotka–Volterra simulation.
#[derive(Clone, Debug)]
pub struct SpatialGeneralLotkaVolterraConfig {
    growth: Array1<f64>,
    diffusion: Diffusion,
    cutoff: f64,
    carrying_capacity: Option<f64>,
    time_step: TimeStep,
}

impl SpatialGeneralLotkaVolterraConfig {
    /// Collects typed population configuration around PiP-owned lattice geometry.
    pub const fn new(
        growth: Array1<f64>,
        diffusion: Diffusion,
        cutoff: f64,
        carrying_capacity: Option<f64>,
        time_step: TimeStep,
    ) -> Self {
        Self {
            growth,
            diffusion,
            cutoff,
            carrying_capacity,
            time_step,
        }
    }

    /// Borrows intrinsic per-species growth rates.
    pub const fn growth(&self) -> &Array1<f64> {
        &self.growth
    }

    /// Borrows finite-difference diffusion configuration.
    pub const fn diffusion(&self) -> &Diffusion {
        &self.diffusion
    }

    /// Returns the hard population cutoff.
    pub const fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Returns the optional global carrying capacity.
    pub const fn carrying_capacity(&self) -> Option<f64> {
        self.carrying_capacity
    }

    /// Returns the physical-time increment.
    pub const fn time_step(&self) -> TimeStep {
        self.time_step
    }
}

/// Spatial absolute-population General Lotka–Volterra simulation with static plugins.
#[derive(Debug)]
pub struct SpatialGeneralLotkaVolterra<A = SpatialGeneralLotkaVolterraRk2, N = NoNoise> {
    engine: Engine<A, N, PopulationInvariant>,
}

impl SpatialGeneralLotkaVolterra {
    /// Builds a deterministic simulation at iteration zero from species-last populations.
    pub fn new(
        initial_space: ArrayD<f64>,
        interaction: InteractionMatrix,
        config: SpatialGeneralLotkaVolterraConfig,
    ) -> Result<Self, DefaultSimulationBuildError> {
        let SpatialGeneralLotkaVolterraConfig {
            growth,
            diffusion,
            cutoff,
            carrying_capacity,
            time_step,
        } = config;
        let algorithm = SpatialGeneralLotkaVolterraRk2::new(growth, diffusion)
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
        let abundance = aggregate_spatial(&initial_space, species, false)?;
        let total = abundance.sum().round().max(0.0);
        let state = assemble_initial_state(abundance, Some(initial_space), total)?;
        let invariant = PopulationInvariant::new(species, cutoff, carrying_capacity)
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
        config: SpatialGeneralLotkaVolterraConfig,
    ) -> Result<Self, DefaultSimulationBuildError> {
        let SpatialGeneralLotkaVolterraConfig {
            growth,
            diffusion,
            cutoff,
            carrying_capacity,
            time_step,
        } = config;
        let algorithm = SpatialGeneralLotkaVolterraRk2::new(growth, diffusion)
            .map_err(DefaultSimulationBuildError::Kernel)?;
        algorithm
            .validate_time_step(time_step)
            .map_err(DefaultSimulationBuildError::Kernel)?;
        let invariant = PopulationInvariant::new(algorithm.species(), cutoff, carrying_capacity)
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

impl<A, N> SpatialGeneralLotkaVolterra<A, N>
where
    A: KernelAlgorithm,
    N: NoiseAlgorithm,
{
    /// Validates and owns a custom spatial-GLV kernel/noise composition.
    pub fn from_plugins(
        state: SystemState,
        kernel: Kernel<A>,
        noise: Noise<N>,
        invariant: PopulationInvariant,
        time_step: TimeStep,
    ) -> Result<Self, SimulationBuildError<A::Error, N::Error>> {
        let engine =
            Engine::new(state, kernel, noise, invariant, time_step).map_err(composition_error)?;
        Ok(Self { engine })
    }

    /// Returns the stable concrete-model identity.
    pub const fn kind(&self) -> SimulationKind {
        SimulationKind::SpatialGeneralLotkaVolterra
    }

    /// Returns the representation required by this concrete model.
    pub const fn abundance_representation(&self) -> AbundanceRepresentation {
        AbundanceRepresentation::AbsoluteCount
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
