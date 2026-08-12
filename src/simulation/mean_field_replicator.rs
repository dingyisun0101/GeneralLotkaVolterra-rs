//! Concrete mean-field replicator simulation.

use ndarray::Array1;
use scientific_workflow::rng_record::RngRecord;
use scientific_workflow::system_state::{SimulationTime, SystemState};

use crate::engine::{Engine, EngineStepError};
use crate::interaction::InteractionMatrix;
use crate::invariant::{FrequencyInvariant, InvariantPolicyError};
use crate::kernel::KernelStepError;
use crate::kernel::{Kernel, KernelAlgorithm, KernelCore, MeanFieldReplicatorRk4};
use crate::noise::{NoNoise, Noise, NoiseAlgorithm};
use crate::{AbundanceRepresentation, TimeStep};

use super::{
    DefaultSimulationBuildError, SimulationBuildError, SimulationKind, assemble_initial_state,
    composition_error, require_representation,
};

/// Immutable inputs that distinguish one mean-field replicator simulation.
#[derive(Clone, Debug)]
pub struct MeanFieldReplicatorConfig {
    growth: Array1<f64>,
    cutoff: f64,
    time_step: TimeStep,
}

impl MeanFieldReplicatorConfig {
    /// Collects typed model inputs; full domain validation occurs at simulation construction.
    pub const fn new(growth: Array1<f64>, cutoff: f64, time_step: TimeStep) -> Self {
        Self {
            growth,
            cutoff,
            time_step,
        }
    }

    /// Borrows intrinsic per-species growth rates.
    pub const fn growth(&self) -> &Array1<f64> {
        &self.growth
    }

    /// Returns the hard frequency cutoff.
    pub const fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Returns the physical-time increment.
    pub const fn time_step(&self) -> TimeStep {
        self.time_step
    }
}

/// Mean-field replicator simulation with statically selected kernel and noise.
#[derive(Debug)]
pub struct MeanFieldReplicator<A = MeanFieldReplicatorRk4, N = NoNoise> {
    engine: Engine<A, N, FrequencyInvariant>,
}

impl MeanFieldReplicator {
    /// Builds a deterministic simulation at iteration zero from normalized frequencies.
    pub fn new(
        initial_abundance: Array1<f64>,
        interaction: InteractionMatrix,
        config: MeanFieldReplicatorConfig,
    ) -> Result<Self, DefaultSimulationBuildError> {
        let state = assemble_initial_state(initial_abundance, None, 1.0)?;
        Self::from_state(
            state,
            AbundanceRepresentation::RelativeFrequency,
            interaction,
            config,
        )
    }

    /// Reconstructs the default deterministic composition around an existing state.
    pub fn from_state(
        state: SystemState,
        representation: AbundanceRepresentation,
        interaction: InteractionMatrix,
        config: MeanFieldReplicatorConfig,
    ) -> Result<Self, DefaultSimulationBuildError> {
        let MeanFieldReplicatorConfig {
            growth,
            cutoff,
            time_step,
        } = config;
        let algorithm =
            MeanFieldReplicatorRk4::new(growth).map_err(DefaultSimulationBuildError::Kernel)?;
        let species = algorithm.growth().len();
        let invariant = FrequencyInvariant::new(species, cutoff)
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

impl<A, N> MeanFieldReplicator<A, N>
where
    A: KernelAlgorithm,
    N: NoiseAlgorithm,
{
    /// Validates and owns a custom mean-field kernel/noise composition.
    pub fn from_plugins(
        state: SystemState,
        representation: AbundanceRepresentation,
        kernel: Kernel<A>,
        noise: Noise<N>,
        invariant: FrequencyInvariant,
        time_step: TimeStep,
    ) -> Result<Self, SimulationBuildError<A::Error, N::Error>> {
        require_representation(representation, AbundanceRepresentation::RelativeFrequency)?;
        let engine =
            Engine::new(state, kernel, noise, invariant, time_step).map_err(composition_error)?;
        Ok(Self { engine })
    }

    /// Returns the stable concrete-model identity.
    pub const fn kind(&self) -> SimulationKind {
        SimulationKind::MeanFieldReplicator
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
