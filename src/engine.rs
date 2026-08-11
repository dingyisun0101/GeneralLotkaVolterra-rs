//! Shared composition and step ordering for every concrete simulation.
//!
//! `Engine` is an implementation-level API. Concrete simulation types wrap it
//! and expose model-specific construction while this module keeps state
//! ownership, plugin ordering, and time advancement identical across models.

use std::error::Error;
use std::fmt;

use scientific_workflow::rng_record::RngRecord;
use scientific_workflow::system_state::{SimulationTime, StateError, SystemState};

use crate::TimeStep;
use crate::invariant::{self, InvariantError, InvariantPolicy};
use crate::kernel::{Kernel, KernelAlgorithm, KernelStepError};
use crate::noise::{Noise, NoiseAlgorithm, NoiseStepError};

/// One authoritative Workflow state composed with one plugin of each kind.
///
/// Fields remain private so only the shared step implementation can mutate the
/// state. Concrete simulations receive immutable state access and may
/// deliberately transfer state ownership with [`Self::into_state`].
#[derive(Debug)]
pub struct Engine<A, N, I> {
    state: SystemState,
    kernel: Kernel<A>,
    noise: Noise<N>,
    invariant: I,
    time_step: TimeStep,
}

/// Construction result for one statically composed engine.
pub type EngineBuildResult<A, N, I> = Result<
    Engine<A, N, I>,
    EngineBuildError<
        <A as KernelAlgorithm>::Error,
        <N as NoiseAlgorithm>::Error,
        <I as InvariantPolicy>::Error,
    >,
>;

/// Step result for one statically composed engine.
pub type EngineStepResult<A, N, I> = Result<
    SimulationTime,
    EngineStepError<
        <A as KernelAlgorithm>::Error,
        <N as NoiseAlgorithm>::Error,
        <I as InvariantPolicy>::Error,
    >,
>;

impl<A, N, I> Engine<A, N, I>
where
    A: KernelAlgorithm,
    N: NoiseAlgorithm,
    I: InvariantPolicy,
{
    /// Validates and takes sole ownership of a complete simulation composition.
    pub fn new(
        state: SystemState,
        kernel: Kernel<A>,
        noise: Noise<N>,
        invariant: I,
        time_step: TimeStep,
    ) -> EngineBuildResult<A, N, I> {
        state
            .simulation_time()
            .checked_advance(Some(time_step.get()))
            .map_err(EngineBuildError::Time)?;
        kernel
            .validate_state(&state)
            .map_err(EngineBuildError::Kernel)?;
        noise
            .validate_state(&state)
            .map_err(EngineBuildError::Noise)?;
        invariant::validate_state(&invariant, &state).map_err(EngineBuildError::Invariant)?;
        Ok(Self {
            state,
            kernel,
            noise,
            invariant,
            time_step,
        })
    }

    /// Borrows the sole authoritative Workflow state immutably.
    pub const fn state(&self) -> &SystemState {
        &self.state
    }

    /// Returns the validated physical-time increment.
    pub const fn time_step(&self) -> TimeStep {
        self.time_step
    }

    /// Returns immutable RNG provenance declared by the noise plugin.
    pub fn rng_record(&self) -> Option<&RngRecord> {
        self.noise.rng_record()
    }

    /// Deliberately transfers ownership of the authoritative Workflow state.
    pub fn into_state(self) -> SystemState {
        self.state
    }

    /// Performs one complete scientific step in the shared model order.
    ///
    /// The exact sequence is deterministic kernel, invariant, noise,
    /// invariant, and finally one Workflow time advancement. A failed phase
    /// stops the sequence and leaves time unchanged. Individual plugins are
    /// responsible for their documented phase-level commit guarantees; the
    /// engine does not clone the full state for global rollback.
    pub fn step(&mut self) -> EngineStepResult<A, N, I> {
        self.state
            .simulation_time()
            .checked_advance(Some(self.time_step.get()))
            .map_err(EngineStepError::Time)?;
        self.kernel
            .step(&mut self.state, self.time_step)
            .map_err(EngineStepError::Kernel)?;
        invariant::enforce_state(&mut self.invariant, &mut self.state)
            .map_err(EngineStepError::InvariantAfterKernel)?;
        self.noise
            .apply(&mut self.state, self.time_step)
            .map_err(EngineStepError::Noise)?;
        invariant::enforce_state(&mut self.invariant, &mut self.state)
            .map_err(EngineStepError::InvariantAfterNoise)?;
        self.state
            .advance_simulation_time(Some(self.time_step.get()))
            .map_err(EngineStepError::Time)
    }
}

/// Failure while validating an engine composition before evolution.
#[derive(Debug)]
#[non_exhaustive]
pub enum EngineBuildError<KE, NE, IE> {
    /// The initial time coordinate cannot advance by the configured step.
    Time(StateError),
    /// The deterministic kernel rejected the initial state.
    Kernel(KernelStepError<KE>),
    /// The stochastic plugin rejected the initial state.
    Noise(NoiseStepError<NE>),
    /// The invariant policy rejected the initial state.
    Invariant(InvariantError<IE>),
}

impl<KE, NE, IE> fmt::Display for EngineBuildError<KE, NE, IE>
where
    KE: fmt::Display,
    NE: fmt::Display,
    IE: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Time(error) => write!(formatter, "engine time validation failed: {error}"),
            Self::Kernel(error) => write!(formatter, "engine kernel validation failed: {error}"),
            Self::Noise(error) => write!(formatter, "engine noise validation failed: {error}"),
            Self::Invariant(error) => {
                write!(formatter, "engine invariant validation failed: {error}")
            }
        }
    }
}

impl<KE, NE, IE> Error for EngineBuildError<KE, NE, IE>
where
    KE: Error + 'static,
    NE: Error + 'static,
    IE: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Time(error) => Some(error),
            Self::Kernel(error) => Some(error),
            Self::Noise(error) => Some(error),
            Self::Invariant(error) => Some(error),
        }
    }
}

/// Failure during one shared engine step.
#[derive(Debug)]
#[non_exhaustive]
pub enum EngineStepError<KE, NE, IE> {
    /// The current time cannot be advanced by the configured step.
    Time(StateError),
    /// The deterministic phase failed.
    Kernel(KernelStepError<KE>),
    /// Invariant restoration after the deterministic phase failed.
    InvariantAfterKernel(InvariantError<IE>),
    /// The stochastic phase failed.
    Noise(NoiseStepError<NE>),
    /// Invariant restoration after the stochastic phase failed.
    InvariantAfterNoise(InvariantError<IE>),
}

impl<KE, NE, IE> fmt::Display for EngineStepError<KE, NE, IE>
where
    KE: fmt::Display,
    NE: fmt::Display,
    IE: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Time(error) => write!(formatter, "engine time advancement failed: {error}"),
            Self::Kernel(error) => write!(formatter, "engine kernel phase failed: {error}"),
            Self::InvariantAfterKernel(error) => {
                write!(formatter, "engine post-kernel invariant failed: {error}")
            }
            Self::Noise(error) => write!(formatter, "engine noise phase failed: {error}"),
            Self::InvariantAfterNoise(error) => {
                write!(formatter, "engine post-noise invariant failed: {error}")
            }
        }
    }
}

impl<KE, NE, IE> Error for EngineStepError<KE, NE, IE>
where
    KE: Error + 'static,
    NE: Error + 'static,
    IE: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Time(error) => Some(error),
            Self::Kernel(error) => Some(error),
            Self::InvariantAfterKernel(error) => Some(error),
            Self::Noise(error) => Some(error),
            Self::InvariantAfterNoise(error) => Some(error),
        }
    }
}
