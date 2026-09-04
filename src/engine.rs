//! Shared composition and step ordering for every concrete simulation.
//!
//! `Engine` is an implementation-level API. Concrete simulation types wrap it
//! and expose model-specific construction while this module keeps state
//! ownership, plugin ordering, and time advancement identical across models.

use std::error::Error;
use std::fmt;

use scientific_workflow::prelude::{StateError, StateTime, SystemState};

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
    StateTime,
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
            .time()
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

    /// Returns the authoritative deterministic RHS residual in configured units.
    pub fn maximum_scaled_residual(
        &mut self,
        absolute_tolerance: f64,
        relative_tolerance: f64,
    ) -> Result<Option<f64>, KernelStepError<A::Error>> {
        self.kernel
            .maximum_scaled_residual(&self.state, absolute_tolerance, relative_tolerance)
    }

    /// Deliberately transfers ownership of the authoritative Workflow state.
    pub fn into_state(self) -> SystemState {
        self.state
    }

    /// Performs one complete scientific step in the shared model order.
    ///
    /// The exact sequence is deterministic kernel, invariant, active noise,
    /// invariant, and finally one Workflow time advancement. A no-op noise
    /// plugin and its redundant final invariant pass are skipped. A failed
    /// phase stops the sequence and leaves time unchanged. Individual plugins
    /// are responsible for their documented phase-level commit guarantees;
    /// the engine does not clone the full state for global rollback.
    pub fn step(&mut self) -> EngineStepResult<A, N, I> {
        self.state
            .time()
            .checked_advance(Some(self.time_step.get()))
            .map_err(EngineStepError::Time)?;
        self.kernel
            .step(&mut self.state, self.time_step)
            .map_err(EngineStepError::Kernel)?;
        invariant::enforce_state(&mut self.invariant, &mut self.state)
            .map_err(EngineStepError::InvariantAfterKernel)?;
        if !self.noise.is_noop() {
            self.noise
                .apply(&mut self.state, self.time_step)
                .map_err(EngineStepError::Noise)?;
            invariant::enforce_state(&mut self.invariant, &mut self.state)
                .map_err(EngineStepError::InvariantAfterNoise)?;
        }
        self.state
            .advance_time(Some(self.time_step.get()))
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use physics_in_parallel::prelude::basic::{Backend, Tensor};
    use scientific_workflow::prelude::{StateError, StateTime, SystemState};
    use thiserror::Error;

    use super::{Engine, EngineBuildError, EngineStepError};
    use crate::interaction::InteractionMatrix;
    use crate::invariant::InvariantPolicy;
    use crate::kernel::{Kernel, KernelAlgorithm, KernelCore, KernelStateView, KernelUpdate};
    use crate::noise::{Noise, NoiseAlgorithm};
    use crate::tensor_compat::DenseTensorExt;
    use crate::{
        ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, TimeStep,
        TotalAbundance, ecological_state_schema,
    };

    type CallLog = Arc<Mutex<Vec<&'static str>>>;

    #[derive(Clone, Copy, Debug, Error)]
    #[error("controlled test failure")]
    struct TestError;

    #[derive(Debug)]
    struct TestKernel {
        scratch: Tensor<f64>,
        calls: CallLog,
    }

    impl KernelAlgorithm for TestKernel {
        type Error = TestError;

        fn validate(
            &self,
            core: &KernelCore,
            state: KernelStateView<'_>,
        ) -> Result<(), Self::Error> {
            (state.abundance().size() == core.species())
                .then_some(())
                .ok_or(TestError)
        }

        fn compute<'a>(
            &'a mut self,
            _core: &KernelCore,
            state: KernelStateView<'_>,
            _time_step: TimeStep,
        ) -> Result<KernelUpdate<'a>, Self::Error> {
            self.calls.lock().unwrap().push("kernel");
            self.scratch
                .copy_from(state.abundance())
                .map_err(|_| TestError)?;
            self.scratch.map_in_place(|value| value + 1.0);
            Ok(KernelUpdate::abundance(&self.scratch))
        }
    }

    #[derive(Debug)]
    struct TestNoise {
        calls: CallLog,
        fail: bool,
        noop: bool,
    }

    impl NoiseAlgorithm for TestNoise {
        type Error = TestError;

        fn is_noop(&self) -> bool {
            self.noop
        }

        fn validate(
            &self,
            _abundance: &AggregateAbundance,
            _space: &SpatialAbundance,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn apply(
            &mut self,
            abundance: &mut AggregateAbundance,
            _space: &mut SpatialAbundance,
            _time_step: TimeStep,
        ) -> Result<(), Self::Error> {
            self.calls.lock().unwrap().push("noise");
            if self.fail {
                return Err(TestError);
            }
            abundance.map_in_place(|value| value + 2.0);
            Ok(())
        }
    }

    #[derive(Debug)]
    struct TestInvariant(CallLog);

    impl InvariantPolicy for TestInvariant {
        type Error = TestError;

        fn validate(
            &self,
            _abundance: &AggregateAbundance,
            _space: &SpatialAbundance,
            _total: &TotalAbundance,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn enforce(
            &mut self,
            abundance: &mut AggregateAbundance,
            _space: &mut SpatialAbundance,
            total: &mut TotalAbundance,
        ) -> Result<(), Self::Error> {
            self.0.lock().unwrap().push("invariant");
            *total = abundance.sum_serial();
            Ok(())
        }
    }

    fn state(time: StateTime) -> SystemState {
        let mut state = ecological_state_schema()
            .resolve()
            .unwrap()
            .create_empty_state(time);
        state
            .insert_payload(
                ABUNDANCE_FIELD,
                Tensor::from_values(&[1], Backend::Dense, vec![1.0]).unwrap(),
            )
            .unwrap();
        state
            .insert_payload(SPACE_FIELD, SpatialAbundance::None)
            .unwrap();
        state.insert_payload(TOTAL_FIELD, 1.0_f64).unwrap();
        state
    }

    fn engine(
        time: StateTime,
        calls: CallLog,
        fail: bool,
        noop: bool,
    ) -> Engine<TestKernel, TestNoise, TestInvariant> {
        let interaction = InteractionMatrix::from_rows(vec![vec![0.0]]).unwrap();
        Engine::new(
            state(time),
            Kernel::new(
                KernelCore::new(interaction),
                TestKernel {
                    scratch: Tensor::zeros(&[1], Backend::Dense).unwrap(),
                    calls: Arc::clone(&calls),
                },
            ),
            Noise::new(TestNoise {
                calls: Arc::clone(&calls),
                fail,
                noop,
            }),
            TestInvariant(calls),
            TimeStep::new(0.5).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn active_noise_preserves_shared_step_order() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let initial = StateTime::from_iteration_and_physical_time(3, 1.0).unwrap();
        let mut engine = engine(initial, Arc::clone(&calls), false, false);
        let advanced = engine.step().unwrap();
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            ["kernel", "invariant", "noise", "invariant"]
        );
        assert_eq!(advanced.iteration(), 4);
        assert_eq!(advanced.physical_time(), Some(1.5));
        assert_eq!(
            engine
                .state()
                .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
                .unwrap()
                .as_slice()[0],
            4.0
        );
        assert_eq!(engine.into_state().time(), advanced);
    }

    #[test]
    fn noop_noise_skips_application_and_second_invariant() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let time = StateTime::from_iteration_and_physical_time(0, 0.0).unwrap();
        let mut engine = engine(time, Arc::clone(&calls), false, true);
        engine.step().unwrap();
        assert_eq!(calls.lock().unwrap().as_slice(), ["kernel", "invariant"]);
    }

    #[test]
    fn noise_failure_stops_later_work_without_advancing_time() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let time = StateTime::from_iteration_and_physical_time(0, 0.0).unwrap();
        let mut engine = engine(time, Arc::clone(&calls), true, false);
        assert!(matches!(engine.step(), Err(EngineStepError::Noise(_))));
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            ["kernel", "invariant", "noise"]
        );
        assert_eq!(engine.state().time(), time);
    }

    #[test]
    fn impossible_time_advance_fails_before_scientific_mutation() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let time = StateTime::from_iteration_and_physical_time(u64::MAX - 1, 0.0).unwrap();
        let mut engine = engine(time, Arc::clone(&calls), false, false);
        engine.step().unwrap();
        calls.lock().unwrap().clear();
        assert!(matches!(
            engine.step(),
            Err(EngineStepError::Time(StateError::IterationOverflow {
                iteration: u64::MAX
            }))
        ));
        assert!(calls.lock().unwrap().is_empty());
    }

    #[test]
    fn missing_physical_time_is_rejected_before_validation() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let interaction = InteractionMatrix::from_rows(vec![vec![0.0]]).unwrap();
        let result = Engine::new(
            state(StateTime::from_iteration(7)),
            Kernel::new(
                KernelCore::new(interaction),
                TestKernel {
                    scratch: Tensor::zeros(&[1], Backend::Dense).unwrap(),
                    calls: Arc::clone(&calls),
                },
            ),
            Noise::new(TestNoise {
                calls: Arc::clone(&calls),
                fail: false,
                noop: false,
            }),
            TestInvariant(Arc::clone(&calls)),
            TimeStep::new(0.5).unwrap(),
        );
        assert!(matches!(
            result,
            Err(EngineBuildError::Time(StateError::MissingPhysicalTime {
                iteration: 7
            }))
        ));
        assert!(calls.lock().unwrap().is_empty());
    }
}
