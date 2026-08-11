use std::sync::{Arc, Mutex};

use general_lotka_volterra_rs::engine::{Engine, EngineStepError};
use general_lotka_volterra_rs::invariant::InvariantPolicy;
use general_lotka_volterra_rs::kernel::{
    InMemorySource, InteractionSource, Kernel, KernelAlgorithm, KernelCore, KernelStateView,
    KernelUpdate,
};
use general_lotka_volterra_rs::noise::{Noise, NoiseAlgorithm};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, TimeStep,
    TotalAbundance, load_state_schema,
};
use ndarray::{Array1, arr2};
use scientific_workflow::system_state::{SimulationTime, StateError, SystemState};
use thiserror::Error;

type CallLog = Arc<Mutex<Vec<&'static str>>>;

#[derive(Clone, Copy, Debug, Error)]
#[error("controlled test failure")]
struct TestError;

#[derive(Debug)]
struct TestKernel {
    scratch: Array1<f64>,
    calls: CallLog,
}

impl KernelAlgorithm for TestKernel {
    type Error = TestError;

    fn validate(&self, core: &KernelCore, state: KernelStateView<'_>) -> Result<(), Self::Error> {
        if state.abundance().len() == core.species() {
            Ok(())
        } else {
            Err(TestError)
        }
    }

    fn compute<'algorithm>(
        &'algorithm mut self,
        _core: &KernelCore,
        state: KernelStateView<'_>,
        _time_step: TimeStep,
    ) -> Result<KernelUpdate<'algorithm>, Self::Error> {
        self.calls.lock().unwrap().push("kernel");
        self.scratch.assign(state.abundance());
        self.scratch.mapv_inplace(|value| value + 1.0);
        Ok(KernelUpdate::abundance(self.scratch.view()))
    }
}

#[derive(Debug)]
struct TestNoise {
    calls: CallLog,
    fail: bool,
}

impl NoiseAlgorithm for TestNoise {
    type Error = TestError;

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
        abundance.mapv_inplace(|value| value + 2.0);
        Ok(())
    }
}

#[derive(Debug)]
struct TestInvariant {
    calls: CallLog,
}

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
        self.calls.lock().unwrap().push("invariant");
        *total = abundance.sum();
        Ok(())
    }
}

fn state(time: SimulationTime) -> SystemState {
    let mut state = load_state_schema().unwrap().create_empty_state(time);
    state
        .insert_payload(ABUNDANCE_FIELD, Array1::from_vec(vec![1.0]))
        .unwrap();
    state
        .insert_payload(SPACE_FIELD, SpatialAbundance::None)
        .unwrap();
    state.insert_payload(TOTAL_FIELD, 1.0_f64).unwrap();
    state
}

fn engine(
    time: SimulationTime,
    calls: CallLog,
    fail_noise: bool,
) -> Engine<TestKernel, TestNoise, TestInvariant> {
    let interaction = InMemorySource::new(arr2(&[[0.0]])).resolve(1).unwrap();
    Engine::new(
        state(time),
        Kernel::new(
            KernelCore::new(interaction),
            TestKernel {
                scratch: Array1::zeros(1),
                calls: Arc::clone(&calls),
            },
        ),
        Noise::new(TestNoise {
            calls: Arc::clone(&calls),
            fail: fail_noise,
        }),
        TestInvariant { calls },
        TimeStep::new(0.5).unwrap(),
    )
    .unwrap()
}

#[test]
fn engine_owns_state_and_applies_the_exact_shared_order() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let initial_time = SimulationTime::from_iteration_and_physical_time(3, 1.0).unwrap();
    let mut engine = engine(initial_time, Arc::clone(&calls), false);

    let advanced = engine.step().unwrap();
    assert_eq!(
        calls.lock().unwrap().as_slice(),
        ["kernel", "invariant", "noise", "invariant"]
    );
    assert_eq!(advanced.iteration(), 4);
    assert_eq!(advanced.physical_time(), Some(1.5));
    assert_eq!(engine.state().simulation_time(), advanced);
    assert_eq!(
        engine
            .state()
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &Array1::from_vec(vec![4.0])
    );
    assert_eq!(
        *engine
            .state()
            .payload::<TotalAbundance>(TOTAL_FIELD)
            .unwrap(),
        4.0
    );

    let state = engine.into_state();
    assert_eq!(state.simulation_time(), advanced);
}

#[test]
fn phase_failure_stops_later_work_and_does_not_advance_time() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let initial_time = SimulationTime::from_iteration_and_physical_time(0, 0.0).unwrap();
    let mut engine = engine(initial_time, Arc::clone(&calls), true);

    assert!(matches!(engine.step(), Err(EngineStepError::Noise(_))));
    assert_eq!(
        calls.lock().unwrap().as_slice(),
        ["kernel", "invariant", "noise"]
    );
    assert_eq!(engine.state().simulation_time(), initial_time);
    assert_eq!(
        engine
            .state()
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &Array1::from_vec(vec![2.0])
    );
    assert_eq!(
        *engine
            .state()
            .payload::<TotalAbundance>(TOTAL_FIELD)
            .unwrap(),
        2.0
    );
}

#[test]
fn impossible_time_advance_fails_before_any_scientific_mutation() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let initial_time = SimulationTime::from_iteration_and_physical_time(u64::MAX - 1, 0.0).unwrap();
    let mut engine = engine(initial_time, Arc::clone(&calls), false);
    engine.step().unwrap();
    calls.lock().unwrap().clear();
    let abundance_before = engine
        .state()
        .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
        .unwrap()
        .clone();

    assert!(matches!(
        engine.step(),
        Err(EngineStepError::Time(StateError::IterationOverflow {
            iteration: u64::MAX
        }))
    ));
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(engine.state().simulation_time().iteration(), u64::MAX);
    assert_eq!(
        engine
            .state()
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &abundance_before
    );
}
