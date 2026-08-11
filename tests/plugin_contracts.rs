use general_lotka_volterra_rs::invariant::{self, InvariantPolicy};
use general_lotka_volterra_rs::kernel::{
    InMemorySource, InteractionSource, InteractionSourceError, Kernel, KernelAlgorithm, KernelCore,
    KernelCoreError, KernelStateView, KernelStepError, KernelUpdate,
};
use general_lotka_volterra_rs::noise::{Noise, NoiseAlgorithm};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, TimeStep,
    TimeStepError, TotalAbundance, load_state_schema,
};
use ndarray::{Array1, Array2, ArrayD, IxDyn, arr2};
use scientific_workflow::system_state::{SimulationTime, SystemState};
use thiserror::Error;

#[derive(Debug, Error)]
enum TestPluginError {
    #[error("plugin requires spatial abundance")]
    SpatialRequired,
    #[error("abundance length {actual} does not match species count {expected}")]
    SpeciesMismatch { expected: usize, actual: usize },
}

fn state(abundance: Vec<f64>, space: SpatialAbundance, total: f64) -> SystemState {
    let schema = load_state_schema().unwrap();
    let time = SimulationTime::from_iteration_and_physical_time(0, 0.0).unwrap();
    let mut state = schema.create_empty_state(time);
    assert!(
        state
            .insert_payload(ABUNDANCE_FIELD, Array1::from_vec(abundance))
            .unwrap()
            .is_none()
    );
    assert!(state.insert_payload(SPACE_FIELD, space).unwrap().is_none());
    assert!(state.insert_payload(TOTAL_FIELD, total).unwrap().is_none());
    state
}

#[derive(Debug)]
struct EulerInteraction {
    require_space: bool,
    scratch: Array1<f64>,
    steps: usize,
}

impl EulerInteraction {
    fn new(require_space: bool) -> Self {
        Self {
            require_space,
            scratch: Array1::zeros(0),
            steps: 0,
        }
    }
}

impl KernelAlgorithm for EulerInteraction {
    type Error = TestPluginError;

    fn validate(&self, core: &KernelCore, state: KernelStateView<'_>) -> Result<(), Self::Error> {
        let abundance = state.abundance();
        if abundance.len() != core.species() {
            return Err(TestPluginError::SpeciesMismatch {
                expected: core.species(),
                actual: abundance.len(),
            });
        }
        if self.require_space && state.space().is_none() {
            return Err(TestPluginError::SpatialRequired);
        }
        Ok(())
    }

    fn compute<'algorithm>(
        &'algorithm mut self,
        core: &KernelCore,
        state: KernelStateView<'_>,
        time_step: TimeStep,
    ) -> Result<KernelUpdate<'algorithm>, Self::Error> {
        if self.scratch.len() != core.species() {
            self.scratch = Array1::zeros(core.species());
        }
        let abundance = state.abundance();
        core.apply_interaction(
            abundance.as_slice().expect("test abundance is contiguous"),
            self.scratch
                .as_slice_mut()
                .expect("test scratch is contiguous"),
        )
        .expect("validated test dimensions match");
        for (proposed, current) in self.scratch.iter_mut().zip(abundance) {
            *proposed = *current + time_step.get() * *proposed;
        }
        self.steps += 1;
        Ok(KernelUpdate::abundance(self.scratch.view()))
    }
}

#[derive(Debug)]
struct AdditiveNoise {
    updates: usize,
}

impl NoiseAlgorithm for AdditiveNoise {
    type Error = TestPluginError;

    fn validate(
        &self,
        abundance: &AggregateAbundance,
        _space: &SpatialAbundance,
    ) -> Result<(), Self::Error> {
        if abundance.is_empty() {
            return Err(TestPluginError::SpeciesMismatch {
                expected: 1,
                actual: 0,
            });
        }
        Ok(())
    }

    fn apply(
        &mut self,
        abundance: &mut AggregateAbundance,
        _space: &mut SpatialAbundance,
        time_step: TimeStep,
    ) -> Result<(), Self::Error> {
        abundance
            .iter_mut()
            .for_each(|value| *value += time_step.get());
        self.updates += 1;
        Ok(())
    }
}

#[derive(Debug)]
struct SumInvariant;

impl InvariantPolicy for SumInvariant {
    type Error = TestPluginError;

    fn validate(
        &self,
        abundance: &AggregateAbundance,
        _space: &SpatialAbundance,
        _total: &TotalAbundance,
    ) -> Result<(), Self::Error> {
        if abundance.is_empty() {
            return Err(TestPluginError::SpeciesMismatch {
                expected: 1,
                actual: 0,
            });
        }
        Ok(())
    }

    fn enforce(
        &mut self,
        abundance: &mut AggregateAbundance,
        _space: &mut SpatialAbundance,
        total: &mut TotalAbundance,
    ) -> Result<(), Self::Error> {
        *total = abundance.sum();
        Ok(())
    }
}

#[test]
fn kernel_core_validates_and_applies_one_shared_matrix() {
    let matrix = InMemorySource::new(arr2(&[[2.0, -1.0], [0.5, 3.0]]))
        .resolve(2)
        .unwrap();
    let core = KernelCore::new(matrix);
    let shared = core.shared_interaction();
    assert!(std::ptr::eq(core.interaction(), shared.as_ref()));

    let mut output = [0.0; 2];
    core.apply_interaction(&[4.0, 2.0], &mut output).unwrap();
    assert_eq!(output, [6.0, 8.0]);

    assert!(matches!(
        InMemorySource::new(Array2::zeros((2, 3))).resolve(2),
        Err(InteractionSourceError::NonSquare { .. })
    ));
    assert!(matches!(
        InMemorySource::new(Array2::from_shape_vec((1, 1), vec![f64::NAN]).unwrap()).resolve(1),
        Err(InteractionSourceError::NonFiniteEntry { .. })
    ));
    assert!(matches!(
        core.apply_interaction(&[1.0], &mut output),
        Err(KernelCoreError::InputLength { .. })
    ));
}

#[test]
fn plugins_mutate_only_borrowed_payloads_and_never_advance_time() {
    let core = KernelCore::new(
        InMemorySource::new(arr2(&[[0.0, 1.0], [1.0, 0.0]]))
            .resolve(2)
            .unwrap(),
    );
    let mut kernel = Kernel::new(core, EulerInteraction::new(false));
    let mut noise = Noise::new(AdditiveNoise { updates: 0 });
    let mut invariant = SumInvariant;
    let mut state = state(vec![1.0, 2.0], None, 3.0);
    let initial_time = state.simulation_time();

    kernel.validate_state(&state).unwrap();
    noise.validate_state(&state).unwrap();
    invariant::validate_state(&invariant, &state).unwrap();

    kernel
        .step(&mut state, TimeStep::new(0.5).unwrap())
        .unwrap();
    invariant::enforce_state(&mut invariant, &mut state).unwrap();
    noise
        .apply(&mut state, TimeStep::new(0.25).unwrap())
        .unwrap();
    invariant::enforce_state(&mut invariant, &mut state).unwrap();

    assert_eq!(state.simulation_time(), initial_time);
    assert_eq!(
        state
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &Array1::from_vec(vec![2.25, 2.75])
    );
    assert_eq!(*state.payload::<TotalAbundance>(TOTAL_FIELD).unwrap(), 5.0);
    assert_eq!(kernel.algorithm().steps, 1);
    assert_eq!(noise.algorithm().updates, 1);
}

#[test]
fn incompatible_spatial_kernel_fails_validation_before_evolution() {
    let core = KernelCore::new(InMemorySource::new(Array2::eye(2)).resolve(2).unwrap());
    let kernel = Kernel::new(core, EulerInteraction::new(true));
    let state = state(vec![0.5, 0.5], None, 1.0);

    assert!(matches!(
        kernel.validate_state(&state),
        Err(KernelStepError::Algorithm(TestPluginError::SpatialRequired))
    ));
    assert_eq!(kernel.algorithm().steps, 0);
}

#[test]
fn validated_time_steps_reject_invalid_increments_before_mutation() {
    let core = KernelCore::new(InMemorySource::new(Array2::eye(2)).resolve(2).unwrap());
    let kernel = Kernel::new(core, EulerInteraction::new(false));
    let noise = Noise::new(AdditiveNoise { updates: 0 });
    let state = state(vec![0.5, 0.5], Some(ArrayD::zeros(IxDyn(&[1, 2]))), 1.0);

    assert!(matches!(
        TimeStep::new(f64::NAN),
        Err(TimeStepError::NonFinite { .. })
    ));
    assert!(matches!(
        TimeStep::new(-0.1),
        Err(TimeStepError::NonPositive { .. })
    ));
    assert_eq!(kernel.algorithm().steps, 0);
    assert_eq!(noise.algorithm().updates, 0);
    assert_eq!(
        state
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &Array1::from_vec(vec![0.5, 0.5])
    );
}
