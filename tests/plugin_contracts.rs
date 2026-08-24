use general_lotka_volterra_rs::interaction::InteractionMatrixError;
use general_lotka_volterra_rs::invariant::{self, InvariantPolicy};
use general_lotka_volterra_rs::kernel::{
    Kernel, KernelAlgorithm, KernelCore, KernelCoreError, KernelStateView, KernelStepError,
    KernelUpdate,
};
use general_lotka_volterra_rs::noise::{Noise, NoiseAlgorithm};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, TimeStep,
    TimeStepError, TotalAbundance, load_state_schema,
};
use physics_in_parallel::prelude::basic::{DenseMatrix, Tensor};
use scientific_workflow::system_state::{SimulationTime, SystemState};
use support::interaction_from_array;
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
            .insert_payload(
                ABUNDANCE_FIELD,
                Tensor::from_vec(&[abundance.len()], abundance)
            )
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
    scratch: Tensor<f64>,
    steps: usize,
}

impl EulerInteraction {
    fn new(require_space: bool) -> Self {
        Self {
            require_space,
            scratch: Tensor::zeros(&[1]),
            steps: 0,
        }
    }
}

impl KernelAlgorithm for EulerInteraction {
    type Error = TestPluginError;

    fn validate(&self, core: &KernelCore, state: KernelStateView<'_>) -> Result<(), Self::Error> {
        let abundance = state.abundance();
        if abundance.size() != core.species() {
            return Err(TestPluginError::SpeciesMismatch {
                expected: core.species(),
                actual: abundance.size(),
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
        if self.scratch.size() != core.species() {
            self.scratch = Tensor::zeros(&[core.species()]);
        }
        let abundance = state.abundance();
        core.apply_interaction(abundance.as_slice(), self.scratch.as_mut_slice())
            .expect("validated test dimensions match");
        for (proposed, current) in self
            .scratch
            .as_mut_slice()
            .iter_mut()
            .zip(abundance.as_slice())
        {
            *proposed = *current + time_step.get() * *proposed;
        }
        self.steps += 1;
        Ok(KernelUpdate::abundance(&self.scratch))
    }
}

#[derive(Debug)]
struct AdditiveNoise {
    updates: usize,
}

impl NoiseAlgorithm for AdditiveNoise {
    type Error = TestPluginError;

    fn rng_record(&self) -> Option<&scientific_workflow::rng_record::RngRecord> {
        None
    }

    fn validate(
        &self,
        abundance: &AggregateAbundance,
        _space: &SpatialAbundance,
    ) -> Result<(), Self::Error> {
        if abundance.size() == 0 {
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
            .as_mut_slice()
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
        if abundance.size() == 0 {
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
        *total = abundance.sum_serial();
        Ok(())
    }
}

#[test]
fn kernel_core_validates_and_applies_one_shared_matrix() {
    let matrix =
        interaction_from_array(DenseMatrix::from_vec(2, 2, vec![2.0, -1.0, 0.5, 3.0])).unwrap();
    let core = KernelCore::new(matrix);
    let shared = core.shared_interaction();
    assert!(std::ptr::eq(core.interaction(), shared.as_ref()));

    let mut output = [0.0; 2];
    core.apply_interaction(&[4.0, 2.0], &mut output).unwrap();
    assert_eq!(output, [6.0, 8.0]);

    assert!(matches!(
        interaction_from_array(DenseMatrix::zeros(2, 3)),
        Err(InteractionMatrixError::NonSquare { .. })
    ));
    assert!(matches!(
        interaction_from_array(DenseMatrix::from_vec(1, 1, vec![f64::NAN])),
        Err(InteractionMatrixError::NonFiniteEntry { .. })
    ));
    assert!(matches!(
        core.apply_interaction(&[1.0], &mut output),
        Err(KernelCoreError::InputLength { .. })
    ));
}

#[test]
fn plugins_mutate_only_borrowed_payloads_and_never_advance_time() {
    let core = KernelCore::new(
        interaction_from_array(DenseMatrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0])).unwrap(),
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
        &Tensor::from_vec(&[2], vec![2.25, 2.75])
    );
    assert_eq!(*state.payload::<TotalAbundance>(TOTAL_FIELD).unwrap(), 5.0);
    assert_eq!(kernel.algorithm().steps, 1);
    assert_eq!(noise.algorithm().updates, 1);
}

#[test]
fn incompatible_spatial_kernel_fails_validation_before_evolution() {
    let core = KernelCore::new(
        interaction_from_array(DenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0])).unwrap(),
    );
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
    let core = KernelCore::new(
        interaction_from_array(DenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0])).unwrap(),
    );
    let kernel = Kernel::new(core, EulerInteraction::new(false));
    let noise = Noise::new(AdditiveNoise { updates: 0 });
    let state = state(vec![0.5, 0.5], Some(Tensor::zeros(&[1, 2])), 1.0);

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
        &Tensor::from_vec(&[2], vec![0.5, 0.5])
    );
}
mod support;
