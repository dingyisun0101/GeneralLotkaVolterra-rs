use general_lotka_volterra_rs::kernel::{
    Kernel, KernelAlgorithm, KernelCore, KernelStateView, KernelStepError, KernelUpdate,
    KernelUpdateError,
};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, TimeStep,
    load_state_schema,
};
use physics_in_parallel::prelude::basic::{DenseMatrix, Tensor};
use scientific_workflow::prelude::{StateTime, SystemState};
use support::interaction_from_array;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("test algorithm failed")]
struct AlgorithmError;

#[derive(Debug)]
struct InvalidBothUpdate {
    abundance: Tensor<f64>,
    space: Tensor<f64>,
}

impl KernelAlgorithm for InvalidBothUpdate {
    type Error = AlgorithmError;

    fn validate(&self, _core: &KernelCore, _state: KernelStateView<'_>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn compute<'algorithm>(
        &'algorithm mut self,
        _core: &KernelCore,
        _state: KernelStateView<'_>,
        _time_step: TimeStep,
    ) -> Result<KernelUpdate<'algorithm>, Self::Error> {
        self.abundance.fill(9.0);
        self.space.fill(f64::NAN);
        Ok(KernelUpdate::both(&self.abundance, &self.space))
    }
}

fn spatial_state() -> SystemState {
    let schema = load_state_schema().unwrap();
    let time = StateTime::from_iteration_and_physical_time(7, 1.5).unwrap();
    let mut state = schema.create_empty_state(time);
    state
        .insert_payload(ABUNDANCE_FIELD, Tensor::from_vec(&[2], vec![1.0, 2.0]))
        .unwrap();
    state
        .insert_payload(SPACE_FIELD, Some(Tensor::from_vec(&[1, 2], vec![1.0, 2.0])))
        .unwrap();
    state.insert_payload(TOTAL_FIELD, 3.0_f64).unwrap();
    state
}

#[test]
fn invalid_multi_payload_update_commits_nothing() {
    let matrix =
        interaction_from_array(DenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0])).unwrap();
    let algorithm = InvalidBothUpdate {
        abundance: Tensor::zeros(&[2]),
        space: Tensor::zeros(&[1, 2]),
    };
    let mut kernel = Kernel::new(KernelCore::new(matrix), algorithm);
    let mut state = spatial_state();
    let initial_time = state.time();

    assert!(matches!(
        kernel.step(&mut state, TimeStep::new(0.1).unwrap()),
        Err(KernelStepError::Update(KernelUpdateError::NonFiniteValue {
            target: SPACE_FIELD,
            ..
        }))
    ));
    assert_eq!(
        state
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &Tensor::from_vec(&[2], vec![1.0, 2.0])
    );
    assert_eq!(
        state.payload::<SpatialAbundance>(SPACE_FIELD).unwrap(),
        &Some(Tensor::from_vec(&[1, 2], vec![1.0, 2.0]))
    );
    assert_eq!(state.time(), initial_time);
}
mod support;
