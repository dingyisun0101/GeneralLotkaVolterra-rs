use general_lotka_volterra_rs::interaction::InteractionMatrix;
use general_lotka_volterra_rs::kernel::{
    Kernel, KernelAlgorithm, KernelCore, KernelStateView, KernelStepError, KernelUpdate,
    KernelUpdateError,
};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AggregateAbundance, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, TimeStep,
    load_state_schema,
};
use ndarray::{Array1, ArrayD, IxDyn, arr2};
use scientific_workflow::system_state::{SimulationTime, SystemState};
use thiserror::Error;

#[derive(Debug, Error)]
#[error("test algorithm failed")]
struct AlgorithmError;

#[derive(Debug)]
struct InvalidBothUpdate {
    abundance: Array1<f64>,
    space: ArrayD<f64>,
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
        Ok(KernelUpdate::both(self.abundance.view(), self.space.view()))
    }
}

fn spatial_state() -> SystemState {
    let schema = load_state_schema().unwrap();
    let time = SimulationTime::from_iteration_and_physical_time(7, 1.5).unwrap();
    let mut state = schema.create_empty_state(time);
    state
        .insert_payload(ABUNDANCE_FIELD, Array1::from_vec(vec![1.0, 2.0]))
        .unwrap();
    state
        .insert_payload(
            SPACE_FIELD,
            Some(ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap()),
        )
        .unwrap();
    state.insert_payload(TOTAL_FIELD, 3.0_f64).unwrap();
    state
}

#[test]
fn invalid_multi_payload_update_commits_nothing() {
    let matrix = InteractionMatrix::from_array(arr2(&[[1.0, 0.0], [0.0, 1.0]]), 2).unwrap();
    let algorithm = InvalidBothUpdate {
        abundance: Array1::zeros(2),
        space: ArrayD::zeros(IxDyn(&[1, 2])),
    };
    let mut kernel = Kernel::new(KernelCore::new(matrix), algorithm);
    let mut state = spatial_state();
    let initial_time = state.simulation_time();

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
        &Array1::from_vec(vec![1.0, 2.0])
    );
    assert_eq!(
        state.payload::<SpatialAbundance>(SPACE_FIELD).unwrap(),
        &Some(ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap())
    );
    assert_eq!(state.simulation_time(), initial_time);
}
