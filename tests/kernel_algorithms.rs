use general_lotka_volterra_rs::kernel::{
    Boundary, Diffusion, InMemorySource, InteractionSource, Kernel, KernelAlgorithmError,
    KernelCore, SpatialGeneralLotkaVolterraRk2, SpatialLayout, SpatialReplicatorRk2,
};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, SPACE_FIELD, SpatialAbundance, TOTAL_FIELD, load_state_schema,
};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use scientific_workflow::system_state::{SimulationTime, SystemState};

fn state(abundance: Vec<f64>, space: SpatialAbundance, total: f64) -> SystemState {
    let time = SimulationTime::from_iteration_and_physical_time(0, 0.0).unwrap();
    let mut state = load_state_schema().unwrap().create_empty_state(time);
    state
        .insert_payload(ABUNDANCE_FIELD, Array1::from_vec(abundance))
        .unwrap();
    state.insert_payload(SPACE_FIELD, space).unwrap();
    state.insert_payload(TOTAL_FIELD, total).unwrap();
    state
}

#[test]
fn spatial_facilities_validate_layout_diffusion_and_stability() {
    let layout = SpatialLayout::new(vec![2, 3, 4]).unwrap();
    assert_eq!(layout.shape(), [2, 3, 4]);
    assert_eq!(layout.spatial_dimensions(), 2);
    assert_eq!(layout.species(), 4);
    assert_eq!(layout.cells(), 6);
    assert_eq!(layout.elements(), 24);
    assert!(matches!(
        SpatialLayout::new(vec![4]),
        Err(KernelAlgorithmError::SpatialRank)
    ));
    assert!(matches!(
        Diffusion::new(Array1::from_vec(vec![-0.1]), vec![1.0], Boundary::Neumann),
        Err(KernelAlgorithmError::InvalidDiffusion { .. })
    ));

    let diffusion =
        Diffusion::new(Array1::from_vec(vec![0.5]), vec![1.0], Boundary::Neumann).unwrap();
    assert_eq!(diffusion.boundary(), Boundary::Neumann);
    let algorithm =
        SpatialGeneralLotkaVolterraRk2::new(vec![2, 1], Array1::zeros(1), diffusion).unwrap();
    algorithm
        .validate_time_step(general_lotka_volterra_rs::TimeStep::new(1.0).unwrap())
        .unwrap();
    assert!(matches!(
        algorithm.validate_time_step(general_lotka_volterra_rs::TimeStep::new(1.01).unwrap()),
        Err(KernelAlgorithmError::UnstableTimeStep { .. })
    ));
}

#[test]
fn spatial_kernel_rejects_non_contiguous_storage() {
    let shape = vec![2, 2, 3];
    let mut values = ArrayD::from_shape_vec(
        IxDyn(&shape),
        vec![0.2, 0.3, 0.5, 0.4, 0.2, 0.4, 0.1, 0.7, 0.2, 0.3, 0.3, 0.4],
    )
    .unwrap();
    values.swap_axes(0, 1);
    assert!(values.as_slice().is_none());

    let interaction = InMemorySource::new(Array2::zeros((3, 3)))
        .resolve(3)
        .unwrap();
    let algorithm = SpatialReplicatorRk2::new(
        shape,
        Array1::zeros(3),
        Diffusion::unit_spacing(Array1::zeros(3), 2, Boundary::Periodic).unwrap(),
    )
    .unwrap();
    let kernel = Kernel::new(KernelCore::new(interaction), algorithm);
    let state = state(vec![0.25, 0.35, 0.4], Some(values), 1.0);

    assert!(matches!(
        kernel.validate_state(&state),
        Err(
            general_lotka_volterra_rs::kernel::KernelStepError::Algorithm(
                KernelAlgorithmError::NonStandardLayout
            )
        )
    ));
}
