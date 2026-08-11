use std::convert::Infallible;

use general_lotka_volterra_rs::invariant::FrequencyInvariant;
use general_lotka_volterra_rs::kernel::{
    Boundary, Diffusion, InMemorySource, InteractionSource, Kernel, KernelAlgorithm, KernelCore,
    KernelStateView, KernelUpdate,
};
use general_lotka_volterra_rs::noise::{Noise, NoiseDomain, ProportionalGaussian};
use general_lotka_volterra_rs::simulation::{
    DefaultSimulationBuildError, SimulationBuildError, SimulationKind,
};
use general_lotka_volterra_rs::{
    ABUNDANCE_FIELD, AbundanceRepresentation, AggregateAbundance, MeanFieldReplicator,
    MeanFieldReplicatorConfig, SPACE_FIELD, SpatialAbundance, SpatialGeneralLotkaVolterra,
    SpatialGeneralLotkaVolterraConfig, SpatialReplicator, SpatialReplicatorConfig, TOTAL_FIELD,
    TimeStep, TotalAbundance,
};
use ndarray::{Array1, Array2, ArrayD, IxDyn};

fn interaction(species: usize) -> general_lotka_volterra_rs::kernel::InteractionMatrix {
    InMemorySource::new(Array2::zeros((species, species)))
        .resolve(species)
        .unwrap()
}

fn time_step() -> TimeStep {
    TimeStep::new(0.1).unwrap()
}

fn mean_field_config(species: usize) -> MeanFieldReplicatorConfig {
    MeanFieldReplicatorConfig::new(Array1::zeros(species), 0.0, time_step())
}

fn spatial_replicator_config(shape: &[usize]) -> SpatialReplicatorConfig {
    let species = *shape.last().unwrap();
    SpatialReplicatorConfig::new(
        shape.to_vec(),
        Array1::zeros(species),
        Diffusion::unit_spacing(Array1::zeros(species), shape.len() - 1, Boundary::Periodic)
            .unwrap(),
        0.0,
        time_step(),
    )
}

fn spatial_general_lotka_volterra_config(shape: &[usize]) -> SpatialGeneralLotkaVolterraConfig {
    let species = *shape.last().unwrap();
    SpatialGeneralLotkaVolterraConfig::new(
        shape.to_vec(),
        Array1::zeros(species),
        Diffusion::unit_spacing(Array1::zeros(species), shape.len() - 1, Boundary::Neumann)
            .unwrap(),
        0.0,
        None,
        time_step(),
    )
}

fn space(shape: &[usize], values: Vec<f64>) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(shape), values).unwrap()
}

fn assert_array_close(actual: &Array1<f64>, expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "{actual} != {expected}"
        );
    }
}

#[derive(Debug)]
struct IdentityAggregate {
    output: Array1<f64>,
}

impl KernelAlgorithm for IdentityAggregate {
    type Error = Infallible;

    fn validate(&self, _core: &KernelCore, _state: KernelStateView<'_>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn compute<'algorithm>(
        &'algorithm mut self,
        _core: &KernelCore,
        state: KernelStateView<'_>,
        _time_step: TimeStep,
    ) -> Result<KernelUpdate<'algorithm>, Self::Error> {
        self.output.assign(state.abundance());
        Ok(KernelUpdate::abundance(self.output.view()))
    }
}

#[test]
fn root_mean_field_api_constructs_steps_and_reconstructs() {
    let mut simulation = MeanFieldReplicator::new(
        Array1::from_vec(vec![0.4, 0.6]),
        interaction(2),
        mean_field_config(2),
    )
    .unwrap();
    assert_eq!(simulation.kind(), SimulationKind::MeanFieldReplicator);
    assert_eq!(
        simulation.abundance_representation(),
        AbundanceRepresentation::RelativeFrequency
    );
    assert_eq!(simulation.time_step(), time_step());
    assert!(
        simulation
            .state()
            .payload::<SpatialAbundance>(SPACE_FIELD)
            .unwrap()
            .is_none()
    );

    assert_eq!(simulation.step().unwrap().iteration(), 1);
    let state = simulation.into_state();
    let mut reconstructed = MeanFieldReplicator::from_state(
        state,
        AbundanceRepresentation::RelativeFrequency,
        interaction(2),
        mean_field_config(2),
    )
    .unwrap();
    assert_eq!(reconstructed.step().unwrap().iteration(), 2);
    assert_eq!(
        reconstructed
            .state()
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &Array1::from_vec(vec![0.4, 0.6])
    );
}

#[test]
fn root_spatial_apis_derive_canonical_aggregates_and_step() {
    let initial_frequency = space(&[2, 2], vec![0.8, 0.2, 0.4, 0.6]);
    let mut replicator = SpatialReplicator::new(
        initial_frequency.clone(),
        interaction(2),
        spatial_replicator_config(&[2, 2]),
    )
    .unwrap();
    assert_eq!(replicator.kind(), SimulationKind::SpatialReplicator);
    assert_array_close(
        replicator
            .state()
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &[0.6, 0.4],
    );
    replicator.step().unwrap();
    assert_eq!(
        replicator
            .state()
            .payload::<SpatialAbundance>(SPACE_FIELD)
            .unwrap()
            .as_ref()
            .unwrap(),
        &initial_frequency
    );
    let mut replicator = SpatialReplicator::from_state(
        replicator.into_state(),
        AbundanceRepresentation::RelativeFrequency,
        interaction(2),
        spatial_replicator_config(&[2, 2]),
    )
    .unwrap();
    assert_eq!(replicator.step().unwrap().iteration(), 2);

    let initial_population = space(&[2, 2], vec![1.0, 0.5, 1.4, 1.5]);
    let mut glv = SpatialGeneralLotkaVolterra::new(
        initial_population.clone(),
        interaction(2),
        spatial_general_lotka_volterra_config(&[2, 2]),
    )
    .unwrap();
    assert_eq!(glv.kind(), SimulationKind::SpatialGeneralLotkaVolterra);
    assert_eq!(
        glv.abundance_representation(),
        AbundanceRepresentation::AbsoluteCount
    );
    assert_array_close(
        glv.state()
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &[2.4, 2.0],
    );
    assert_eq!(
        *glv.state().payload::<TotalAbundance>(TOTAL_FIELD).unwrap(),
        4.0
    );
    glv.step().unwrap();
    assert_eq!(
        glv.state()
            .payload::<SpatialAbundance>(SPACE_FIELD)
            .unwrap()
            .as_ref()
            .unwrap(),
        &initial_population
    );
    let mut glv = SpatialGeneralLotkaVolterra::from_state(
        glv.into_state(),
        AbundanceRepresentation::AbsoluteCount,
        interaction(2),
        spatial_general_lotka_volterra_config(&[2, 2]),
    )
    .unwrap();
    assert_eq!(glv.step().unwrap().iteration(), 2);
}

#[test]
fn construction_rejects_representation_shape_and_matrix_mismatches() {
    let simulation = MeanFieldReplicator::new(
        Array1::from_vec(vec![0.5, 0.5]),
        interaction(2),
        mean_field_config(2),
    )
    .unwrap();
    let state = simulation.into_state();
    assert!(matches!(
        MeanFieldReplicator::from_state(
            state,
            AbundanceRepresentation::AbsoluteCount,
            interaction(2),
            mean_field_config(2),
        ),
        Err(DefaultSimulationBuildError::Composition(
            SimulationBuildError::RepresentationMismatch { .. }
        ))
    ));

    assert!(
        MeanFieldReplicator::new(
            Array1::from_vec(vec![0.5, 0.5]),
            interaction(1),
            mean_field_config(2),
        )
        .is_err()
    );

    let spatial = SpatialReplicator::new(
        space(&[2, 2], vec![0.5; 4]),
        interaction(2),
        spatial_replicator_config(&[2, 2]),
    )
    .unwrap();
    assert!(
        SpatialReplicator::from_state(
            spatial.into_state(),
            AbundanceRepresentation::RelativeFrequency,
            interaction(2),
            spatial_replicator_config(&[1, 2]),
        )
        .is_err()
    );
}

#[test]
fn mean_field_accepts_compatible_custom_kernel_and_noise_plugins() {
    let state = MeanFieldReplicator::new(
        Array1::from_vec(vec![0.25, 0.75]),
        interaction(2),
        mean_field_config(2),
    )
    .unwrap()
    .into_state();
    let algorithm = IdentityAggregate {
        output: Array1::zeros(2),
    };
    let noise = ProportionalGaussian::new(0.0, 42, NoiseDomain::aggregate(2).unwrap()).unwrap();
    let mut simulation = MeanFieldReplicator::from_plugins(
        state,
        AbundanceRepresentation::RelativeFrequency,
        Kernel::new(KernelCore::new(interaction(2)), algorithm),
        Noise::new(noise),
        FrequencyInvariant::new(2, 0.0).unwrap(),
        time_step(),
    )
    .unwrap();

    simulation.step().unwrap();
    assert_eq!(
        simulation
            .state()
            .payload::<AggregateAbundance>(ABUNDANCE_FIELD)
            .unwrap(),
        &Array1::from_vec(vec![0.25, 0.75])
    );
}
