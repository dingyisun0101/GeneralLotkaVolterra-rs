//! Workflow execution-unit integration for built-in GLV dynamics.

use ecological_state_toolkit::inputs::{
    EcologicalInputs, EcologicalInputsError, ResolvedEcologicalInputs,
};
use ecological_state_toolkit::state_schema::ecological_state_schema;
use ecological_state_toolkit::terminal_state::{
    StopReason, TERMINAL_STATE_METADATA_KEY, TerminationSignal,
};
use ecological_state_toolkit::trajectory::{
    AbundanceView, DetectionPolicy, EquilibriumEvidence, EquilibriumPolicy, PeriodicOrbitPolicy,
    ResidualTolerance, TerminalPolicy, TrajectoryObservation, TrajectoryObservationPolicy,
    TrajectoryObserver, TrajectoryObserverError,
};
use physics_in_parallel::prelude::basic::{RngConfig, Tensor};
use scientific_workflow::prelude::{
    ExecutionUnit, InitializationContext, MemberCompletion, MemberView, ObservationPlan,
    ObservationStream, SystemState, SystemStateSchema, UnitResult,
};
use scientific_workflow::state::StateSchemaProvider;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::initialization::categorical_to_species_field;
use crate::invariant::FrequencyInvariant;
use crate::kernel::{Kernel, KernelAlgorithm, KernelCore, MeanFieldReplicatorRk4};
use crate::noise::{DemographicGaussian, Noise, NoiseAlgorithm, NoiseDomain};
use crate::simulation::{
    MeanFieldReplicator, MeanFieldReplicatorConfig, SpatialGeneralLotkaVolterra,
    SpatialGeneralLotkaVolterraConfig, SpatialReplicator, SpatialReplicatorConfig, assemble_state,
};
use crate::{ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD, TimeStep};

const NOISE_SEED_PURPOSE: &str = "noise";

/// Complete constants for one Workflow-scheduled GLV execution.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GlvConstants {
    /// Stable identity used for recording and member-scoped seed provenance.
    pub identity: String,
    /// Shared model-ready interaction and canonical initial-state artifacts.
    pub inputs: EcologicalInputs,
    /// Concrete GLV dynamics and their numerical parameters.
    pub model: GlvModelConfig,
    /// Uniform Workflow recording streams.
    #[serde(default)]
    pub recording: GlvObservationConfig,
    /// Terminal-state observation and optional deterministic detection.
    #[serde(default)]
    pub observation: ObservationConfig,
    /// Absolute iteration cap.
    pub maximum_iterations: u64,
}

/// Scalar-or-vector species parameter with dimension inferred from the inputs.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum SpeciesValues {
    Scalar(f64),
    Values(Vec<f64>),
}

impl SpeciesValues {
    fn tensor(
        &self,
        species: usize,
        field: &'static str,
    ) -> Result<Tensor<f64>, GlvExecutionError> {
        let values = match self {
            Self::Scalar(value) => vec![*value; species],
            Self::Values(values) if values.len() == species => values.clone(),
            Self::Values(values) => {
                return Err(GlvExecutionError::SpeciesParameterLength {
                    field,
                    expected: species,
                    actual: values.len(),
                });
            }
        };
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(GlvExecutionError::NonFiniteSpeciesParameter {
                field,
                index,
                value,
            });
        }
        Ok(Tensor::from_vec(&[species], values))
    }

    fn validate(&self, species: usize, field: &'static str) -> Result<(), GlvExecutionError> {
        self.tensor(species, field).map(drop)
    }

    fn nonnegative_tensor(
        &self,
        species: usize,
        field: &'static str,
    ) -> Result<Tensor<f64>, GlvExecutionError> {
        let tensor = self.tensor(species, field)?;
        if let Some(value) = tensor.as_slice().iter().copied().find(|value| *value < 0.0) {
            return Err(GlvExecutionError::InvalidNonnegative { field, value });
        }
        Ok(tensor)
    }
}

/// Built-in GLV scientific compositions.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GlvModelConfig {
    MeanFieldReplicator {
        growth: SpeciesValues,
        extinction_cutoff: f64,
        time_step: f64,
    },
    MeanFieldReplicatorDemographic {
        growth: SpeciesValues,
        extinction_cutoff: f64,
        time_step: f64,
        sigma: f64,
        #[serde(default)]
        rng: RngConfig,
    },
    SpatialReplicator {
        growth: SpeciesValues,
        diffusion: SpeciesValues,
        extinction_cutoff: f64,
        time_step: f64,
    },
    SpatialGeneralLotkaVolterra {
        growth: SpeciesValues,
        diffusion: SpeciesValues,
        extinction_cutoff: f64,
        carrying_capacity: Option<f64>,
        initial_population_per_site: f64,
        time_step: f64,
    },
}

impl GlvModelConfig {
    fn validate(
        &self,
        species: usize,
        lattice: &physics_in_parallel::prelude::basic::SquareLatticeConfig,
    ) -> Result<(), GlvExecutionError> {
        let (growth, cutoff, time_step) = match self {
            Self::MeanFieldReplicator {
                growth,
                extinction_cutoff,
                time_step,
            }
            | Self::MeanFieldReplicatorDemographic {
                growth,
                extinction_cutoff,
                time_step,
                ..
            }
            | Self::SpatialReplicator {
                growth,
                extinction_cutoff,
                time_step,
                ..
            }
            | Self::SpatialGeneralLotkaVolterra {
                growth,
                extinction_cutoff,
                time_step,
                ..
            } => (growth, *extinction_cutoff, *time_step),
        };
        growth.validate(species, "growth")?;
        TimeStep::new(time_step)?;
        if !cutoff.is_finite() || cutoff < 0.0 {
            return Err(GlvExecutionError::InvalidNonnegative {
                field: "extinction_cutoff",
                value: cutoff,
            });
        }
        match self {
            Self::MeanFieldReplicatorDemographic { sigma, .. }
                if !sigma.is_finite() || *sigma < 0.0 =>
            {
                Err(GlvExecutionError::InvalidNonnegative {
                    field: "sigma",
                    value: *sigma,
                })
            }
            Self::SpatialReplicator {
                growth,
                diffusion,
                time_step,
                ..
            } => {
                let diffusion = crate::kernel::Diffusion::new(
                    diffusion.nonnegative_tensor(species, "diffusion")?,
                    lattice.clone(),
                )?;
                let algorithm = crate::kernel::SpatialReplicatorRk2::new(
                    growth.tensor(species, "growth")?,
                    diffusion,
                )?;
                algorithm.validate_time_step(TimeStep::new(*time_step)?)?;
                Ok(())
            }
            Self::SpatialGeneralLotkaVolterra {
                growth,
                diffusion,
                carrying_capacity,
                initial_population_per_site,
                time_step,
                ..
            } => {
                let diffusion = crate::kernel::Diffusion::new(
                    diffusion.nonnegative_tensor(species, "diffusion")?,
                    lattice.clone(),
                )?;
                let algorithm = crate::kernel::SpatialGeneralLotkaVolterraRk2::new(
                    growth.tensor(species, "growth")?,
                    diffusion,
                )?;
                algorithm.validate_time_step(TimeStep::new(*time_step)?)?;
                if carrying_capacity.is_some_and(|value| !value.is_finite() || value < 0.0) {
                    return Err(GlvExecutionError::InvalidNonnegative {
                        field: "carrying_capacity",
                        value: carrying_capacity.unwrap(),
                    });
                }
                if !initial_population_per_site.is_finite() || *initial_population_per_site <= 0.0 {
                    return Err(GlvExecutionError::InvalidPositive {
                        field: "initial_population_per_site",
                        value: *initial_population_per_site,
                    });
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn stochastic(&self) -> bool {
        matches!(self, Self::MeanFieldReplicatorDemographic { .. })
    }
}

/// Sampling cadence for the uniform ecological recording streams.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct GlvObservationConfig {
    pub signal_interval: u64,
    pub space_interval: u64,
    pub checkpoint_interval: u64,
}

impl Default for GlvObservationConfig {
    fn default() -> Self {
        Self {
            signal_interval: 1,
            space_interval: 10,
            checkpoint_interval: 100,
        }
    }
}

/// Bounded terminal observation policy.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum ObservationConfig {
    TerminalOnly,
    Detect {
        #[serde(default = "default_true")]
        equilibrium: bool,
        #[serde(default = "default_true")]
        periodic_orbit: bool,
    },
}

const fn default_true() -> bool {
    true
}

impl Default for ObservationConfig {
    fn default() -> Self {
        Self::Detect {
            equilibrium: true,
            periodic_orbit: true,
        }
    }
}

impl ObservationConfig {
    fn policy(self, interval: u64) -> TrajectoryObservationPolicy {
        let terminal = TerminalPolicy {
            sample_interval_iterations: interval,
            trailing_window_samples: 128,
        };
        match self {
            Self::TerminalOnly => TrajectoryObservationPolicy::TerminalOnly(terminal),
            Self::Detect {
                equilibrium,
                periodic_orbit,
            } => TrajectoryObservationPolicy::Detect(DetectionPolicy {
                terminal,
                start_after_iteration: 0,
                equilibrium: equilibrium.then(|| EquilibriumPolicy {
                    base_window_samples: 16,
                    confirmation_window_multipliers: vec![1, 2, 4],
                    maximum_observable_distance: 1.0e-7,
                    maximum_relative_mass_range: Some(1.0e-7),
                    support_threshold: 1.0e-10,
                    residual_tolerance: ResidualTolerance {
                        absolute: 1.0e-10,
                        relative: 1.0e-8,
                    },
                }),
                periodic_orbit: periodic_orbit.then_some(PeriodicOrbitPolicy {
                    minimum_period_samples: 2,
                    maximum_period_samples: 128,
                    repeated_cycles: 3,
                    maximum_recurrence_distance: 1.0e-6,
                    minimum_orbit_amplitude: 1.0e-4,
                }),
                detect_absorbing_state: false,
            }),
        }
    }
}

trait RuntimeSimulation: Send {
    fn state(&self) -> &SystemState;
    fn step_runtime(&mut self) -> UnitResult;
    fn maximum_scaled_residual(&mut self, tolerance: ResidualTolerance) -> UnitResult<Option<f64>>;
}

impl<A, N> RuntimeSimulation for MeanFieldReplicator<A, N>
where
    A: KernelAlgorithm + Send + 'static,
    N: NoiseAlgorithm + Send + 'static,
{
    fn state(&self) -> &SystemState {
        self.state()
    }

    fn step_runtime(&mut self) -> UnitResult {
        self.step()?;
        Ok(())
    }

    fn maximum_scaled_residual(&mut self, tolerance: ResidualTolerance) -> UnitResult<Option<f64>> {
        Ok(self.maximum_scaled_residual(tolerance.absolute, tolerance.relative)?)
    }
}

impl<A, N> RuntimeSimulation for SpatialReplicator<A, N>
where
    A: KernelAlgorithm + Send + 'static,
    N: NoiseAlgorithm + Send + 'static,
{
    fn state(&self) -> &SystemState {
        self.state()
    }

    fn step_runtime(&mut self) -> UnitResult {
        self.step()?;
        Ok(())
    }

    fn maximum_scaled_residual(&mut self, tolerance: ResidualTolerance) -> UnitResult<Option<f64>> {
        Ok(self.maximum_scaled_residual(tolerance.absolute, tolerance.relative)?)
    }
}

impl<A, N> RuntimeSimulation for SpatialGeneralLotkaVolterra<A, N>
where
    A: KernelAlgorithm + Send + 'static,
    N: NoiseAlgorithm + Send + 'static,
{
    fn state(&self) -> &SystemState {
        self.state()
    }

    fn step_runtime(&mut self) -> UnitResult {
        self.step()?;
        Ok(())
    }

    fn maximum_scaled_residual(&mut self, tolerance: ResidualTolerance) -> UnitResult<Option<f64>> {
        Ok(self.maximum_scaled_residual(tolerance.absolute, tolerance.relative)?)
    }
}

/// One independently stateful GLV execution unit.
pub struct GlvUnit {
    identity: Box<str>,
    simulation: Box<dyn RuntimeSimulation>,
    observer: Option<TrajectoryObserver>,
    completion_reason: Option<Map<String, Value>>,
    maximum_iterations: u64,
}

#[scientific_workflow::execution_unit("glv")]
impl ExecutionUnit for GlvUnit {
    type Constants = GlvConstants;

    fn standard_state_schema() -> Option<StateSchemaProvider> {
        Some(ecological_state_schema())
    }

    fn preflight(
        constants: &Self::Constants,
        schema: &SystemStateSchema,
    ) -> UnitResult<ObservationPlan> {
        validate_constants(constants, schema)?;
        Ok(ObservationPlan::streams([
            ObservationStream::fields("signal", [ABUNDANCE_FIELD, TOTAL_FIELD])?
                .every_iterations(constants.recording.signal_interval)?,
            ObservationStream::fields("space", [ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD])?
                .every_iterations(constants.recording.space_interval)?,
            ObservationStream::fields("checkpoint", [ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD])?
                .every_iterations(constants.recording.checkpoint_interval)?,
        ])?)
    }

    fn initialize(
        constants: Self::Constants,
        schema: &SystemStateSchema,
        context: &InitializationContext,
    ) -> UnitResult<Self> {
        validate_constants(&constants, schema)?;
        let GlvConstants {
            identity,
            inputs,
            model,
            recording,
            observation,
            maximum_iterations,
        } = constants;
        let inputs = resolve_inputs(inputs)?;
        let simulation = build_member(schema, &identity, context, model, inputs)?;
        let mut unit = Self {
            identity: identity.into_boxed_str(),
            simulation,
            observer: TrajectoryObserver::from_policy(
                observation.policy(recording.signal_interval),
            )?,
            completion_reason: None,
            maximum_iterations,
        };
        let signal = unit.observe_current()?;
        if let Some(signal) = signal {
            unit.finish(StopReason::Detected(signal))?;
        } else if maximum_iterations == 0 {
            unit.finish(StopReason::MaximumIterations)?;
        }
        Ok(unit)
    }

    fn member_count(&self) -> usize {
        1
    }

    fn member(&self, index: usize) -> Option<MemberView<'_>> {
        (index == 0).then(|| {
            MemberView::new(
                &self.identity,
                self.simulation.state(),
                self.completion_reason
                    .as_ref()
                    .map(MemberCompletion::with_reason),
                Some(self.maximum_iterations),
            )
        })
    }

    fn step(&mut self) -> UnitResult {
        self.simulation.step_runtime()?;
        if let Some(signal) = self.observe_current()? {
            self.finish(StopReason::Detected(signal))?;
        } else if self.simulation.state().time().iteration() >= self.maximum_iterations {
            self.finish(StopReason::MaximumIterations)?;
        }
        Ok(())
    }
}

impl GlvUnit {
    fn observe_current(&mut self) -> UnitResult<Option<TerminationSignal>> {
        let Some(observer) = self.observer.as_ref() else {
            return Ok(None);
        };
        let iteration = self.simulation.state().time().iteration();
        let evidence = if observer.requires_equilibrium_evidence(iteration) {
            let tolerance = ResidualTolerance {
                absolute: 1.0e-10,
                relative: 1.0e-8,
            };
            EquilibriumEvidence::MaximumScaledResidual {
                value: self
                    .simulation
                    .maximum_scaled_residual(tolerance)?
                    .ok_or(GlvExecutionError::MissingResidual)?,
            }
        } else {
            EquilibriumEvidence::Unavailable
        };
        let observation = trajectory_observation(self.simulation.state(), evidence)?;
        Ok(self
            .observer
            .as_mut()
            .expect("observer presence was checked")
            .observe(observation)?)
    }

    fn finish(&mut self, stop_reason: StopReason) -> UnitResult {
        let observer = self
            .observer
            .take()
            .expect("GLV always retains a terminal observer until completion");
        let terminal = observer.finish(
            trajectory_observation(self.simulation.state(), EquilibriumEvidence::Unavailable)?,
            stop_reason.clone(),
        )?;
        let kind = match &stop_reason {
            StopReason::Detected(signal) => match signal {
                TerminationSignal::Equilibrium(_) => "equilibrium",
                TerminationSignal::PeriodicOrbit(_) => "periodic_orbit",
                TerminationSignal::AbsorbingState(_) => "absorbing_state",
            },
            StopReason::MaximumIterations => "maximum_iterations",
            StopReason::Requested => "requested",
            StopReason::ModelSpecific(_) => "model_specific",
        };
        let mut reason = Map::from_iter([("kind".to_owned(), kind.into())]);
        reason.insert("stop_reason".to_owned(), serde_json::to_value(stop_reason)?);
        reason.insert(
            TERMINAL_STATE_METADATA_KEY.to_owned(),
            serde_json::to_value(terminal)?,
        );
        self.completion_reason = Some(reason);
        Ok(())
    }
}

fn validate_constants(
    constants: &GlvConstants,
    schema: &SystemStateSchema,
) -> Result<(), GlvExecutionError> {
    if constants.identity.is_empty() || constants.identity.trim() != constants.identity {
        return Err(GlvExecutionError::InvalidIdentity {
            identity: constants.identity.clone(),
        });
    }
    let expected = [ABUNDANCE_FIELD, SPACE_FIELD, TOTAL_FIELD];
    let actual = schema
        .field_schemas()
        .iter()
        .map(|field| field.name())
        .collect::<Vec<_>>();
    if actual != expected {
        return Err(GlvExecutionError::InvalidSchema {
            expected: expected.to_vec(),
            actual: actual.into_iter().map(str::to_owned).collect(),
        });
    }
    constants.inputs.validate()?;
    let species = constants.inputs.interaction().descriptor().species();
    constants.model.validate(
        species,
        constants.inputs.initial_state().descriptor().lattice(),
    )?;
    if constants.model.stochastic()
        && matches!(constants.observation, ObservationConfig::Detect { .. })
    {
        return Err(GlvExecutionError::StochasticDetection);
    }
    Ok(())
}

fn resolve_inputs(inputs: EcologicalInputs) -> Result<ResolvedEcologicalInputs, GlvExecutionError> {
    Ok(inputs.resolve()?)
}

fn build_member(
    schema: &SystemStateSchema,
    identity: &str,
    context: &InitializationContext,
    model: GlvModelConfig,
    inputs: ResolvedEcologicalInputs,
) -> UnitResult<Box<dyn RuntimeSimulation>> {
    let (interaction, initial) = inputs.into_parts();
    let species = interaction.species();
    debug_assert_eq!(species, initial.num_taxa());
    Ok(match model {
        GlvModelConfig::MeanFieldReplicator {
            growth,
            extinction_cutoff,
            time_step,
        } => Box::new(MeanFieldReplicator::new_with_schema(
            schema,
            Tensor::from_vec(&[species], initial.frequencies()),
            interaction,
            MeanFieldReplicatorConfig::new(
                growth.tensor(species, "growth")?,
                extinction_cutoff,
                TimeStep::new(time_step)?,
            ),
        )?),
        GlvModelConfig::MeanFieldReplicatorDemographic {
            growth,
            extinction_cutoff,
            time_step,
            sigma,
            rng,
        } => {
            let rng = if rng.seed().is_some() {
                rng
            } else {
                RngConfig::new(
                    Some(context.member_seed(identity, NOISE_SEED_PURPOSE)?),
                    rng.method(),
                )
            };
            let abundance = Tensor::from_vec(&[species], initial.frequencies());
            let state = assemble_state(schema, abundance, None, 1.0)?;
            let kernel = Kernel::new(
                KernelCore::new(interaction),
                MeanFieldReplicatorRk4::new(growth.tensor(species, "growth")?)?,
            );
            let noise = Noise::new(DemographicGaussian::new(
                sigma,
                rng,
                NoiseDomain::aggregate(species)?,
            )?);
            let invariant = FrequencyInvariant::new(species, extinction_cutoff)?;
            Box::new(MeanFieldReplicator::from_plugins(
                state,
                kernel,
                noise,
                invariant,
                TimeStep::new(time_step)?,
            )?)
        }
        GlvModelConfig::SpatialReplicator {
            growth,
            diffusion,
            extinction_cutoff,
            time_step,
        } => {
            let space = categorical_to_species_field(&initial, 1.0)?;
            let diffusion = crate::kernel::Diffusion::new(
                diffusion.tensor(species, "diffusion")?,
                initial.space().config().clone(),
            )?;
            Box::new(SpatialReplicator::new_with_schema(
                schema,
                space,
                interaction,
                SpatialReplicatorConfig::new(
                    growth.tensor(species, "growth")?,
                    diffusion,
                    extinction_cutoff,
                    TimeStep::new(time_step)?,
                ),
            )?)
        }
        GlvModelConfig::SpatialGeneralLotkaVolterra {
            growth,
            diffusion,
            extinction_cutoff,
            carrying_capacity,
            initial_population_per_site,
            time_step,
        } => {
            let space = categorical_to_species_field(&initial, initial_population_per_site)?;
            let diffusion = crate::kernel::Diffusion::new(
                diffusion.tensor(species, "diffusion")?,
                initial.space().config().clone(),
            )?;
            Box::new(SpatialGeneralLotkaVolterra::new_with_schema(
                schema,
                space,
                interaction,
                SpatialGeneralLotkaVolterraConfig::new(
                    growth.tensor(species, "growth")?,
                    diffusion,
                    extinction_cutoff,
                    carrying_capacity,
                    TimeStep::new(time_step)?,
                ),
            )?)
        }
    })
}

fn trajectory_observation<'a>(
    state: &'a SystemState,
    equilibrium_evidence: EquilibriumEvidence<'a>,
) -> Result<TrajectoryObservation<'a>, GlvExecutionError> {
    let abundance = state.payload::<crate::AggregateAbundance>(ABUNDANCE_FIELD)?;
    Ok(TrajectoryObservation {
        iteration: state.time().iteration(),
        physical_time: state.time().physical_time(),
        abundance: AbundanceView::Continuous(abundance.as_slice()),
        detector_observable: None,
        equilibrium_evidence,
    })
}

/// Configuration or construction failure at GLV's Workflow boundary.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GlvExecutionError {
    #[error("GLV identity `{identity}` is empty or has surrounding whitespace")]
    InvalidIdentity { identity: String },
    #[error("GLV schema fields {actual:?} do not match required order {expected:?}")]
    InvalidSchema {
        expected: Vec<&'static str>,
        actual: Vec<String>,
    },
    #[error("{field} has {actual} values, expected {expected}")]
    SpeciesParameterLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{field}[{index}] is not finite: {value}")]
    NonFiniteSpeciesParameter {
        field: &'static str,
        index: usize,
        value: f64,
    },
    #[error("{field} must be finite and nonnegative, found {value}")]
    InvalidNonnegative { field: &'static str, value: f64 },
    #[error("{field} must be finite and positive, found {value}")]
    InvalidPositive { field: &'static str, value: f64 },
    #[error("stochastic GLV requires `observation.mode = terminal_only`")]
    StochasticDetection,
    #[error("deterministic equilibrium detection requires a model residual")]
    MissingResidual,
    #[error(transparent)]
    Inputs(#[from] EcologicalInputsError),
    #[error(transparent)]
    TimeStep(#[from] crate::TimeStepError),
    #[error(transparent)]
    Kernel(#[from] crate::kernel::KernelAlgorithmError),
    #[error(transparent)]
    State(#[from] scientific_workflow::prelude::StateError),
    #[error(transparent)]
    Observer(#[from] TrajectoryObserverError),
}
