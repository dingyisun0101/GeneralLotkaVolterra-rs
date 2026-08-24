//! Built-in GLV task templates for application-owned Studys.

use std::error::Error;
use std::fs;
use std::path::PathBuf;

use ndarray::Array1;
use physics_in_parallel::prelude::basic::{RngConfig, SquareLatticeConfig};
use scientific_workflow::prelude::basics::{
    ExecutionScope, RngRecord, SamplingInterval, SimulationTime, StateStreamConfig, SystemState,
};
use scientific_workflow::prelude::study::TaskContext;
use serde::{Deserialize, Serialize};

use crate::study_inputs::GlvConfiguration;

use crate::initialization::{
    ResolvedSpatialInitialState, categorical_to_species_field, resolve_spatial_initial_state,
};
use crate::interaction::{
    InteractionArtifactDescriptor, InteractionMatrix, persist_interaction_matrix,
};
use crate::invariant::FrequencyInvariant;
use crate::kernel::{Diffusion, Kernel, KernelAlgorithm, KernelCore, MeanFieldReplicatorRk4};
use crate::noise::{DemographicGaussian, Noise, NoiseAlgorithm, NoiseDomain};
use crate::reading::verify_completed_glv_checkpoint;
use crate::recording::{GlvRecording, GlvRecordingMetadata, TerminationReason};
use crate::simulation::assemble_initial_state;
use crate::simulation::{
    MeanFieldReplicator, MeanFieldReplicatorConfig, SimulationKind, SpatialGeneralLotkaVolterra,
    SpatialGeneralLotkaVolterraConfig, SpatialReplicator, SpatialReplicatorConfig,
};
use crate::{AbundanceRepresentation, TimeStep};
use ecological_model_core::initial_state::InitialStateSource;
use ecological_model_core::terminal_state::{StopReason, TerminationSignal};
use ecological_model_core::trajectory::{
    AbundanceView, DetectionPolicy, EquilibriumEvidence, EquilibriumPolicy, PeriodicOrbitPolicy,
    ResidualTolerance, TerminalPolicy, TrajectoryObservation, TrajectoryObservationPolicy,
    TrajectoryObserver,
};

pub const INTERACTION_INPUT_KEY: &str = "/interaction";

/// Task-specific interaction input resolved through the workload path table.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionInput {
    pub path_key: String,
}

/// Inline or path-resolved aggregate frequencies for a mean-field initial state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum InitialAbundanceInput {
    Inline(Vec<f64>),
    Path { path_key: String },
}

impl InitialAbundanceInput {
    fn resolve(self, task: &GlvConfiguration) -> Result<Vec<f64>, TemplateTaskError> {
        match self {
            Self::Inline(values) => Ok(values),
            Self::Path { path_key } => {
                if path_key.trim().is_empty() {
                    return Err("initial-abundance path key must not be empty".into());
                }
                let path = task.resolve_path(&path_key)?;
                Ok(serde_json::from_slice(&fs::read(path)?)?)
            }
        }
    }
}

impl InteractionInput {
    fn resolve(&self, task: &GlvConfiguration) -> Result<PathBuf, TemplateTaskError> {
        if self.path_key.trim().is_empty() {
            return Err("interaction path key must not be empty".into());
        }
        Ok(task.resolve_path(&self.path_key)?)
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum ObservationConfig {
    Disabled,
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

/// Error returned by one advanced template task implementation.
pub type TemplateTaskError = Box<dyn Error + Send + Sync + 'static>;

/// Complete built-in GLV inputs compositions.
///
/// A variant fixes model assembly only. Scientific values, recording streams,
/// path aliases, and parameter sweeps remain in application-owned Workflow
/// inputs documents.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GlvTemplate {
    /// Deterministic non-spatial replicator dynamics.
    MeanFieldReplicator,
    /// Non-spatial replicator dynamics with demographic Gaussian noise.
    MeanFieldReplicatorDemographic,
    /// Deterministic spatial local-frequency replicator dynamics.
    SpatialReplicator,
    /// Deterministic spatial absolute-population GLV dynamics.
    SpatialGeneralLotkaVolterra,
}

impl GlvTemplate {
    /// Returns the stable template identity used in progress messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MeanFieldReplicator => "mean_field_replicator",
            Self::MeanFieldReplicatorDemographic => "mean_field_replicator_demographic",
            Self::SpatialReplicator => "spatial_replicator",
            Self::SpatialGeneralLotkaVolterra => "spatial_general_lotka_volterra",
        }
    }
}

impl GlvTemplate {
    /// Executes one study task using GLV's model and I/O capabilities.
    ///
    /// The application owns inputs loading, phase construction, scheduling,
    /// and the [`scientific_workflow::study::Study`]. GLV owns model
    /// construction, evolution, recording, terminal-state publication, and
    /// final checkpoint verification for this task.
    pub fn run_task(
        self,
        scope: &ExecutionScope,
        task: &GlvConfiguration,
        context: &TaskContext,
    ) -> Result<(), TemplateTaskError> {
        let maximum_iterations = task.decode_value("/maximum_iterations")?;
        context.set_iteration(0)?;
        context.set_target_iteration(maximum_iterations)?;
        match self {
            Self::MeanFieldReplicator => run_mean_field(scope, task, context, false),
            Self::MeanFieldReplicatorDemographic => run_mean_field(scope, task, context, true),
            Self::SpatialReplicator => run_spatial_replicator(scope, task, context),
            Self::SpatialGeneralLotkaVolterra => run_spatial_glv(scope, task, context),
        }
    }
}

fn run_mean_field(
    scope: &ExecutionScope,
    task: &GlvConfiguration,
    context: &TaskContext,
    demographic: bool,
) -> Result<(), TemplateTaskError> {
    let cutoff = task.decode_value("/cutoff")?;
    let time_step = TimeStep::new(task.decode_value("/physical_time_increment")?)?;
    let maximum_iterations = task.decode_value("/maximum_iterations")?;
    let streams: Vec<StateStreamConfig> = task.decode_value("/recording")?;
    let noise = demographic
        .then(|| {
            Ok::<_, TemplateTaskError>((
                task.decode_value::<f64>("/sigma")?,
                task.decode_value::<RngConfig>("/rng")?,
            ))
        })
        .transpose()?;

    context.set_detail("resolving interaction matrix");
    let interaction_input: InteractionInput = task.decode_value(INTERACTION_INPUT_KEY)?;
    let interaction = InteractionMatrix::load_json(interaction_input.resolve(task)?)?;
    let persisted = persist_interaction_matrix(scope, &interaction)?;
    let species = interaction.species();
    let initial_abundance = task
        .value("/initial_abundance")
        .map(|_| task.decode_value::<InitialAbundanceInput>("/initial_abundance"))
        .transpose()?
        .map(|input| input.resolve(task).map(Array1::from_vec))
        .transpose()?
        .unwrap_or_else(|| Array1::from_elem(species, 1.0 / species as f64));
    let growth = decode_species_values(task, "/growth", species)?;

    context.set_detail("constructing simulation");
    if let Some((sigma, rng)) = noise {
        let initial_state = assemble_initial_state(initial_abundance, None, 1.0)?;
        let kernel = Kernel::new(
            KernelCore::new(interaction),
            MeanFieldReplicatorRk4::new(growth)?,
        );
        let noise = Noise::new(DemographicGaussian::new(
            sigma,
            rng,
            NoiseDomain::aggregate(species)?,
        )?);
        let invariant = FrequencyInvariant::new(species, cutoff)?;
        let simulation =
            MeanFieldReplicator::from_plugins(initial_state, kernel, noise, invariant, time_step)?;
        finish_task(
            scope,
            task,
            context,
            streams,
            maximum_iterations,
            persisted.descriptor(),
            None,
            simulation,
        )
    } else {
        let config = MeanFieldReplicatorConfig::new(growth, cutoff, time_step);
        let simulation = MeanFieldReplicator::new(initial_abundance, interaction, config)?;
        finish_task(
            scope,
            task,
            context,
            streams,
            maximum_iterations,
            persisted.descriptor(),
            None,
            simulation,
        )
    }
}

fn run_spatial_replicator(
    scope: &ExecutionScope,
    task: &GlvConfiguration,
    context: &TaskContext,
) -> Result<(), TemplateTaskError> {
    let spatial_shape: Vec<usize> = task.decode_value("/spatial_shape")?;
    let spacing = task
        .value("/spacing")
        .map(|_| task.decode_value::<Vec<f64>>("/spacing"))
        .transpose()?;
    let boundary = task.decode_value("/boundary")?;
    let cutoff = task.decode_value("/cutoff")?;
    let time_step = TimeStep::new(task.decode_value("/physical_time_increment")?)?;
    let maximum_iterations = task.decode_value("/maximum_iterations")?;
    let streams: Vec<StateStreamConfig> = task.decode_value("/recording")?;
    let initialization: InitialStateSource = task.decode_value("/initialization")?;

    context.set_detail("resolving interaction matrix");
    let interaction_input: InteractionInput = task.decode_value(INTERACTION_INPUT_KEY)?;
    let interaction = InteractionMatrix::load_json(interaction_input.resolve(task)?)?;
    let persisted = persist_interaction_matrix(scope, &interaction)?;
    let species = interaction.species();
    let growth = decode_species_values(task, "/growth", species)?;
    let diffusion_coefficients = decode_species_values(task, "/diffusion", species)?;

    context.set_detail("constructing simulation");
    let lattice = SquareLatticeConfig::try_new(&spatial_shape, boundary, spacing.as_deref())?;
    let resolved = resolve_spatial_initial_state(&initialization, scope, lattice.clone(), species)?;
    let initial_space = categorical_to_species_field(resolved.initial(), 1.0)?;
    let diffusion = Diffusion::new(diffusion_coefficients, lattice)?;
    let simulation = SpatialReplicator::new(
        initial_space,
        interaction,
        SpatialReplicatorConfig::new(growth, diffusion, cutoff, time_step),
    )?;
    finish_task(
        scope,
        task,
        context,
        streams,
        maximum_iterations,
        persisted.descriptor(),
        Some(&resolved),
        simulation,
    )
}

fn decode_species_values(
    task: &GlvConfiguration,
    name: &str,
    species: usize,
) -> Result<Array1<f64>, TemplateTaskError> {
    match task.value(name).and_then(serde_json::Value::as_f64) {
        Some(value) => Ok(Array1::from_elem(species, value)),
        None => Ok(Array1::from_vec(task.decode_value(name)?)),
    }
}

fn run_spatial_glv(
    scope: &ExecutionScope,
    task: &GlvConfiguration,
    context: &TaskContext,
) -> Result<(), TemplateTaskError> {
    let spatial_shape: Vec<usize> = task.decode_value("/spatial_shape")?;
    let spacing = task
        .value("/spacing")
        .map(|_| task.decode_value::<Vec<f64>>("/spacing"))
        .transpose()?;
    let boundary = task.decode_value("/boundary")?;
    let cutoff = task.decode_value("/cutoff")?;
    let carrying_capacity = task
        .value("/carrying_capacity")
        .map(|_| task.decode_value::<f64>("/carrying_capacity"))
        .transpose()?;
    let initial_population_per_site: f64 = task.decode_value("/initial_population_per_site")?;
    let time_step = TimeStep::new(task.decode_value("/physical_time_increment")?)?;
    let maximum_iterations = task.decode_value("/maximum_iterations")?;
    let streams: Vec<StateStreamConfig> = task.decode_value("/recording")?;
    let initialization: InitialStateSource = task.decode_value("/initialization")?;

    context.set_detail("resolving interaction matrix");
    let interaction_input: InteractionInput = task.decode_value(INTERACTION_INPUT_KEY)?;
    let interaction = InteractionMatrix::load_json(interaction_input.resolve(task)?)?;
    let persisted = persist_interaction_matrix(scope, &interaction)?;
    let species = interaction.species();
    let growth = decode_species_values(task, "/growth", species)?;
    let diffusion_coefficients = decode_species_values(task, "/diffusion", species)?;

    context.set_detail("constructing simulation");
    let lattice = SquareLatticeConfig::try_new(&spatial_shape, boundary, spacing.as_deref())?;
    let resolved = resolve_spatial_initial_state(&initialization, scope, lattice.clone(), species)?;
    let initial_space =
        categorical_to_species_field(resolved.initial(), initial_population_per_site)?;
    let diffusion = Diffusion::new(diffusion_coefficients, lattice)?;
    let simulation = SpatialGeneralLotkaVolterra::new(
        initial_space,
        interaction,
        SpatialGeneralLotkaVolterraConfig::new(
            growth,
            diffusion,
            cutoff,
            carrying_capacity,
            time_step,
        ),
    )?;
    finish_task(
        scope,
        task,
        context,
        streams,
        maximum_iterations,
        persisted.descriptor(),
        Some(&resolved),
        simulation,
    )
}

trait StandardTemplateSimulation {
    fn kind(&self) -> SimulationKind;
    fn abundance_representation(&self) -> AbundanceRepresentation;
    fn state(&self) -> &SystemState;
    fn rng_record(&self) -> Option<&RngRecord>;
    fn step_template(&mut self) -> Result<SimulationTime, TemplateTaskError>;
    fn maximum_scaled_residual(
        &mut self,
        tolerance: ResidualTolerance,
    ) -> Result<Option<f64>, TemplateTaskError>;
}

impl<A, N> StandardTemplateSimulation for MeanFieldReplicator<A, N>
where
    A: KernelAlgorithm,
    N: NoiseAlgorithm,
{
    fn kind(&self) -> SimulationKind {
        self.kind()
    }

    fn abundance_representation(&self) -> AbundanceRepresentation {
        self.abundance_representation()
    }

    fn state(&self) -> &SystemState {
        self.state()
    }

    fn rng_record(&self) -> Option<&RngRecord> {
        self.rng_record()
    }

    fn step_template(&mut self) -> Result<SimulationTime, TemplateTaskError> {
        Ok(self.step()?)
    }

    fn maximum_scaled_residual(
        &mut self,
        tolerance: ResidualTolerance,
    ) -> Result<Option<f64>, TemplateTaskError> {
        Ok(self.maximum_scaled_residual(tolerance.absolute, tolerance.relative)?)
    }
}

impl<A, N> StandardTemplateSimulation for SpatialReplicator<A, N>
where
    A: KernelAlgorithm,
    N: NoiseAlgorithm,
{
    fn kind(&self) -> SimulationKind {
        self.kind()
    }

    fn abundance_representation(&self) -> AbundanceRepresentation {
        self.abundance_representation()
    }

    fn state(&self) -> &SystemState {
        self.state()
    }

    fn rng_record(&self) -> Option<&RngRecord> {
        self.rng_record()
    }

    fn step_template(&mut self) -> Result<SimulationTime, TemplateTaskError> {
        Ok(self.step()?)
    }

    fn maximum_scaled_residual(
        &mut self,
        tolerance: ResidualTolerance,
    ) -> Result<Option<f64>, TemplateTaskError> {
        Ok(self.maximum_scaled_residual(tolerance.absolute, tolerance.relative)?)
    }
}

impl<A, N> StandardTemplateSimulation for SpatialGeneralLotkaVolterra<A, N>
where
    A: KernelAlgorithm,
    N: NoiseAlgorithm,
{
    fn kind(&self) -> SimulationKind {
        self.kind()
    }

    fn abundance_representation(&self) -> AbundanceRepresentation {
        self.abundance_representation()
    }

    fn state(&self) -> &SystemState {
        self.state()
    }

    fn rng_record(&self) -> Option<&RngRecord> {
        self.rng_record()
    }

    fn step_template(&mut self) -> Result<SimulationTime, TemplateTaskError> {
        Ok(self.step()?)
    }

    fn maximum_scaled_residual(
        &mut self,
        tolerance: ResidualTolerance,
    ) -> Result<Option<f64>, TemplateTaskError> {
        Ok(self.maximum_scaled_residual(tolerance.absolute, tolerance.relative)?)
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_task<S>(
    scope: &ExecutionScope,
    task: &GlvConfiguration,
    context: &TaskContext,
    streams: Vec<StateStreamConfig>,
    maximum_iterations: u64,
    interaction: &InteractionArtifactDescriptor,
    initial_state: Option<&ResolvedSpatialInitialState>,
    mut simulation: S,
) -> Result<(), TemplateTaskError>
where
    S: StandardTemplateSimulation,
{
    let observation_config: ObservationConfig = task
        .value("/observation")
        .map(|_| task.decode_value("/observation"))
        .transpose()?
        .unwrap_or_default();
    if matches!(observation_config, ObservationConfig::Detect { .. })
        && simulation.rng_record().is_some()
    {
        return Err(
            "stochastic GLV requires explicit terminal_only or disabled observation".into(),
        );
    }
    let interval = signal_interval(&streams)?;
    let mut observer =
        TrajectoryObserver::from_policy(observation_policy(observation_config, interval))?;
    let mut metadata = GlvRecordingMetadata::new(
        simulation.kind(),
        simulation.abundance_representation(),
        task.configuration(),
        interaction,
        simulation.rng_record(),
    )?;
    if let Some(initial_state) = initial_state {
        metadata = metadata.with_initial_state(
            initial_state.descriptor(),
            initial_state.initial().rng_record(),
        )?;
    }
    let recording_directory = task_recording_directory(scope, task)?;
    let mut recording =
        GlvRecording::start(&recording_directory, streams, metadata, simulation.state())?;

    context.set_detail("evolving");
    let mut termination_signal = observe_glv(&mut simulation, observer.as_mut())?;
    while termination_signal.is_none()
        && simulation.state().simulation_time().iteration() < maximum_iterations
    {
        if context.is_cancelled() {
            return Err("execution cancelled by Ctrl-C".into());
        }
        let time = simulation.step_template()?;
        recording.observe_state(simulation.state())?;
        context.set_iteration(time.iteration())?;
        termination_signal = observe_glv(&mut simulation, observer.as_mut())?;
    }

    let termination_reason = termination_signal
        .clone()
        .map(TerminationReason::from)
        .unwrap_or(TerminationReason::MaximumIterations);

    context.set_detail("validating recording");
    let terminal_state = if let Some(observer) = observer {
        Some(observer.finish(
            trajectory_observation(simulation.state(), EquilibriumEvidence::Unavailable)?,
            termination_signal.map_or(StopReason::MaximumIterations, StopReason::Detected),
        )?)
    } else {
        None
    };
    let scientific_termination =
        (!matches!(termination_reason, TerminationReason::MaximumIterations)).then(|| {
            (
                simulation.state().simulation_time().iteration(),
                termination_reason.as_str(),
            )
        });
    if let Some(terminal_state) = &terminal_state {
        recording.complete(simulation.state(), termination_reason, terminal_state)?;
        publish_terminal_state(&recording_directory, terminal_state)?;
    } else {
        recording.complete_without_terminal(simulation.state(), termination_reason)?;
    }
    verify_completed_glv_checkpoint(recording_directory, simulation.state())?;
    if let Some((iteration, reason)) = scientific_termination {
        context.set_target_iteration(iteration)?;
        context.report(format!(
            "scientific termination: {} at iteration {iteration}",
            reason
        ))?;
    }
    Ok(())
}

fn publish_terminal_state(
    recording_directory: &std::path::Path,
    terminal_state: &ecological_model_core::terminal_state::TerminalState,
) -> Result<(), TemplateTaskError> {
    let path = recording_directory.join("terminal-state.json");
    let temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(terminal_state)?)?;
    fs::rename(&temporary, &path)?;

    let published =
        ecological_model_core::terminal_state::TerminalState::from_json_bytes(&fs::read(&path)?)?;
    if published != *terminal_state {
        return Err("published terminal-state JSON failed round-trip validation".into());
    }
    Ok(())
}

fn task_recording_directory(
    scope: &ExecutionScope,
    task: &GlvConfiguration,
) -> Result<std::path::PathBuf, TemplateTaskError> {
    match task.value("/recording_name") {
        Some(_) => {
            let name = task.decode_value::<String>("/recording_name")?;
            Ok(scope.named_task_recording_directory(&name)?)
        }
        None => Ok(scope.task_recording_directory(task.ordinal())),
    }
}

fn observe_glv<S>(
    simulation: &mut S,
    observer: Option<&mut TrajectoryObserver>,
) -> Result<Option<TerminationSignal>, TemplateTaskError>
where
    S: StandardTemplateSimulation,
{
    let Some(observer) = observer else {
        return Ok(None);
    };
    let iteration = simulation.state().simulation_time().iteration();
    let evidence = if observer.requires_equilibrium_evidence(iteration) {
        let tolerance = ResidualTolerance {
            absolute: 1.0e-10,
            relative: 1.0e-8,
        };
        EquilibriumEvidence::MaximumScaledResidual {
            value: simulation
                .maximum_scaled_residual(tolerance)?
                .ok_or("deterministic GLV residual is unavailable")?,
        }
    } else {
        EquilibriumEvidence::Unavailable
    };
    Ok(observer.observe(trajectory_observation(simulation.state(), evidence)?)?)
}

fn trajectory_observation<'a>(
    state: &'a SystemState,
    evidence: EquilibriumEvidence<'a>,
) -> Result<TrajectoryObservation<'a>, TemplateTaskError> {
    let abundance = state.payload::<crate::AggregateAbundance>(crate::ABUNDANCE_FIELD)?;
    let values = abundance
        .as_slice()
        .ok_or("GLV aggregate abundance must be contiguous")?;
    Ok(TrajectoryObservation {
        iteration: state.simulation_time().iteration(),
        physical_time: state.simulation_time().physical_time(),
        abundance: AbundanceView::Continuous(values),
        detector_observable: None,
        equilibrium_evidence: evidence,
    })
}

fn signal_interval(streams: &[StateStreamConfig]) -> Result<u64, TemplateTaskError> {
    let stream = streams
        .iter()
        .find(|stream| stream.name() == crate::SIGNAL_STREAM)
        .ok_or("recording configuration lacks the canonical signal stream")?;
    let SamplingInterval::Iterations(interval) = stream.sampling_interval();
    Ok(interval.get())
}

fn observation_policy(config: ObservationConfig, interval: u64) -> TrajectoryObservationPolicy {
    let terminal = TerminalPolicy {
        sample_interval_iterations: interval,
        trailing_window_samples: 128,
    };
    match config {
        ObservationConfig::Disabled => TrajectoryObservationPolicy::Disabled,
        ObservationConfig::TerminalOnly => TrajectoryObservationPolicy::TerminalOnly(terminal),
        ObservationConfig::Detect {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::{TERMINATION_DIAGNOSTICS_METADATA_KEY, TERMINATION_REASON_METADATA_KEY};
    use scientific_workflow::prelude::study::{Phase, Study, Task};
    use serde_json::Value;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn run_study(template: GlvTemplate, root: &Path) -> ExecutionScope {
        let inputs = crate::load_glv_inputs(root).unwrap();
        let scope =
            ExecutionScope::create_generated(inputs.resolve_path("recordings").unwrap()).unwrap();
        let task_scope = scope.clone();
        let tasks = inputs.combinations().map(|configuration| {
            let ordinal = configuration.ordinal();
            let scope = task_scope.clone();
            Task::progress(
                format!("{}-{ordinal}", template.as_str()),
                format!("{} {ordinal}", template.as_str()),
                move |context| Ok(template.run_task(&scope, &configuration, context)?),
            )
        });
        let phase = Phase::builder(1, "GLV simulation")
            .tasks(tasks)
            .build()
            .unwrap();
        Study::builder(root.join("study-record.json"))
            .phase(phase)
            .hidden()
            .build()
            .unwrap()
            .run_phases([1])
            .unwrap();
        scope
    }

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    static RUN_LOCK: Mutex<()> = Mutex::new(());

    struct TestStudy(PathBuf);

    impl TestStudy {
        fn stationary() -> Self {
            let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir()
                .join(format!("glv-termination-{}-{sequence}", std::process::id()));
            fs::create_dir_all(root.join("config")).unwrap();
            fs::create_dir(root.join("inputs")).unwrap();
            fs::write(
                root.join("config/fixed.json"),
                r#"{
                    "initial_abundance": [1.0, 0.0],
                    "growth": [0.0, 0.0],
                    "cutoff": 0.0,
                    "physical_time_increment": 0.1,
                    "maximum_iterations": 1200,
                    "interaction": {"path_key": "interaction_matrix"},
                    "observation": {
                        "mode": "detect",
                        "equilibrium": true,
                        "periodic_orbit": false
                    },
                    "recording": [
                        {"name":"signal","sampling_interval":10,"fields":["abundance","total"],"storage":{"layout":{"kind":"chunked","target_bytes":65536},"storage_queue_bytes":262144}},
                        {"name":"space","sampling_interval":10,"fields":["abundance","space","total"],"storage":{"layout":{"kind":"chunked","target_bytes":65536},"storage_queue_bytes":262144}},
                        {"name":"checkpoint","sampling_interval":10,"fields":["abundance","space","total"],"storage":{"layout":{"kind":"individual_files"},"storage_queue_bytes":262144}}
                    ]
                }"#,
            )
            .unwrap();
            fs::write(
                root.join("config/sweep.json"),
                r#"{"mode":"cartesian","axes":{}}"#,
            )
            .unwrap();
            fs::write(
                root.join("config/paths.json"),
                r#"{"interaction_matrix":"inputs/interaction.json","recordings":"output"}"#,
            )
            .unwrap();
            fs::write(
                root.join("inputs/interaction.json"),
                r#"{"kind":"matrix","version":1,"scalar":"f64","shape":[2,2],"data":[0.0,0.0,0.0,0.0]}"#,
            )
            .unwrap();
            Self(root)
        }

        fn capped() -> Self {
            let inputs = Self::stationary();
            let path = inputs.0.join("config/fixed.json");
            let mut config: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            config.as_object_mut().unwrap().remove("observation");
            config["maximum_iterations"] = Value::from(3);
            fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
            inputs
        }

        fn inferred_uniform() -> Self {
            let inputs = Self::stationary();
            let path = inputs.0.join("config/fixed.json");
            let mut config: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            config.as_object_mut().unwrap().remove("initial_abundance");
            config["growth"] = Value::from(0.0);
            fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
            inputs
        }

        fn path_abundance() -> Self {
            let inputs = Self::stationary();
            let fixed_path = inputs.0.join("config/fixed.json");
            let mut config: Value =
                serde_json::from_slice(&fs::read(&fixed_path).unwrap()).unwrap();
            config["initial_abundance"] = serde_json::json!({"path_key":"initial_abundance"});
            fs::write(fixed_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
            let paths_path = inputs.0.join("config/paths.json");
            let mut paths: Value = serde_json::from_slice(&fs::read(&paths_path).unwrap()).unwrap();
            paths["initial_abundance"] = Value::from("inputs/initial-abundance.json");
            fs::write(paths_path, serde_json::to_vec_pretty(&paths).unwrap()).unwrap();
            fs::write(
                inputs.0.join("inputs/initial-abundance.json"),
                b"[0.25,0.75]",
            )
            .unwrap();
            inputs
        }

        fn collapses_at_cap() -> Self {
            let inputs = Self::stationary();
            let path = inputs.0.join("config/fixed.json");
            let mut config: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            config["initial_abundance"] = serde_json::json!([0.98, 0.02]);
            config["growth"] = serde_json::json!([0.0, -100.0]);
            config["cutoff"] = Value::from(0.01);
            config["physical_time_increment"] = Value::from(0.01);
            config["maximum_iterations"] = Value::from(1);
            fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
            inputs
        }

        fn nonstationary_population_monoculture() -> Self {
            let inputs = Self::stationary();
            let path = inputs.0.join("config/fixed.json");
            let config = serde_json::json!({
                "spatial_shape": [2, 2],
                "initialization": {
                    "source": "recipe",
                    "recipe": {
                        "method": "random",
                        "distribution": {"kind": "inline", "weights": [1.0, 0.0]},
                        "rng": {"seed": 7}
                    }
                },
                "initial_population_per_site": 1.0,
                "growth": [0.35, 0.0],
                "diffusion": 0.0,
                "boundary": "neumann",
                "cutoff": 0.0,
                "physical_time_increment": 0.01,
                "maximum_iterations": 0,
                "interaction": {"path_key": "interaction_matrix"},
                "observation": {"mode":"detect","equilibrium":true,"periodic_orbit":false},
                "recording": [
                    {"name":"signal","sampling_interval":10,"fields":["abundance","total"],"storage":{"layout":{"kind":"chunked","target_bytes":65536},"storage_queue_bytes":262144}},
                    {"name":"space","sampling_interval":10,"fields":["abundance","space","total"],"storage":{"layout":{"kind":"chunked","target_bytes":65536},"storage_queue_bytes":262144}},
                    {"name":"checkpoint","sampling_interval":10,"fields":["abundance","space","total"],"storage":{"layout":{"kind":"individual_files"},"storage_queue_bytes":262144}}
                ]
            });
            fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
            inputs
        }
    }

    impl Drop for TestStudy {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn built_in_template_names_are_stable() {
        assert_eq!(
            GlvTemplate::MeanFieldReplicator.as_str(),
            "mean_field_replicator"
        );
        assert_eq!(
            GlvTemplate::MeanFieldReplicatorDemographic.as_str(),
            "mean_field_replicator_demographic"
        );
    }

    #[test]
    fn mean_field_accepts_path_resolved_initial_abundance() {
        let _guard = RUN_LOCK.lock().unwrap();
        let inputs = TestStudy::path_abundance();
        let scope = run_study(GlvTemplate::MeanFieldReplicator, &inputs.0);
        let fixed_point =
            crate::open_accepted_fixed_point(scope.task_recording_directory(0)).unwrap();
        assert_eq!(fixed_point.composition(), [0.25, 0.75]);
    }

    #[test]
    fn task_stops_at_a_confirmed_fixed_point_and_records_evidence() {
        let _guard = RUN_LOCK.lock().unwrap();
        let inputs = TestStudy::stationary();
        let scope = run_study(GlvTemplate::MeanFieldReplicator, &inputs.0);
        let document: Value = serde_json::from_slice(
            &fs::read(scope.task_recording_directory(0).join("metadata.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            document["terminal_metadata"][TERMINATION_REASON_METADATA_KEY],
            "equilibrium"
        );
        assert_eq!(
            document["terminal_metadata"][TERMINATION_DIAGNOSTICS_METADATA_KEY]["completed_windows"],
            3
        );
        let fixed_point =
            crate::open_accepted_fixed_point(scope.task_recording_directory(0)).unwrap();
        assert!(fixed_point.iteration() < 1200);
        assert_eq!(fixed_point.composition(), [1.0, 0.0]);
        let encoded = fixed_point.to_json_bytes().unwrap();
        assert_eq!(
            crate::AcceptedFixedPoint::from_json_bytes(&encoded).unwrap(),
            fixed_point
        );
        let terminal = crate::open_terminal_state(scope.task_recording_directory(0)).unwrap();
        assert_eq!(
            terminal.classification(),
            ecological_model_core::terminal_state::TerminalClassification::Equilibrium
        );
        assert_eq!(terminal.composition(), [1.0, 0.0]);
        assert_eq!(terminal.sample_count(), 1);
        let exported = crate::TerminalState::from_json_bytes(
            &fs::read(
                scope
                    .task_recording_directory(0)
                    .join("terminal-state.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(exported, terminal);
    }

    #[test]
    fn mean_field_infers_species_and_uniform_initial_state_from_interaction() {
        let _guard = RUN_LOCK.lock().unwrap();
        let inputs = TestStudy::inferred_uniform();
        let scope = run_study(GlvTemplate::MeanFieldReplicator, &inputs.0);
        let terminal = crate::open_terminal_state(scope.task_recording_directory(0)).unwrap();
        assert_eq!(terminal.composition(), [0.5, 0.5]);
    }

    #[test]
    fn task_accepts_a_collapse_on_the_last_allowed_step() {
        let _guard = RUN_LOCK.lock().unwrap();
        let inputs = TestStudy::collapses_at_cap();
        let scope = run_study(GlvTemplate::MeanFieldReplicator, &inputs.0);
        let terminal = crate::open_terminal_state(scope.task_recording_directory(0)).unwrap();
        assert_eq!(terminal.iteration(), 1);
        assert_eq!(terminal.composition(), [0.99, 0.01]);
    }

    #[test]
    fn population_monoculture_with_nonzero_residual_is_not_a_fixed_point() {
        let _guard = RUN_LOCK.lock().unwrap();
        let inputs = TestStudy::nonstationary_population_monoculture();
        let scope = run_study(GlvTemplate::SpatialGeneralLotkaVolterra, &inputs.0);
        let terminal = crate::open_terminal_state(scope.task_recording_directory(0)).unwrap();
        assert_eq!(
            terminal.classification(),
            ecological_model_core::terminal_state::TerminalClassification::TrailingAverage
        );
        assert_eq!(
            terminal.stop_reason(),
            &ecological_model_core::terminal_state::StopReason::MaximumIterations
        );
        assert_eq!(terminal.iteration(), 0);
    }

    #[test]
    fn capped_task_publishes_a_trailing_average_with_an_explicit_marker() {
        let _guard = RUN_LOCK.lock().unwrap();
        let inputs = TestStudy::capped();
        let scope = run_study(GlvTemplate::MeanFieldReplicator, &inputs.0);
        let terminal = crate::open_terminal_state(scope.task_recording_directory(0)).unwrap();
        assert_eq!(
            terminal.classification(),
            ecological_model_core::terminal_state::TerminalClassification::TrailingAverage
        );
        assert_eq!(
            terminal.stop_reason(),
            &ecological_model_core::terminal_state::StopReason::MaximumIterations
        );
        assert_eq!(terminal.composition(), [1.0, 0.0]);
        assert_eq!(terminal.sample_count(), 2);
        assert_eq!(terminal.first_sample_iteration(), 0);
        assert_eq!(terminal.last_sample_iteration(), 3);
        let exported = crate::TerminalState::from_json_bytes(
            &fs::read(
                scope
                    .task_recording_directory(0)
                    .join("terminal-state.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(exported, terminal);
        assert!(matches!(
            crate::open_accepted_fixed_point(scope.task_recording_directory(0)),
            Err(crate::AcceptedFixedPointError::NotAcceptedFixedPoint { .. })
        ));
    }
}
