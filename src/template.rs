//! Project templates and the single ordinary-user execution entry point.

use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use ndarray::Array1;
use physics_in_parallel::rng::RngConfig;
use physics_in_parallel::space::discrete::square_lattice::SquareLatticeConfig;
use scientific_workflow::configuration::{ConfigurationError, TaskConfig};
use scientific_workflow::execution::{ExecutionScope, ExecutionScopeError};
use scientific_workflow::reporting::{ProgressReporter, ReportingError, TaskProgress};
use scientific_workflow::rng_record::RngRecord;
use scientific_workflow::storage::StateStreamConfig;
use scientific_workflow::system_state::{SimulationTime, SystemState};
use serde::{Deserialize, Serialize};
use thiserror::Error as ThisError;

use crate::initialization::{
    ResolvedSpatialInitialState, SpatialInitialStateSource, categorical_to_species_field,
};
use crate::interaction::{
    InteractionArtifactDescriptor, InteractionMatrix, persist_interaction_matrix,
};
use crate::invariant::FrequencyInvariant;
use crate::kernel::{Diffusion, Kernel, KernelAlgorithm, KernelCore, MeanFieldReplicatorRk4};
use crate::noise::{DemographicGaussian, Noise, NoiseAlgorithm, NoiseDomain};
use crate::project::{GlvProjectError, load_glv_project};
use crate::reading::verify_completed_glv_checkpoint;
use crate::recording::{GlvRecording, GlvRecordingMetadata, TerminationReason};
use crate::simulation::{
    MeanFieldReplicator, MeanFieldReplicatorConfig, SimulationKind, SpatialGeneralLotkaVolterra,
    SpatialGeneralLotkaVolterraConfig, SpatialReplicator, SpatialReplicatorConfig,
};
use crate::terminal_state::{TerminalStateMonitor, TerminalStatePolicy};
use crate::termination::{ResidualTolerance, TerminationMonitor, TerminationPolicy};
use crate::{AbundanceRepresentation, TimeStep};

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct AutomaticTerminationConfig {
    #[serde(default)]
    fixed_point: bool,
    #[serde(default)]
    oscillation: bool,
}

/// Error returned by one advanced template task implementation.
pub type TemplateTaskError = Box<dyn Error + Send + Sync + 'static>;

/// Complete built-in GLV project compositions.
///
/// A variant fixes model assembly only. Scientific values, recording streams,
/// path aliases, and parameter sweeps remain in the Workflow project documents
/// found beneath the configuration folder passed to [`run`].
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

/// Advanced contract for a complete GLV project composition.
///
/// [`run`] retains all generic orchestration. Implementors receive Workflow's
/// own scope, reporter, and task values directly and should assemble only their
/// scientific model plus its thin recording adapter. Each task must finish its
/// progress handle successfully; otherwise the outer reporter rejects project
/// completion.
pub trait GlvProjectTemplate {
    /// Stable human-readable template identity.
    fn name(&self) -> &str;

    /// Runs one complete Workflow task inside the already-created execution.
    fn run_task(
        &mut self,
        scope: &ExecutionScope,
        reporter: &ProgressReporter,
        task: TaskConfig,
    ) -> Result<(), TemplateTaskError>;
}

impl GlvProjectTemplate for GlvTemplate {
    fn name(&self) -> &str {
        self.as_str()
    }

    fn run_task(
        &mut self,
        scope: &ExecutionScope,
        reporter: &ProgressReporter,
        task: TaskConfig,
    ) -> Result<(), TemplateTaskError> {
        let maximum_iterations = task.decode_value("maximum_iterations")?;
        let progress = reporter.start_task(&task, 0, Some(maximum_iterations))?;
        self.run_task_with_progress(scope, task, progress)
    }
}

impl GlvTemplate {
    fn run_task_with_progress(
        &mut self,
        scope: &ExecutionScope,
        task: TaskConfig,
        progress: TaskProgress,
    ) -> Result<(), TemplateTaskError> {
        match self {
            Self::MeanFieldReplicator => run_mean_field(scope, task, progress, false),
            Self::MeanFieldReplicatorDemographic => run_mean_field(scope, task, progress, true),
            Self::SpatialReplicator => run_spatial_replicator(scope, task, progress),
            Self::SpatialGeneralLotkaVolterra => run_spatial_glv(scope, task, progress),
        }
    }
}

/// Failure while executing a complete GLV project.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum GlvRunError {
    /// The supplied path was not the conventional `config` directory.
    #[error("GLV configuration folder must be named `config`, got `{path}`")]
    ConfigurationFolder { path: PathBuf },
    /// Workflow could not load or GLV could not validate the project.
    #[error(transparent)]
    Project(#[from] GlvProjectError),
    /// Workflow could not resolve the configured recording root.
    #[error(transparent)]
    Configuration(#[from] ConfigurationError),
    /// Workflow could not create the execution directory.
    #[error(transparent)]
    ExecutionScope(#[from] ExecutionScopeError),
    /// Workflow progress reporting failed.
    #[error(transparent)]
    Reporting(#[from] ReportingError),
    /// An embedding reporter can drive only a single task per GLV project.
    #[error("external progress requires exactly one GLV task, found {actual}")]
    ExternalProgressTaskCount { actual: u64 },
    /// One template task failed.
    #[error("template `{template}` task {task_ordinal} failed: {source}")]
    Task {
        /// Stable template identity.
        template: String,
        /// Workflow task ordinal.
        task_ordinal: u64,
        /// Scientific or recording failure returned by the template.
        #[source]
        source: TemplateTaskError,
    },
}

/// Runs every Workflow task for one built-in or advanced GLV template.
///
/// `configuration_folder` must be the conventional `config` directory. Its
/// parent is the Workflow project root against which `paths.json` entries are
/// resolved. The returned Workflow scope identifies the generated output.
pub fn run(
    mut template: impl GlvProjectTemplate,
    configuration_folder: impl AsRef<Path>,
) -> Result<ExecutionScope, GlvRunError> {
    let configuration_folder = configuration_folder.as_ref();
    let project_root = project_root(configuration_folder)?;
    let project = load_glv_project(project_root)?;
    let scope = ExecutionScope::create_generated(project.resolve_path("recordings")?)?;
    let reporter = ProgressReporter::for_project(&project).start()?;
    reporter.report(format!(
        "running template {} in {}",
        template.name(),
        scope.directory().display()
    ))?;

    for task in project.task_configs() {
        let task_ordinal = task.task_ordinal();
        if let Err(source) = template.run_task(&scope, &reporter, task) {
            let message = format!(
                "template {} task {task_ordinal} failed: {source}",
                template.name()
            );
            let _ = reporter.fail(message);
            return Err(GlvRunError::Task {
                template: template.name().to_owned(),
                task_ordinal,
                source,
            });
        }
    }

    reporter.complete(format!(
        "template {} complete: {}",
        template.name(),
        scope.directory().display()
    ))?;
    Ok(scope)
}

/// Runs one built-in, single-task GLV project with a progress handle supplied
/// by an embedding application. The caller owns the parent reporter.
pub fn run_with_progress(
    mut template: GlvTemplate,
    configuration_folder: impl AsRef<Path>,
    progress: TaskProgress,
) -> Result<ExecutionScope, GlvRunError> {
    let configuration_folder = configuration_folder.as_ref();
    let project_root = project_root(configuration_folder)?;
    let project = load_glv_project(project_root)?;
    let actual = project.task_count();
    if actual != 1 {
        return Err(GlvRunError::ExternalProgressTaskCount { actual });
    }
    let scope = ExecutionScope::create_generated(project.resolve_path("recordings")?)?;
    let task = project
        .task_configs()
        .next()
        .expect("a validated one-task project has one task");
    let task_ordinal = task.task_ordinal();
    template
        .run_task_with_progress(&scope, task, progress)
        .map_err(|source| GlvRunError::Task {
            template: template.name().to_owned(),
            task_ordinal,
            source,
        })?;
    Ok(scope)
}

fn project_root(configuration_folder: &Path) -> Result<&Path, GlvRunError> {
    if configuration_folder.file_name() != Some(OsStr::new("config")) {
        return Err(GlvRunError::ConfigurationFolder {
            path: configuration_folder.to_path_buf(),
        });
    }
    Ok(configuration_folder
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new(".")))
}

fn run_mean_field(
    scope: &ExecutionScope,
    task: TaskConfig,
    progress: TaskProgress,
    demographic: bool,
) -> Result<(), TemplateTaskError> {
    let initial_abundance = if task.value("K").is_some() {
        let species = task.decode_value::<usize>("K")?;
        if species == 0 {
            return Err("mean-field well-mixed species count `K` must be nonzero".into());
        }
        Array1::from_elem(species, 1.0 / species as f64)
    } else {
        Array1::from_vec(task.decode_value("initial_abundance")?)
    };
    let growth = match task.value("growth").and_then(serde_json::Value::as_f64) {
        Some(value) => Array1::from_elem(initial_abundance.len(), value),
        None => Array1::from_vec(task.decode_value("growth")?),
    };
    let cutoff = task.decode_value("cutoff")?;
    let time_step = TimeStep::new(task.decode_value("physical_time_increment")?)?;
    let maximum_iterations = task.decode_value("maximum_iterations")?;
    let streams: Vec<StateStreamConfig> = task.decode_value("recording")?;
    let noise = demographic
        .then(|| {
            Ok::<_, TemplateTaskError>((
                task.decode_value::<f64>("sigma")?,
                task.decode_value::<RngConfig>("rng")?,
            ))
        })
        .transpose()?;

    progress.set_phase("resolving interaction matrix");
    let species = initial_abundance.len();
    let interaction =
        InteractionMatrix::load_json(task.resolve_path("interaction_matrix")?, species)?;
    let persisted = persist_interaction_matrix(scope, &interaction)?;

    progress.set_phase("constructing simulation");
    let config = MeanFieldReplicatorConfig::new(growth.clone(), cutoff, time_step);
    if let Some((sigma, rng)) = noise {
        let initial_state =
            MeanFieldReplicator::new(initial_abundance, interaction.clone(), config)?.into_state();
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
        let simulation = MeanFieldReplicator::from_plugins(
            initial_state,
            AbundanceRepresentation::RelativeFrequency,
            kernel,
            noise,
            invariant,
            time_step,
        )?;
        finish_task(
            scope,
            task,
            progress,
            streams,
            maximum_iterations,
            persisted.descriptor(),
            None,
            simulation,
        )
    } else {
        let simulation = MeanFieldReplicator::new(initial_abundance, interaction, config)?;
        finish_task(
            scope,
            task,
            progress,
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
    task: TaskConfig,
    progress: TaskProgress,
) -> Result<(), TemplateTaskError> {
    let spatial_shape: Vec<usize> = task.decode_value("spatial_shape")?;
    let species = task
        .value("K")
        .map(|_| task.decode_value::<usize>("K"))
        .transpose()?;
    if species == Some(0) {
        return Err("spatial species count `K` must be nonzero".into());
    }
    let growth = decode_species_values(&task, "growth", species)?;
    let species = species.unwrap_or(growth.len());
    let diffusion_coefficients = decode_species_values(&task, "diffusion", Some(species))?;
    let spacing: Vec<f64> = task.decode_value("spacing")?;
    let boundary = task.decode_value("boundary")?;
    let cutoff = task.decode_value("cutoff")?;
    let time_step = TimeStep::new(task.decode_value("physical_time_increment")?)?;
    let maximum_iterations = task.decode_value("maximum_iterations")?;
    let streams: Vec<StateStreamConfig> = task.decode_value("recording")?;
    let initialization: SpatialInitialStateSource = task.decode_value("initialization")?;

    progress.set_phase("resolving interaction matrix");
    let interaction =
        InteractionMatrix::load_json(task.resolve_path("interaction_matrix")?, species)?;
    let persisted = persist_interaction_matrix(scope, &interaction)?;

    progress.set_phase("constructing simulation");
    let lattice = SquareLatticeConfig::try_new(&spatial_shape, boundary, Some(&spacing))?;
    let resolved = initialization.resolve(&task, scope, lattice.clone(), species)?;
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
        progress,
        streams,
        maximum_iterations,
        persisted.descriptor(),
        Some(&resolved),
        simulation,
    )
}

fn decode_species_values(
    task: &TaskConfig,
    name: &str,
    species: Option<usize>,
) -> Result<Array1<f64>, TemplateTaskError> {
    match (
        task.value(name).and_then(serde_json::Value::as_f64),
        species,
    ) {
        (Some(value), Some(species)) => Ok(Array1::from_elem(species, value)),
        (Some(_), None) => Err(format!("scalar `{name}` requires species count `K`").into()),
        (None, _) => Ok(Array1::from_vec(task.decode_value(name)?)),
    }
}

fn run_spatial_glv(
    scope: &ExecutionScope,
    task: TaskConfig,
    progress: TaskProgress,
) -> Result<(), TemplateTaskError> {
    let spatial_shape: Vec<usize> = task.decode_value("spatial_shape")?;
    let growth = Array1::from_vec(task.decode_value("growth")?);
    let diffusion_coefficients = Array1::from_vec(task.decode_value("diffusion")?);
    let spacing: Vec<f64> = task.decode_value("spacing")?;
    let boundary = task.decode_value("boundary")?;
    let cutoff = task.decode_value("cutoff")?;
    let carrying_capacity = task.decode_value("carrying_capacity")?;
    let initial_population_per_site: f64 = task.decode_value("initial_population_per_site")?;
    let time_step = TimeStep::new(task.decode_value("physical_time_increment")?)?;
    let maximum_iterations = task.decode_value("maximum_iterations")?;
    let streams: Vec<StateStreamConfig> = task.decode_value("recording")?;
    let initialization: SpatialInitialStateSource = task.decode_value("initialization")?;

    progress.set_phase("resolving interaction matrix");
    let species = growth.len();
    let interaction =
        InteractionMatrix::load_json(task.resolve_path("interaction_matrix")?, species)?;
    let persisted = persist_interaction_matrix(scope, &interaction)?;

    progress.set_phase("constructing simulation");
    let lattice = SquareLatticeConfig::try_new(&spatial_shape, boundary, Some(&spacing))?;
    let resolved = initialization.resolve(&task, scope, lattice.clone(), species)?;
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
        progress,
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
    task: TaskConfig,
    progress: TaskProgress,
    streams: Vec<StateStreamConfig>,
    maximum_iterations: u64,
    interaction: &InteractionArtifactDescriptor,
    initial_state: Option<&ResolvedSpatialInitialState>,
    mut simulation: S,
) -> Result<(), TemplateTaskError>
where
    S: StandardTemplateSimulation,
{
    let mut terminal_state_monitor = TerminalStateMonitor::new(TerminalStatePolicy::default())?;
    terminal_state_monitor.observe(simulation.state())?;
    let automatic_termination: AutomaticTerminationConfig = task
        .value("termination")
        .map(|_| task.decode_value("termination"))
        .transpose()?
        .unwrap_or_default();
    let termination_policy = TerminationPolicy::automatic(
        automatic_termination.fixed_point,
        automatic_termination.oscillation,
    );
    if termination_policy.is_some() && simulation.rng_record().is_some() {
        return Err(
            "termination monitoring is supported only for deterministic simulations".into(),
        );
    }
    let mut monitor = termination_policy
        .map(TerminationMonitor::new)
        .transpose()?;
    let mut metadata = GlvRecordingMetadata::new(
        simulation.kind(),
        simulation.abundance_representation(),
        task.parameters(),
        interaction,
        simulation.rng_record(),
    )?;
    if let Some(initial_state) = initial_state {
        metadata = metadata.with_initial_state(
            initial_state.descriptor(),
            initial_state.initial().rng_record(),
        )?;
    }
    let recording_directory = scope.task_recording_directory(task.task_ordinal());
    let mut recording =
        GlvRecording::start(&recording_directory, streams, metadata, simulation.state())?;

    progress.set_phase("evolving");
    let mut termination_reason = evaluate_automatic_termination(&mut simulation, monitor.as_mut())?
        .map(TerminationReason::from);
    while termination_reason.is_none()
        && simulation.state().simulation_time().iteration() < maximum_iterations
    {
        if progress.is_cancelled() {
            return Err("execution cancelled by Ctrl-C".into());
        }
        let time = simulation.step_template()?;
        recording.observe_state(simulation.state())?;
        terminal_state_monitor.observe(simulation.state())?;
        progress.set_iteration(time.iteration())?;
        termination_reason = evaluate_automatic_termination(&mut simulation, monitor.as_mut())?
            .map(TerminationReason::from);
    }

    let termination_reason = termination_reason.unwrap_or(TerminationReason::MaximumIterations);

    progress.set_phase("validating recording");
    let progress_completion_reason = match &termination_reason {
        TerminationReason::MaximumIterations => None,
        reason => Some(format!("scientific termination: {}", reason.as_str())),
    };
    let terminal_state = terminal_state_monitor.finish(simulation.state(), &termination_reason)?;
    recording.complete(simulation.state(), termination_reason, &terminal_state)?;
    publish_terminal_state(scope, task.task_ordinal(), &terminal_state)?;
    verify_completed_glv_checkpoint(recording_directory, simulation.state())?;
    progress.complete(progress_completion_reason)?;
    Ok(())
}

fn publish_terminal_state(
    scope: &ExecutionScope,
    task_ordinal: u64,
    terminal_state: &crate::TerminalState,
) -> Result<(), TemplateTaskError> {
    let path = scope
        .directory()
        .join(format!("task-{task_ordinal:06}-terminal-state.json"));
    let temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(terminal_state)?)?;
    fs::rename(&temporary, &path)?;

    let published = crate::TerminalState::from_json_bytes(&fs::read(&path)?)?;
    if published != *terminal_state {
        return Err("published terminal-state JSON failed round-trip validation".into());
    }
    Ok(())
}

fn evaluate_automatic_termination<S>(
    simulation: &mut S,
    monitor: Option<&mut TerminationMonitor>,
) -> Result<Option<crate::termination::ConvergenceReason>, TemplateTaskError>
where
    S: StandardTemplateSimulation,
{
    let Some(monitor) = monitor else {
        return Ok(None);
    };
    let iteration = simulation.state().simulation_time().iteration();
    let fixed_tolerance = monitor
        .policy()
        .fixed_point
        .as_ref()
        .map(|config| config.residual_tolerance);
    let absorbing_replicator = matches!(
        simulation.kind(),
        SimulationKind::MeanFieldReplicator | SimulationKind::SpatialReplicator
    ) && monitor.has_single_supported_species(simulation.state())?;

    if absorbing_replicator {
        let tolerance = fixed_tolerance.expect("single-support check requires fixed-point policy");
        let residual = simulation
            .maximum_scaled_residual(tolerance)?
            .ok_or(crate::termination::TerminationError::ResidualUnavailable)?;
        if let Some(reason) =
            monitor.evaluate_absorbing_fixed_point(simulation.state(), residual)?
        {
            return Ok(Some(reason));
        }
    }
    if !monitor.should_sample(iteration) {
        return Ok(None);
    }
    let residual = if let Some(tolerance) = fixed_tolerance {
        simulation
            .maximum_scaled_residual(tolerance)?
            .ok_or(crate::termination::TerminationError::ResidualUnavailable)?
    } else {
        0.0
    };
    Ok(monitor.observe(simulation.state(), residual)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::{
        COMPLETED_ITERATION_METADATA_KEY, TERMINATION_DIAGNOSTICS_METADATA_KEY,
        TERMINATION_REASON_METADATA_KEY,
    };
    use serde_json::Value;
    use std::fs;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    static RUN_LOCK: Mutex<()> = Mutex::new(());

    struct TestProject(PathBuf);

    impl TestProject {
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
                    "maximum_iterations": 0,
                    "termination": {
                        "fixed_point": true,
                        "oscillation": false
                    },
                    "recording": [
                        {"name":"signal","sampling_interval":10,"fields":["abundance","total"],"storage_limits":[65536,262144]},
                        {"name":"space","sampling_interval":10,"fields":["abundance","space","total"],"storage_limits":[65536,262144]},
                        {"name":"checkpoint","sampling_interval":10,"fields":["abundance","space","total"],"storage_limits":[65536,262144]}
                    ]
                }"#,
            )
            .unwrap();
            fs::write(
                root.join("config/sweep.json"),
                r#"{"mode":"cartesian","axes":[]}"#,
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
            let project = Self::stationary();
            let path = project.0.join("config/fixed.json");
            let mut config: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            config.as_object_mut().unwrap().remove("termination");
            config["maximum_iterations"] = Value::from(3);
            fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
            project
        }

        fn collapses_at_cap() -> Self {
            let project = Self::stationary();
            let path = project.0.join("config/fixed.json");
            let mut config: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            config["initial_abundance"] = serde_json::json!([0.98, 0.02]);
            config["growth"] = serde_json::json!([0.0, -100.0]);
            config["cutoff"] = Value::from(0.01);
            config["physical_time_increment"] = Value::from(0.01);
            config["maximum_iterations"] = Value::from(1);
            fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
            project
        }

        fn nonstationary_population_monoculture() -> Self {
            let project = Self::stationary();
            let path = project.0.join("config/fixed.json");
            let config = serde_json::json!({
                "spatial_shape": [2, 2],
                "initialization": {
                    "source": "config",
                    "config": {
                        "method": "random",
                        "distribution": {"kind": "inline", "weights": [1.0, 0.0]},
                        "rng": {"seed": 7}
                    }
                },
                "initial_population_per_site": 1.0,
                "growth": [0.35, 0.0],
                "diffusion": [0.0, 0.0],
                "spacing": [1.0, 1.0],
                "boundary": "neumann",
                "cutoff": 0.0,
                "carrying_capacity": 100.0,
                "physical_time_increment": 0.01,
                "maximum_iterations": 0,
                "termination": {"fixed_point": true, "oscillation": false},
                "recording": [
                    {"name":"signal","sampling_interval":10,"fields":["abundance","total"],"storage_limits":[65536,262144]},
                    {"name":"space","sampling_interval":10,"fields":["abundance","space","total"],"storage_limits":[65536,262144]},
                    {"name":"checkpoint","sampling_interval":10,"fields":["abundance","space","total"],"storage_limits":[65536,262144]}
                ]
            });
            fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
            project
        }
    }

    impl Drop for TestProject {
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
    fn entry_point_requires_the_configuration_folder() {
        let error = project_root(Path::new("project")).unwrap_err();
        assert!(matches!(error, GlvRunError::ConfigurationFolder { .. }));
        assert_eq!(
            project_root(Path::new("project/config")).unwrap(),
            Path::new("project")
        );
    }

    #[test]
    fn ordinary_runner_stops_at_a_confirmed_fixed_point_and_records_evidence() {
        let _guard = RUN_LOCK.lock().unwrap();
        let project = TestProject::stationary();
        let scope = run(GlvTemplate::MeanFieldReplicator, project.0.join("config")).unwrap();
        let document: Value = serde_json::from_slice(
            &fs::read(scope.task_recording_directory(0).join("metadata.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            document["terminal_metadata"][TERMINATION_REASON_METADATA_KEY],
            "fixed_point"
        );
        assert_eq!(
            document["terminal_metadata"][COMPLETED_ITERATION_METADATA_KEY],
            0
        );
        assert_eq!(
            document["terminal_metadata"][TERMINATION_DIAGNOSTICS_METADATA_KEY]["completed_windows"],
            1
        );
        let fixed_point =
            crate::open_accepted_fixed_point(scope.task_recording_directory(0)).unwrap();
        assert_eq!(fixed_point.iteration(), 0);
        assert_eq!(fixed_point.physical_time(), Some(0.0));
        assert_eq!(fixed_point.composition(), [1.0, 0.0]);
        let encoded = fixed_point.to_json_bytes().unwrap();
        assert_eq!(
            crate::AcceptedFixedPoint::from_json_bytes(&encoded).unwrap(),
            fixed_point
        );
        let terminal = crate::open_terminal_state(scope.task_recording_directory(0)).unwrap();
        assert!(terminal.is_accepted_fixed_point());
        assert_eq!(terminal.composition(), [1.0, 0.0]);
        assert_eq!(terminal.sample_count(), 1);
        let exported = crate::TerminalState::from_json_bytes(
            &fs::read(scope.directory().join("task-000000-terminal-state.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(exported, terminal);
    }

    #[test]
    fn runner_accepts_a_collapse_on_the_last_allowed_step() {
        let _guard = RUN_LOCK.lock().unwrap();
        let project = TestProject::collapses_at_cap();
        let scope = run(GlvTemplate::MeanFieldReplicator, project.0.join("config")).unwrap();
        let fixed_point =
            crate::open_accepted_fixed_point(scope.task_recording_directory(0)).unwrap();
        assert_eq!(fixed_point.iteration(), 1);
        assert_eq!(fixed_point.composition(), [1.0, 0.0]);
    }

    #[test]
    fn population_monoculture_with_nonzero_residual_is_not_a_fixed_point() {
        let _guard = RUN_LOCK.lock().unwrap();
        let project = TestProject::nonstationary_population_monoculture();
        let scope = run(
            GlvTemplate::SpatialGeneralLotkaVolterra,
            project.0.join("config"),
        )
        .unwrap();
        let terminal = crate::open_terminal_state(scope.task_recording_directory(0)).unwrap();
        assert!(!terminal.is_accepted_fixed_point());
        assert_eq!(terminal.termination_reason(), "maximum_iterations");
        assert_eq!(terminal.iteration(), 0);
    }

    #[test]
    fn capped_runner_publishes_a_trailing_average_with_an_explicit_marker() {
        let _guard = RUN_LOCK.lock().unwrap();
        let project = TestProject::capped();
        let scope = run(GlvTemplate::MeanFieldReplicator, project.0.join("config")).unwrap();
        let terminal = crate::open_terminal_state(scope.task_recording_directory(0)).unwrap();
        assert!(!terminal.is_accepted_fixed_point());
        assert_eq!(terminal.termination_reason(), "maximum_iterations");
        assert_eq!(terminal.composition(), [1.0, 0.0]);
        assert_eq!(terminal.sample_count(), 2);
        assert_eq!(terminal.first_sample_iteration(), 0);
        assert_eq!(terminal.last_sample_iteration(), 3);
        let exported = crate::TerminalState::from_json_bytes(
            &fs::read(scope.directory().join("task-000000-terminal-state.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(exported, terminal);
        assert!(matches!(
            crate::open_accepted_fixed_point(scope.task_recording_directory(0)),
            Err(crate::AcceptedFixedPointError::NotAcceptedFixedPoint { .. })
        ));
    }
}
