//! Project templates and the single ordinary-user execution entry point.

use std::error::Error;
use std::ffi::OsStr;
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
use crate::termination::{ResidualTolerance, TerminationMonitor, TerminationPolicy};
use crate::{AbundanceRepresentation, TimeStep};

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
        match self {
            Self::MeanFieldReplicator => run_mean_field(scope, reporter, task, false),
            Self::MeanFieldReplicatorDemographic => run_mean_field(scope, reporter, task, true),
            Self::SpatialReplicator => run_spatial_replicator(scope, reporter, task),
            Self::SpatialGeneralLotkaVolterra => run_spatial_glv(scope, reporter, task),
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
    reporter: &ProgressReporter,
    task: TaskConfig,
    demographic: bool,
) -> Result<(), TemplateTaskError> {
    let initial_abundance = Array1::from_vec(task.decode_value("initial_abundance")?);
    let growth = Array1::from_vec(task.decode_value("growth")?);
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

    let progress = reporter.start_task(&task, 0, Some(maximum_iterations))?;
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
    reporter: &ProgressReporter,
    task: TaskConfig,
) -> Result<(), TemplateTaskError> {
    let spatial_shape: Vec<usize> = task.decode_value("spatial_shape")?;
    let growth = Array1::from_vec(task.decode_value("growth")?);
    let diffusion_coefficients = Array1::from_vec(task.decode_value("diffusion")?);
    let spacing: Vec<f64> = task.decode_value("spacing")?;
    let boundary = task.decode_value("boundary")?;
    let cutoff = task.decode_value("cutoff")?;
    let time_step = TimeStep::new(task.decode_value("physical_time_increment")?)?;
    let maximum_iterations = task.decode_value("maximum_iterations")?;
    let streams: Vec<StateStreamConfig> = task.decode_value("recording")?;
    let initialization: SpatialInitialStateSource = task.decode_value("initialization")?;

    let progress = reporter.start_task(&task, 0, Some(maximum_iterations))?;
    progress.set_phase("resolving interaction matrix");
    let species = growth.len();
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

fn run_spatial_glv(
    scope: &ExecutionScope,
    reporter: &ProgressReporter,
    task: TaskConfig,
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

    let progress = reporter.start_task(&task, 0, Some(maximum_iterations))?;
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
    let termination_policy: Option<TerminationPolicy> = task
        .value("termination")
        .map(|_| task.decode_value("termination"))
        .transpose()?;
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
    let mut termination_reason = TerminationReason::MaximumIterations;
    while simulation.state().simulation_time().iteration() < maximum_iterations {
        let time = simulation.step_template()?;
        recording.observe_state(simulation.state())?;
        progress.set_iteration(time.iteration())?;
        let should_sample = monitor
            .as_ref()
            .is_some_and(|monitor| monitor.should_sample(time.iteration()));
        if should_sample {
            let tolerance = monitor
                .as_ref()
                .and_then(|monitor| monitor.policy().fixed_point.as_ref())
                .map(|config| config.residual_tolerance);
            let scaled_residual = if let Some(tolerance) = tolerance {
                simulation
                    .maximum_scaled_residual(tolerance)?
                    .ok_or(crate::termination::TerminationError::ResidualUnavailable)?
            } else {
                0.0
            };
            if let Some(reason) = monitor
                .as_mut()
                .expect("sample decision requires a monitor")
                .observe(simulation.state(), scaled_residual)?
            {
                termination_reason = reason.into();
                break;
            }
        }
    }

    progress.set_phase("validating recording");
    let progress_completion_reason = match &termination_reason {
        TerminationReason::MaximumIterations => None,
        reason => Some(format!("scientific termination: {}", reason.as_str())),
    };
    recording.complete(simulation.state(), termination_reason)?;
    verify_completed_glv_checkpoint(recording_directory, simulation.state())?;
    progress.complete(progress_completion_reason)?;
    Ok(())
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
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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
                    "initial_abundance": [0.4, 0.6],
                    "growth": [0.0, 0.0],
                    "cutoff": 0.0,
                    "physical_time_increment": 0.1,
                    "maximum_iterations": 100,
                    "termination": {
                        "start_after_iteration": 1,
                        "sample_interval_iterations": 1,
                        "observable": "global_state",
                        "fixed_point": {
                            "base_window_samples": 2,
                            "confirmation_window_multipliers": [1, 2],
                            "composition_tolerance": 0.0,
                            "relative_mass_tolerance": 0.0,
                            "mass_floor": 1e-12,
                            "support_threshold": 0.0,
                            "residual_tolerance": {"absolute": 1e-12, "relative": 1e-12}
                        },
                        "oscillation": null
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
            6
        );
        assert_eq!(
            document["terminal_metadata"][TERMINATION_DIAGNOSTICS_METADATA_KEY]["completed_windows"],
            2
        );
    }
}
