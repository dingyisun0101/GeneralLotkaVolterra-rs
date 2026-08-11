//! Project templates and the single ordinary-user execution entry point.

use std::error::Error;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use ndarray::{Array1, ArrayD, IxDyn};
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

use crate::invariant::FrequencyInvariant;
use crate::kernel::{
    Diffusion, InteractionArtifactDescriptor, InteractionSource, JsonInteractionSource, Kernel,
    KernelAlgorithm, KernelCore, MeanFieldReplicatorRk4, persist_interaction_matrix,
};
use crate::noise::{DemographicGaussian, Noise, NoiseAlgorithm, NoiseDomain};
use crate::project::{GlvProjectError, load_glv_project};
use crate::reading::verify_completed_glv_checkpoint;
use crate::recording::{GlvRecording, GlvRecordingMetadata, TerminationReason};
use crate::simulation::{
    MeanFieldReplicator, MeanFieldReplicatorConfig, SimulationKind, SpatialGeneralLotkaVolterra,
    SpatialGeneralLotkaVolterraConfig, SpatialReplicator, SpatialReplicatorConfig,
};
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
        JsonInteractionSource::resolved_file(task.resolve_path("interaction_matrix")?)
            .resolve(species)?;
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
    let initial_cell: Vec<f64> = task.decode_value("initial_cell")?;
    let growth = Array1::from_vec(task.decode_value("growth")?);
    let diffusion_coefficients = Array1::from_vec(task.decode_value("diffusion")?);
    let spacing: Vec<f64> = task.decode_value("spacing")?;
    let boundary = task.decode_value("boundary")?;
    let cutoff = task.decode_value("cutoff")?;
    let time_step = TimeStep::new(task.decode_value("physical_time_increment")?)?;
    let maximum_iterations = task.decode_value("maximum_iterations")?;
    let streams: Vec<StateStreamConfig> = task.decode_value("recording")?;

    let progress = reporter.start_task(&task, 0, Some(maximum_iterations))?;
    progress.set_phase("resolving interaction matrix");
    let species = initial_cell.len();
    let interaction =
        JsonInteractionSource::resolved_file(task.resolve_path("interaction_matrix")?)
            .resolve(species)?;
    let persisted = persist_interaction_matrix(scope, &interaction)?;

    progress.set_phase("constructing simulation");
    let initial_space = tiled_space(&spatial_shape, &initial_cell)?;
    let lattice = SquareLatticeConfig::try_new(&spatial_shape, boundary, Some(&spacing))?;
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
        simulation,
    )
}

fn run_spatial_glv(
    scope: &ExecutionScope,
    reporter: &ProgressReporter,
    task: TaskConfig,
) -> Result<(), TemplateTaskError> {
    let spatial_shape: Vec<usize> = task.decode_value("spatial_shape")?;
    let initial_cell: Vec<f64> = task.decode_value("initial_cell")?;
    let growth = Array1::from_vec(task.decode_value("growth")?);
    let diffusion_coefficients = Array1::from_vec(task.decode_value("diffusion")?);
    let spacing: Vec<f64> = task.decode_value("spacing")?;
    let boundary = task.decode_value("boundary")?;
    let cutoff = task.decode_value("cutoff")?;
    let carrying_capacity = task.decode_value("carrying_capacity")?;
    let time_step = TimeStep::new(task.decode_value("physical_time_increment")?)?;
    let maximum_iterations = task.decode_value("maximum_iterations")?;
    let streams: Vec<StateStreamConfig> = task.decode_value("recording")?;

    let progress = reporter.start_task(&task, 0, Some(maximum_iterations))?;
    progress.set_phase("resolving interaction matrix");
    let species = initial_cell.len();
    let interaction =
        JsonInteractionSource::resolved_file(task.resolve_path("interaction_matrix")?)
            .resolve(species)?;
    let persisted = persist_interaction_matrix(scope, &interaction)?;

    progress.set_phase("constructing simulation");
    let initial_space = tiled_space(&spatial_shape, &initial_cell)?;
    let lattice = SquareLatticeConfig::try_new(&spatial_shape, boundary, Some(&spacing))?;
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
        simulation,
    )
}

fn tiled_space(
    spatial_shape: &[usize],
    initial_cell: &[f64],
) -> Result<ArrayD<f64>, TemplateTaskError> {
    let cells = spatial_shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension));
    let cells = cells.ok_or("spatial cell count overflows usize")?;
    let mut shape = spatial_shape.to_vec();
    shape.push(initial_cell.len());
    Ok(ArrayD::from_shape_vec(
        IxDyn(&shape),
        initial_cell.repeat(cells),
    )?)
}

trait StandardTemplateSimulation {
    fn kind(&self) -> SimulationKind;
    fn abundance_representation(&self) -> AbundanceRepresentation;
    fn state(&self) -> &SystemState;
    fn rng_record(&self) -> Option<&RngRecord>;
    fn step_template(&mut self) -> Result<SimulationTime, TemplateTaskError>;
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
}

#[allow(clippy::too_many_arguments)]
fn finish_task<S>(
    scope: &ExecutionScope,
    task: TaskConfig,
    progress: TaskProgress,
    streams: Vec<StateStreamConfig>,
    maximum_iterations: u64,
    interaction: &InteractionArtifactDescriptor,
    mut simulation: S,
) -> Result<(), TemplateTaskError>
where
    S: StandardTemplateSimulation,
{
    let metadata = GlvRecordingMetadata::new(
        simulation.kind(),
        simulation.abundance_representation(),
        task.parameters(),
        interaction,
        simulation.rng_record(),
    )?;
    let recording_directory = scope.task_recording_directory(task.task_ordinal());
    let mut recording =
        GlvRecording::start(&recording_directory, streams, metadata, simulation.state())?;

    progress.set_phase("evolving");
    while simulation.state().simulation_time().iteration() < maximum_iterations {
        let time = simulation.step_template()?;
        recording.observe_state(simulation.state())?;
        progress.set_iteration(time.iteration())?;
    }

    progress.set_phase("validating recording");
    recording.complete(simulation.state(), TerminationReason::MaximumIterations)?;
    verify_completed_glv_checkpoint(recording_directory, simulation.state())?;
    progress.complete(None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
