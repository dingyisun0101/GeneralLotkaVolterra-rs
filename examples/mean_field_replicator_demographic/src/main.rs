//! Seeded demographic-noise mean-field replicator workflow.

use std::error::Error;
use std::path::PathBuf;

use general_lotka_volterra_rs::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let project = load_glv_project(project_root)?;
    let scope = ExecutionScope::create_generated(project.resolve_path("recordings")?)?;
    let reporter = ProgressReporter::for_project(&project).start()?;
    reporter.report(format!("execution scope: {}", scope.directory().display()))?;

    for task in project.task_configs() {
        run_task(&scope, &reporter, task)?;
    }

    reporter.complete(format!(
        "demographic mean-field execution complete: {}",
        scope.directory().display()
    ))?;
    Ok(())
}

fn run_task(
    scope: &ExecutionScope,
    reporter: &ProgressReporter,
    task: TaskConfig,
) -> Result<(), Box<dyn Error>> {
    // Workflow performs typed per-key decoding directly. No application-owned
    // task or recording transport structure is required.
    let initial_abundance = Array1::from_vec(task.decode_value("initial_abundance")?);
    let growth = Array1::from_vec(task.decode_value("growth")?);
    let cutoff = task.decode_value("cutoff")?;
    let sigma = task.decode_value("sigma")?;
    let seed = task.decode_value("seed")?;
    let time_step = TimeStep::new(task.decode_value("physical_time_increment")?)?;
    let maximum_iterations = task.decode_value("maximum_iterations")?;
    let recording_config: Vec<StateStreamConfig> = task.decode_value("recording")?;

    let progress = reporter.start_task(&task, 0, Some(maximum_iterations))?;
    progress.set_phase("resolving interaction matrix");
    let species = initial_abundance.len();
    let interaction =
        JsonInteractionSource::resolved_file(task.resolve_path("interaction_matrix")?)
            .resolve(species)?;
    let persisted = persist_interaction_matrix(scope, &interaction)?;

    progress.set_phase("constructing seeded simulation");
    // Use the concrete constructor as the canonical state-assembly boundary,
    // then replace only the default no-noise composition.
    let initial_state = MeanFieldReplicator::new(
        initial_abundance,
        interaction.clone(),
        MeanFieldReplicatorConfig::new(growth.clone(), cutoff, time_step),
    )?
    .into_state();
    let kernel = Kernel::new(
        KernelCore::new(interaction),
        MeanFieldReplicatorRk4::new(growth)?,
    );
    let noise = Noise::new(DemographicGaussian::new(
        sigma,
        RngConfig::new(Some(seed), None, None),
        NoiseDomain::aggregate(species)?,
    )?);
    let invariant = FrequencyInvariant::new(species, cutoff)?;
    let mut simulation = MeanFieldReplicator::from_plugins(
        initial_state,
        AbundanceRepresentation::RelativeFrequency,
        kernel,
        noise,
        invariant,
        time_step,
    )?;

    // RNG method, implementation version, key encoding, and seed enter the
    // recording metadata through the plugin's immutable Workflow RNG record.
    let metadata = GlvRecordingMetadata::new(
        simulation.kind(),
        simulation.abundance_representation(),
        task.parameters(),
        persisted.descriptor(),
        simulation.rng_record(),
    )?;
    let recording_directory = scope.task_recording_directory(task.task_ordinal());
    let mut recording = GlvRecording::start(
        &recording_directory,
        recording_config,
        metadata,
        simulation.state(),
    )?;

    progress.set_phase("evolving");
    while simulation.state().simulation_time().iteration() < maximum_iterations {
        let time = simulation.step()?;
        recording.observe_state(simulation.state())?;
        progress.set_iteration(time.iteration())?;
    }

    progress.set_phase("validating recording");
    recording.complete(simulation.state(), TerminationReason::MaximumIterations)?;
    verify_completed_glv_checkpoint(recording_directory, simulation.state())?;
    progress.complete(None)?;
    Ok(())
}
