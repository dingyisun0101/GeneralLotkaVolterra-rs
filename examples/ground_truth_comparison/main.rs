//! Direct deterministic comparison against an independent reference integrator.

use std::error::Error;
use std::io;
use std::path::PathBuf;
use std::process::Command;

use general_lotka_volterra_rs::kernel::{
    Boundary, Diffusion, InteractionSource, JsonInteractionSource,
};
use general_lotka_volterra_rs::{
    MeanFieldReplicator, MeanFieldReplicatorConfig, SPACE_FIELD, SpatialAbundance,
    SpatialGeneralLotkaVolterra, SpatialGeneralLotkaVolterraConfig, TimeStep,
};
use ndarray::{Array1, ArrayD, IxDyn};
use scientific_workflow::prelude::{ScientificProject, TaskConfig};
use serde::Deserialize;

#[derive(Deserialize)]
struct MeanFieldInputs {
    initial_abundance: Vec<f64>,
    growth: Vec<f64>,
    cutoff: f64,
    tolerance: f64,
}

#[derive(Deserialize)]
struct SpatialInputs {
    shape: Vec<usize>,
    initial_space: Vec<f64>,
    growth: Vec<f64>,
    diffusion: Vec<f64>,
    boundary: String,
    cutoff: f64,
    tolerance: f64,
}

#[derive(Deserialize)]
struct Inputs {
    physical_time_increment: f64,
    maximum_iterations: u64,
    mean_field: MeanFieldInputs,
    spatial_no_diffusion: SpatialInputs,
    spatial_periodic_diffusion: SpatialInputs,
}

#[derive(Deserialize)]
struct References {
    mean_field: Vec<f64>,
    spatial_no_diffusion: Vec<f64>,
    spatial_periodic_diffusion: Vec<f64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let project = ScientificProject::load(project_root())?;
    let task = project.task_config(0)?;
    let inputs = Inputs {
        physical_time_increment: task.decode_value("physical_time_increment")?,
        maximum_iterations: task.decode_value("maximum_iterations")?,
        mean_field: task.decode_value("mean_field")?,
        spatial_no_diffusion: task.decode_value("spatial_no_diffusion")?,
        spatial_periodic_diffusion: task.decode_value("spatial_periodic_diffusion")?,
    };

    let mean_field = run_mean_field(&task, &inputs)?;
    let spatial_no_diffusion = run_spatial(
        &task,
        "spatial_no_diffusion_matrix",
        &inputs.spatial_no_diffusion,
        inputs.physical_time_increment,
        inputs.maximum_iterations,
    )?;
    let spatial_periodic_diffusion = run_spatial(
        &task,
        "spatial_periodic_diffusion_matrix",
        &inputs.spatial_periodic_diffusion,
        inputs.physical_time_increment,
        inputs.maximum_iterations,
    )?;
    let references = reference_values(&task)?;

    assert_close(
        "mean_field_replicator",
        &mean_field,
        &references.mean_field,
        inputs.mean_field.tolerance,
    )?;
    assert_close(
        "spatial_glv_no_diffusion",
        &spatial_no_diffusion,
        &references.spatial_no_diffusion,
        inputs.spatial_no_diffusion.tolerance,
    )?;
    assert_close(
        "spatial_glv_periodic_diffusion",
        &spatial_periodic_diffusion,
        &references.spatial_periodic_diffusion,
        inputs.spatial_periodic_diffusion.tolerance,
    )?;
    Ok(())
}

fn run_mean_field(task: &TaskConfig, inputs: &Inputs) -> Result<Vec<f64>, Box<dyn Error>> {
    let case = &inputs.mean_field;
    let species = case.initial_abundance.len();
    let interaction = JsonInteractionSource::resolved_file(task.resolve_path("mean_field_matrix")?)
        .resolve(species)?;
    let mut simulation = MeanFieldReplicator::new(
        Array1::from_vec(case.initial_abundance.clone()),
        interaction,
        MeanFieldReplicatorConfig::new(
            Array1::from_vec(case.growth.clone()),
            case.cutoff,
            TimeStep::new(inputs.physical_time_increment)?,
        ),
    )?;
    for _ in 0..inputs.maximum_iterations {
        simulation.step()?;
    }
    Ok(simulation
        .state()
        .payload::<general_lotka_volterra_rs::AggregateAbundance>(
            general_lotka_volterra_rs::ABUNDANCE_FIELD,
        )?
        .to_vec())
}

fn run_spatial(
    task: &TaskConfig,
    matrix_path_key: &str,
    case: &SpatialInputs,
    dt: f64,
    iterations: u64,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let species = *case
        .shape
        .last()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "spatial shape is empty"))?;
    let interaction = JsonInteractionSource::resolved_file(task.resolve_path(matrix_path_key)?)
        .resolve(species)?;
    let boundary = match case.boundary.as_str() {
        "periodic" => Boundary::Periodic,
        "neumann" => Boundary::Neumann,
        value => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported boundary {value}"),
            )
            .into());
        }
    };
    let diffusion = Diffusion::unit_spacing(
        Array1::from_vec(case.diffusion.clone()),
        case.shape.len() - 1,
        boundary,
    )?;
    let initial = ArrayD::from_shape_vec(IxDyn(&case.shape), case.initial_space.clone())?;
    let mut simulation = SpatialGeneralLotkaVolterra::new(
        initial,
        interaction,
        SpatialGeneralLotkaVolterraConfig::new(
            case.shape.clone(),
            Array1::from_vec(case.growth.clone()),
            diffusion,
            case.cutoff,
            None,
            TimeStep::new(dt)?,
        ),
    )?;
    for _ in 0..iterations {
        simulation.step()?;
    }
    Ok(simulation
        .state()
        .payload::<SpatialAbundance>(SPACE_FIELD)?
        .as_ref()
        .expect("spatial GLV retains its space payload")
        .iter()
        .copied()
        .collect())
}

fn reference_values(task: &TaskConfig) -> Result<References, Box<dyn Error>> {
    let script = task.resolve_path("reference_integrator")?;
    let mut last_not_found = None;
    for interpreter in ["python", "python3"] {
        match Command::new(interpreter).arg(&script).output() {
            Ok(output) if output.status.success() => {
                return Ok(serde_json::from_slice(&output.stdout)?);
            }
            Ok(output) => {
                return Err(io::Error::other(format!(
                    "{} failed: {}",
                    script.display(),
                    String::from_utf8_lossy(&output.stderr)
                ))
                .into());
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                last_not_found = Some(error);
            }
            Err(error) => return Err(error.into()),
        }
    }
    Err(last_not_found
        .unwrap_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Python not found"))
        .into())
}

fn assert_close(
    name: &str,
    actual: &[f64],
    expected: &[f64],
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{name}: length {} != {}", actual.len(), expected.len()),
        )
        .into());
    }
    let maximum_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    if maximum_error > tolerance {
        return Err(io::Error::other(format!(
            "{name}: maximum absolute error {maximum_error:e} exceeds {tolerance:e}"
        ))
        .into());
    }
    println!("ok {name}: max_abs_error={maximum_error:e}");
    Ok(())
}

fn project_root() -> PathBuf {
    std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/ground_truth_comparison")
        })
}
