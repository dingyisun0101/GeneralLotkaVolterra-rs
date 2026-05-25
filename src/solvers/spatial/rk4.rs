//! Arbitrary-dimensional spatial GLV RK4 solver.
//!
//! Purpose:
//! This module implements deterministic reaction-diffusion dynamics for a
//! species-last spatial field, `space[x0, x1, ..., x{k-1}, i]`.
//!
//! The local reaction is GLV population dynamics and diffusion is a
//! finite-difference Laplacian across the spatial axes.
//!
//! Evolution contract:
//! One solver step follows this sequence: RK4 raw reaction-diffusion step,
//! spatial sanitize/global-state refresh, then snapshot check.

use std::io::{Error, ErrorKind, Result};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayD};

use crate::io::signal::SignalWriter;
use crate::io::space::SpaceWriter;
use crate::solvers::termination::{
    SolveOutcome, TerminationChecker, TerminationConfig, TerminationReason,
};
use crate::{Mode, SIGNAL_OUTPUT_FILE_SIZE, SPACE_OUTPUT_FILE_SIZE, SystemState};

/// Boundary policy for finite-difference diffusion.
#[derive(Clone, Copy, Debug)]
pub enum Boundary {
    /// Wrap neighbors at each spatial edge.
    Periodic,

    /// Reflect zero-flux boundaries by reusing the edge value out of bounds.
    Neumann,
}

/// Diffusion configuration for the spatial RK4 solver.
///
/// Details:
/// - Purpose: Stores per-species diffusion strengths, per-axis grid spacing,
///   and boundary behavior.
/// - Parameters:
///   - `coefficients`: Diffusion coefficient `D_i` for each species.
///   - `spacing`: Grid spacing for each spatial axis.
///   - `boundary`: Boundary condition used by the Laplacian.
#[derive(Clone, Debug)]
pub struct Diffusion {
    pub coefficients: Array1<f64>,
    pub spacing: Vec<f64>,
    pub boundary: Boundary,
}

impl Diffusion {
    /// Construct diffusion with unit grid spacing on every spatial axis.
    ///
    /// Details:
    /// - Purpose: Convenience constructor for regular unit grids.
    /// - Parameters:
    ///   - `coefficients`: Per-species diffusion strengths.
    ///   - `spatial_ndim`: Number of spatial axes, excluding species.
    ///   - `boundary`: Boundary condition used by the Laplacian.
    #[inline]
    pub fn unit_spacing(
        coefficients: Array1<f64>,
        spatial_ndim: usize,
        boundary: Boundary,
    ) -> Self {
        Self {
            coefficients,
            spacing: vec![1.0; spatial_ndim],
            boundary,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Dynamics {
    GlvPopulation,
    LocalReplicatorFrequency,
}

/// Scratch buffers for spatial RK4 (avoid repeated allocations).
///
/// Details:
/// - Purpose: Owns stage derivatives and temporary spatial storage for the hot
///   integration loop.
/// - Parameters:
///   - (none): Construct with `SpatialRk4Scratch::new`.
struct SpatialRk4Scratch {
    k1: ArrayD<f64>,
    k2: ArrayD<f64>,
    k3: ArrayD<f64>,
    k4: ArrayD<f64>,
    tmp: ArrayD<f64>,
}

impl SpatialRk4Scratch {
    #[inline]
    fn new(shape: &[usize]) -> Self {
        Self {
            k1: ArrayD::zeros(shape),
            k2: ArrayD::zeros(shape),
            k3: ArrayD::zeros(shape),
            k4: ArrayD::zeros(shape),
            tmp: ArrayD::zeros(shape),
        }
    }
}

/// Cached shape facts for species-last spatial arrays.
///
/// Details:
/// - Purpose: Stores dimensional metadata and row-major strides so the solver
///   can use flat indexing in the hot loop.
/// - Parameters:
///   - (none): Construct with `SpatialLayout::new`.
struct SpatialLayout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    spatial_ndim: usize,
    num_species: usize,
    num_cells: usize,
}

impl SpatialLayout {
    fn new(shape: &[usize]) -> Result<Self> {
        if shape.len() < 2 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "spatial state shape must include at least one spatial axis and one species axis",
            ));
        }

        let num_species = *shape.last().expect("shape length checked");
        if num_species == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "species axis length must be > 0",
            ));
        }

        if shape[..shape.len() - 1].contains(&0) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "spatial axis lengths must be > 0",
            ));
        }

        let mut strides = vec![1; shape.len()];
        for axis in (0..shape.len() - 1).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1];
        }

        let spatial_ndim = shape.len() - 1;
        let num_cells = shape[..spatial_ndim].iter().product();

        Ok(Self {
            shape: shape.to_vec(),
            strides,
            spatial_ndim,
            num_species,
            num_cells,
        })
    }
}

/// Compute the spatial GLV reaction-diffusion RHS in-place.
///
/// Details:
/// - Purpose: Evaluates
///       out_i(x) = u_i(x) * (g_i + Σ_j V_ij u_j(x)) + D_i Δu_i(x)
///   without allocating.
/// - Parameters:
///   - `space`: Current species-last spatial field.
///   - `growth_vector`: Growth vector `g`.
///   - `interaction_matrix`: Interaction matrix `V`.
///   - `diffusion`: Diffusion configuration.
///   - `layout`: Cached species-last shape/stride facts.
///   - `out`: Destination derivative.
#[inline]
fn rhs_inplace(
    space: &ArrayD<f64>,
    growth_vector: &Array1<f64>,
    interaction_matrix: &Array2<f64>,
    diffusion: &Diffusion,
    layout: &SpatialLayout,
    dynamics: Dynamics,
    out: &mut ArrayD<f64>,
) -> Result<()> {
    let u = space.as_slice_memory_order().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidInput,
            "spatial state must use standard contiguous memory layout",
        )
    })?;
    let y = out.as_slice_memory_order_mut().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidInput,
            "spatial output must use standard contiguous memory layout",
        )
    })?;

    let d = layout.num_species;

    for cell in 0..layout.num_cells {
        let base = cell * d;

        let upsilon = match dynamics {
            Dynamics::GlvPopulation => 0.0,
            Dynamics::LocalReplicatorFrequency => {
                let mut acc = 0.0;
                for i in 0..d {
                    let mut interaction = 0.0;
                    for j in 0..d {
                        interaction += interaction_matrix[(i, j)] * u[base + j];
                    }
                    acc += u[base + i] * (growth_vector[i] + interaction);
                }
                acc
            }
        };

        for species in 0..d {
            let center_idx = base + species;
            let center = u[center_idx];

            let mut interaction = 0.0;
            for j in 0..d {
                interaction += interaction_matrix[(species, j)] * u[base + j];
            }

            let mut laplacian = 0.0;
            for axis in 0..layout.spatial_ndim {
                let axis_len = layout.shape[axis];
                let stride = layout.strides[axis];
                let coord = (base / stride) % axis_len;
                let inv_dx2 = 1.0 / (diffusion.spacing[axis] * diffusion.spacing[axis]);

                let plus_idx = if coord + 1 < axis_len {
                    center_idx + stride
                } else {
                    match diffusion.boundary {
                        Boundary::Periodic => center_idx - (axis_len - 1) * stride,
                        Boundary::Neumann => center_idx,
                    }
                };

                let minus_idx = if coord > 0 {
                    center_idx - stride
                } else {
                    match diffusion.boundary {
                        Boundary::Periodic => center_idx + (axis_len - 1) * stride,
                        Boundary::Neumann => center_idx,
                    }
                };

                laplacian += (u[plus_idx] + u[minus_idx] - 2.0 * center) * inv_dx2;
            }

            let reaction = match dynamics {
                Dynamics::GlvPopulation => center * (growth_vector[species] + interaction),
                Dynamics::LocalReplicatorFrequency => {
                    center * (growth_vector[species] + interaction - upsilon)
                }
            };

            y[center_idx] = reaction + diffusion.coefficients[species] * laplacian;
        }
    }

    Ok(())
}

/// One RK4 reaction-diffusion step writing into `out`.
///
/// Details:
/// - Purpose: Advances one raw deterministic spatial step without enforcing
///   mode-specific state invariants.
/// - Parameters:
///   - `space`: Current species-last spatial field.
///   - `growth_vector`: Growth vector `g`.
///   - `interaction_matrix`: Interaction matrix `V`.
///   - `diffusion`: Diffusion configuration.
///   - `dt`: Step size.
///   - `layout`: Cached species-last shape/stride facts.
///   - `sc`: Reusable RK4 scratch storage.
///   - `out`: Raw next-space destination.
#[inline]
fn rk4_step_inplace_raw(
    space: &ArrayD<f64>,
    growth_vector: &Array1<f64>,
    interaction_matrix: &Array2<f64>,
    diffusion: &Diffusion,
    dt: f64,
    layout: &SpatialLayout,
    dynamics: Dynamics,
    sc: &mut SpatialRk4Scratch,
    out: &mut ArrayD<f64>,
) -> Result<()> {
    let u = space.as_slice_memory_order().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidInput,
            "spatial state must use standard contiguous memory layout",
        )
    })?;
    let half_dt = 0.5 * dt;
    let dt_over_6 = dt / 6.0;

    rhs_inplace(
        space,
        growth_vector,
        interaction_matrix,
        diffusion,
        layout,
        dynamics,
        &mut sc.k1,
    )?;

    {
        let k1 = sc
            .k1
            .as_slice_memory_order()
            .expect("scratch is contiguous");
        let tmp = sc
            .tmp
            .as_slice_memory_order_mut()
            .expect("scratch is contiguous");
        for i in 0..u.len() {
            tmp[i] = u[i] + half_dt * k1[i];
        }
    }
    rhs_inplace(
        &sc.tmp,
        growth_vector,
        interaction_matrix,
        diffusion,
        layout,
        dynamics,
        &mut sc.k2,
    )?;

    {
        let k2 = sc
            .k2
            .as_slice_memory_order()
            .expect("scratch is contiguous");
        let tmp = sc
            .tmp
            .as_slice_memory_order_mut()
            .expect("scratch is contiguous");
        for i in 0..u.len() {
            tmp[i] = u[i] + half_dt * k2[i];
        }
    }
    rhs_inplace(
        &sc.tmp,
        growth_vector,
        interaction_matrix,
        diffusion,
        layout,
        dynamics,
        &mut sc.k3,
    )?;

    {
        let k3 = sc
            .k3
            .as_slice_memory_order()
            .expect("scratch is contiguous");
        let tmp = sc
            .tmp
            .as_slice_memory_order_mut()
            .expect("scratch is contiguous");
        for i in 0..u.len() {
            tmp[i] = u[i] + dt * k3[i];
        }
    }
    rhs_inplace(
        &sc.tmp,
        growth_vector,
        interaction_matrix,
        diffusion,
        layout,
        dynamics,
        &mut sc.k4,
    )?;

    {
        let k1 = sc
            .k1
            .as_slice_memory_order()
            .expect("scratch is contiguous");
        let k2 = sc
            .k2
            .as_slice_memory_order()
            .expect("scratch is contiguous");
        let k3 = sc
            .k3
            .as_slice_memory_order()
            .expect("scratch is contiguous");
        let k4 = sc
            .k4
            .as_slice_memory_order()
            .expect("scratch is contiguous");
        let y = out
            .as_slice_memory_order_mut()
            .expect("output is contiguous");

        for i in 0..u.len() {
            y[i] = u[i] + dt_over_6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }

    Ok(())
}

/// Clamp spatial values, apply optional global capacity, and refresh `state`.
///
/// Details:
/// - Purpose: Treats `space` as the source of truth, restores population
///   feasibility, and writes the global per-species totals into `state`.
/// - Parameters:
///   - `gs`: Spatial state to sanitize.
///   - `layout`: Cached species-last shape/stride facts.
#[inline]
fn sanitize_space_and_refresh_state(
    gs: &mut SystemState<f64>,
    layout: &SpatialLayout,
) -> Result<()> {
    let Mode::Population {
        cutoff,
        carrying_capacity,
    } = gs.mode
    else {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "spatial RK4 currently supports Mode::Population only",
        ));
    };

    let cutoff = cutoff.unwrap_or(0.0).max(0.0);
    let Some(space) = gs.space.as_mut() else {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "spatial RK4 requires SystemState.space",
        ));
    };
    let u = space.as_slice_memory_order_mut().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidInput,
            "spatial state must use standard contiguous memory layout",
        )
    })?;

    gs.state.fill(0.0);
    for cell in 0..layout.num_cells {
        let base = cell * layout.num_species;
        for species in 0..layout.num_species {
            let idx = base + species;
            if !u[idx].is_finite() || u[idx] <= 0.0 || u[idx] < cutoff {
                u[idx] = 0.0;
            }
            gs.state[species] += u[idx];
        }
    }

    let mut total = gs.state.sum();
    if let Some(capacity) = carrying_capacity {
        if capacity <= 0.0 {
            for x in u.iter_mut() {
                *x = 0.0;
            }
            gs.state.fill(0.0);
            gs.mass = 0.0;
            return Ok(());
        }

        if total > capacity && total > 0.0 {
            let scale = capacity / total;
            for x in u.iter_mut() {
                *x *= scale;
            }
            for x in gs.state.iter_mut() {
                *x *= scale;
            }
            total = capacity;
        }
    }

    gs.mass = total.round().max(0.0);
    Ok(())
}

/// Normalize every spatial cell onto the simplex and refresh global frequency.
///
/// Details:
/// - Purpose: Treats `space` as local species frequencies, enforces one
///   simplex per spatial cell, and writes the cell-average frequency into
///   `state`.
/// - Parameters:
///   - `gs`: Spatial state to sanitize.
///   - `layout`: Cached species-last shape/stride facts.
#[inline]
fn sanitize_local_simplex_space_and_refresh_state(
    gs: &mut SystemState<f64>,
    layout: &SpatialLayout,
) -> Result<()> {
    let Mode::Frequency { cutoff } = gs.mode else {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "spatial replicator RK4 currently supports Mode::Frequency only",
        ));
    };

    let cutoff = cutoff.unwrap_or(0.0).max(0.0);
    let Some(space) = gs.space.as_mut() else {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "spatial RK4 requires SystemState.space",
        ));
    };
    let u = space.as_slice_memory_order_mut().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidInput,
            "spatial state must use standard contiguous memory layout",
        )
    })?;

    gs.state.fill(0.0);
    for cell in 0..layout.num_cells {
        let base = cell * layout.num_species;
        let mut local_sum = 0.0;

        for species in 0..layout.num_species {
            let idx = base + species;
            if !u[idx].is_finite() || u[idx] <= 0.0 || u[idx] < cutoff {
                u[idx] = 0.0;
            }
            local_sum += u[idx];
        }

        if local_sum > 0.0 {
            let inv = 1.0 / local_sum;
            for species in 0..layout.num_species {
                let idx = base + species;
                u[idx] *= inv;
                gs.state[species] += u[idx];
            }
        } else {
            let uniform = 1.0 / layout.num_species as f64;
            for species in 0..layout.num_species {
                let idx = base + species;
                u[idx] = uniform;
                gs.state[species] += uniform;
            }
        }
    }

    let inv_cells = 1.0 / layout.num_cells as f64;
    for x in gs.state.iter_mut() {
        *x *= inv_cells;
    }
    gs.mass = 1.0;

    Ok(())
}

fn validate_inputs(
    layout: &SpatialLayout,
    interaction_matrix: &Array2<f64>,
    growth_vector: Option<&Array1<f64>>,
    diffusion: &Diffusion,
    dt: f64,
    save_signal_interval: usize,
    save_space_interval: usize,
) -> Result<()> {
    let d = layout.num_species;

    if interaction_matrix.nrows() != d || interaction_matrix.ncols() != d {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "interaction_matrix must be square with size matching the species axis",
        ));
    }
    if let Some(g) = growth_vector {
        if g.len() != d {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "growth_vector length must match the species axis",
            ));
        }
    }
    if diffusion.coefficients.len() != d {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "diffusion coefficient length must match the species axis",
        ));
    }
    if diffusion.spacing.len() != layout.spatial_ndim {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "diffusion spacing length must match the number of spatial axes",
        ));
    }
    if diffusion
        .coefficients
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "diffusion coefficients must be finite and nonnegative",
        ));
    }
    if diffusion
        .spacing
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "diffusion spacing values must be finite and positive",
        ));
    }
    if !dt.is_finite() || dt < 0.0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "dt must be finite and nonnegative",
        ));
    }
    if save_signal_interval == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "save_signal_interval must be >= 1",
        ));
    }
    if save_space_interval == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "save_space_interval must be >= 1",
        ));
    }

    let max_diffusion = diffusion
        .coefficients
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    if max_diffusion > 0.0 {
        let inv_dx2_sum: f64 = diffusion.spacing.iter().map(|dx| 1.0 / (dx * dx)).sum();
        let conservative_limit = 1.0 / (2.0 * max_diffusion * inv_dx2_sum);
        if dt > conservative_limit {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "dt={dt} exceeds conservative explicit diffusion stability limit {conservative_limit}"
                ),
            ));
        }
    }

    Ok(())
}

/// Integrate a single spatial trajectory and persist split signal/space output.
///
/// Details:
/// - Purpose: Runs one spatial trajectory and writes aggregate signal and full
///   spatial snapshots to independent output streams.
/// - Parameters:
///   - `gs_i`: Initial spatial population state consumed by the solver.
///   - `interaction_matrix`: Square interaction matrix `V`.
///   - `growth_vector`: Optional growth vector `g`; defaults to zero.
///   - `diffusion`: Per-species diffusion coefficients and grid metadata.
///   - `dt`: Step size.
///   - `num_steps`: Number of integration steps.
///   - `save_signal_interval`: Save aggregate state every Nth step; `t = 0`
///     is always saved.
///   - `save_space_interval`: Include full spatial field every Nth step;
///     `t = 0` is always saved.
///   - `output_path`: Directory for split signal/space JSON output.
///   - `progress_counter`: Optional shared progress counter.
fn solve_impl(
    mut gs_i: SystemState<f64>,
    interaction_matrix: &Array2<f64>,
    growth_vector: Option<&Array1<f64>>,
    diffusion: &Diffusion,
    dt: f64,
    num_steps: usize,
    save_signal_interval: usize,
    save_space_interval: usize,
    output_path: &Path,
    progress_counter: Option<&AtomicUsize>,
    dynamics: Dynamics,
    termination: TerminationConfig,
) -> Result<SolveOutcome> {
    let Some(space) = gs_i.space.take() else {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "spatial RK4 requires SystemState.space",
        ));
    };
    let space = space.to_owned();
    let layout = SpatialLayout::new(space.shape())?;

    validate_inputs(
        &layout,
        interaction_matrix,
        growth_vector,
        diffusion,
        dt,
        save_signal_interval,
        save_space_interval,
    )?;

    let d = layout.num_species;
    let growth_vector_owned = growth_vector
        .map(|x| x.to_owned())
        .unwrap_or_else(|| Array1::zeros(d));

    if gs_i.state.len() != d {
        gs_i.state = Array1::zeros(d);
    }
    gs_i.space = Some(space);
    match dynamics {
        Dynamics::GlvPopulation => sanitize_space_and_refresh_state(&mut gs_i, &layout)?,
        Dynamics::LocalReplicatorFrequency => {
            sanitize_local_simplex_space_and_refresh_state(&mut gs_i, &layout)?
        }
    }

    let mut gs_curr = gs_i;
    let space_len = gs_curr.space.as_ref().map(|space| space.len()).unwrap_or(0);
    let mut signal_writer = SignalWriter::new(
        output_path,
        gs_curr.mode.clone(),
        SIGNAL_OUTPUT_FILE_SIZE,
        gs_curr.state.len(),
    )?;
    let mut space_writer = SpaceWriter::new(
        output_path,
        gs_curr.mode.clone(),
        SPACE_OUTPUT_FILE_SIZE,
        gs_curr.state.len(),
        space_len,
    )?;
    signal_writer.push(&gs_curr)?;
    space_writer.push(&gs_curr)?;

    if let Some(counter) = progress_counter {
        counter.store(0, Ordering::Relaxed);
    }

    let shape = layout.shape.clone();
    let mode0 = gs_curr.mode.clone();
    let mut gs_next = SystemState::from_arrays(
        mode0,
        0,
        Array1::zeros(d),
        Some(ArrayD::zeros(shape.clone())),
    );
    let mut sc = SpatialRk4Scratch::new(&shape);
    let mut next_space = ArrayD::zeros(shape);
    let mut termination_checker = TerminationChecker::new(termination)?;

    let start_time = gs_curr.time;
    let mut steps_run = 0usize;
    let mut termination_reason = TerminationReason::MaxSteps;
    for step in 1..=num_steps {
        let curr_space = gs_curr.space.as_ref().expect("space initialized");
        rk4_step_inplace_raw(
            curr_space,
            &growth_vector_owned,
            interaction_matrix,
            diffusion,
            dt,
            &layout,
            dynamics,
            &mut sc,
            &mut next_space,
        )?;

        gs_next.space = Some(next_space);
        match dynamics {
            Dynamics::GlvPopulation => sanitize_space_and_refresh_state(&mut gs_next, &layout)?,
            Dynamics::LocalReplicatorFrequency => {
                sanitize_local_simplex_space_and_refresh_state(&mut gs_next, &layout)?
            }
        }
        gs_next.time = start_time + step;

        std::mem::swap(&mut gs_curr, &mut gs_next);
        next_space = gs_next.space.take().expect("space buffer retained");
        steps_run = step;

        let save_signal = step % save_signal_interval == 0;
        let save_space = step % save_space_interval == 0;
        if save_signal {
            signal_writer.push(&gs_curr)?;
        }
        if save_space {
            space_writer.push(&gs_curr)?;
        }

        if let Some(counter) = progress_counter {
            counter.store(step, Ordering::Relaxed);
        }

        if let Some(checker) = termination_checker.as_mut() {
            if let Some(reason) = checker.check(&gs_curr, step) {
                termination_reason = reason;
                if !save_signal {
                    signal_writer.push(&gs_curr)?;
                }
                if !save_space {
                    space_writer.push(&gs_curr)?;
                }
                break;
            }
        }
    }

    let signal_stats = signal_writer.finish()?;
    let space_stats = space_writer.finish()?;

    Ok(SolveOutcome {
        final_state: gs_curr,
        steps_run,
        reason: termination_reason,
        signal_stats,
        space_stats: Some(space_stats),
    })
}

/// Integrate a single spatial GLV trajectory and persist split signal/space output.
///
/// Details:
/// - Purpose: Runs one trajectory of arbitrary-dimensional spatial GLV
///   reaction-diffusion over `Mode::Population` fields.
/// - Parameters:
///   - `gs_i`: Initial spatial population state consumed by the solver.
///   - `interaction_matrix`: Square interaction matrix `V`.
///   - `growth_vector`: Optional growth vector `g`; defaults to zero.
///   - `diffusion`: Per-species diffusion coefficients and grid metadata.
///   - `dt`: Step size.
///   - `num_steps`: Number of integration steps.
///   - `save_signal_interval`: Save aggregate state every Nth step; `t = 0`
///     is always saved.
///   - `save_space_interval`: Include full spatial field every Nth step;
///     `t = 0` is always saved.
///   - `output_path`: Directory for split signal/space JSON output.
///   - `progress_counter`: Optional shared progress counter.
pub fn solve(
    gs_i: SystemState<f64>,
    interaction_matrix: &Array2<f64>,
    growth_vector: Option<&Array1<f64>>,
    diffusion: &Diffusion,
    dt: f64,
    num_steps: usize,
    save_signal_interval: usize,
    save_space_interval: usize,
    output_path: &Path,
    progress_counter: Option<&AtomicUsize>,
) -> Result<SystemState<f64>> {
    Ok(solve_with_termination(
        gs_i,
        interaction_matrix,
        growth_vector,
        diffusion,
        dt,
        num_steps,
        save_signal_interval,
        save_space_interval,
        output_path,
        progress_counter,
        TerminationConfig::disabled(),
    )?
    .final_state)
}

/// Integrate a single spatial GLV trajectory with explicit termination.
pub fn solve_with_termination(
    gs_i: SystemState<f64>,
    interaction_matrix: &Array2<f64>,
    growth_vector: Option<&Array1<f64>>,
    diffusion: &Diffusion,
    dt: f64,
    num_steps: usize,
    save_signal_interval: usize,
    save_space_interval: usize,
    output_path: &Path,
    progress_counter: Option<&AtomicUsize>,
    termination: TerminationConfig,
) -> Result<SolveOutcome> {
    solve_impl(
        gs_i,
        interaction_matrix,
        growth_vector,
        diffusion,
        dt,
        num_steps,
        save_signal_interval,
        save_space_interval,
        output_path,
        progress_counter,
        Dynamics::GlvPopulation,
        termination,
    )
}

/// Integrate a single spatial replicator trajectory and persist split signal/space output.
///
/// Details:
/// - Purpose: Runs one trajectory of arbitrary-dimensional local-simplex
///   reaction-diffusion over `Mode::Frequency` fields.
/// - Parameters:
///   - `gs_i`: Initial spatial frequency state consumed by the solver.
///   - `interaction_matrix`: Square interaction matrix `V`.
///   - `growth_vector`: Optional growth vector `g`; defaults to zero.
///   - `diffusion`: Per-species diffusion coefficients and grid metadata.
///   - `dt`: Step size.
///   - `num_steps`: Number of integration steps.
///   - `save_signal_interval`: Save aggregate state every Nth step; `t = 0`
///     is always saved.
///   - `save_space_interval`: Include full spatial field every Nth step;
///     `t = 0` is always saved.
///   - `output_path`: Directory for split signal/space JSON output.
///   - `progress_counter`: Optional shared progress counter.
pub fn solve_replicator(
    gs_i: SystemState<f64>,
    interaction_matrix: &Array2<f64>,
    growth_vector: Option<&Array1<f64>>,
    diffusion: &Diffusion,
    dt: f64,
    num_steps: usize,
    save_signal_interval: usize,
    save_space_interval: usize,
    output_path: &Path,
    progress_counter: Option<&AtomicUsize>,
) -> Result<SystemState<f64>> {
    Ok(solve_replicator_with_termination(
        gs_i,
        interaction_matrix,
        growth_vector,
        diffusion,
        dt,
        num_steps,
        save_signal_interval,
        save_space_interval,
        output_path,
        progress_counter,
        TerminationConfig::disabled(),
    )?
    .final_state)
}

/// Integrate a single spatial replicator trajectory with explicit termination.
pub fn solve_replicator_with_termination(
    gs_i: SystemState<f64>,
    interaction_matrix: &Array2<f64>,
    growth_vector: Option<&Array1<f64>>,
    diffusion: &Diffusion,
    dt: f64,
    num_steps: usize,
    save_signal_interval: usize,
    save_space_interval: usize,
    output_path: &Path,
    progress_counter: Option<&AtomicUsize>,
    termination: TerminationConfig,
) -> Result<SolveOutcome> {
    solve_impl(
        gs_i,
        interaction_matrix,
        growth_vector,
        diffusion,
        dt,
        num_steps,
        save_signal_interval,
        save_space_interval,
        output_path,
        progress_counter,
        Dynamics::LocalReplicatorFrequency,
        termination,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;
    use std::fs;

    fn temp_output_dir(test_name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "glv_spatial_rk4_{test_name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        path
    }

    #[test]
    fn zero_reaction_and_zero_diffusion_preserves_arbitrary_dimensional_space() {
        let shape = vec![2, 3, 2, 2];
        let data = vec![
            1.0, 2.0, 0.5, 1.5, 2.0, 1.0, 3.0, 0.25, 1.25, 2.5, 0.75, 1.75, 1.0, 2.0, 0.5, 1.5,
            2.0, 1.0, 3.0, 0.25, 1.25, 2.5, 0.75, 1.75,
        ];
        let space = ArrayD::from_shape_vec(IxDyn(&shape), data.clone()).expect("valid shape");
        let gs = SystemState::from_arrays(
            Mode::Population {
                cutoff: None,
                carrying_capacity: None,
            },
            0,
            Array1::zeros(2),
            Some(space),
        );
        let interaction_matrix = Array2::zeros((2, 2));
        let growth_vector = Array1::zeros(2);
        let diffusion = Diffusion::unit_spacing(Array1::zeros(2), 3, Boundary::Periodic);
        let output_path = temp_output_dir("preserve");

        let out = solve(
            gs,
            &interaction_matrix,
            Some(&growth_vector),
            &diffusion,
            0.1,
            3,
            1,
            1,
            &output_path,
            None,
        )
        .expect("solve succeeds");

        let final_space = out.space.expect("space retained");
        assert_eq!(final_space.shape(), shape.as_slice());
        assert_eq!(
            final_space.as_slice_memory_order().unwrap(),
            data.as_slice()
        );
        let _ = fs::remove_dir_all(output_path);
    }

    #[test]
    fn periodic_diffusion_conserves_total_population_on_three_dimensional_grid() {
        let shape = vec![2, 2, 2, 1];
        let mut data = vec![0.0; 8];
        data[0] = 1.0;
        let space = ArrayD::from_shape_vec(IxDyn(&shape), data).expect("valid shape");
        let gs = SystemState::from_arrays(
            Mode::Population {
                cutoff: None,
                carrying_capacity: None,
            },
            0,
            Array1::zeros(1),
            Some(space),
        );
        let interaction_matrix = Array2::zeros((1, 1));
        let growth_vector = Array1::zeros(1);
        let diffusion = Diffusion::unit_spacing(Array1::from_vec(vec![0.1]), 3, Boundary::Periodic);
        let output_path = temp_output_dir("mass");

        let out = solve(
            gs,
            &interaction_matrix,
            Some(&growth_vector),
            &diffusion,
            0.01,
            5,
            5,
            5,
            &output_path,
            None,
        )
        .expect("solve succeeds");

        assert!((out.state[0] - 1.0).abs() < 1e-12);
        assert!((out.mass - 1.0).abs() < 1e-12);
        let _ = fs::remove_dir_all(output_path);
    }

    #[test]
    fn replicator_solver_keeps_each_spatial_cell_on_simplex() {
        let shape = vec![2, 2, 2];
        let data = vec![0.9, 0.1, 0.25, 0.75, 0.4, 0.6, 0.8, 0.2];
        let space = ArrayD::from_shape_vec(IxDyn(&shape), data).expect("valid shape");
        let gs = SystemState::from_arrays(
            Mode::Frequency { cutoff: None },
            0,
            Array1::zeros(2),
            Some(space),
        );
        let interaction_matrix = Array2::zeros((2, 2));
        let growth_vector = Array1::zeros(2);
        let diffusion =
            Diffusion::unit_spacing(Array1::from_vec(vec![0.01, 0.01]), 2, Boundary::Neumann);
        let output_path = temp_output_dir("replicator_simplex");

        let out = solve_replicator(
            gs,
            &interaction_matrix,
            Some(&growth_vector),
            &diffusion,
            0.01,
            3,
            3,
            3,
            &output_path,
            None,
        )
        .expect("solve succeeds");

        let final_space = out.space.expect("space retained");
        let values = final_space.as_slice_memory_order().unwrap();
        for cell in 0..4 {
            let base = cell * 2;
            assert!((values[base] + values[base + 1] - 1.0).abs() < 1e-12);
        }
        assert!((out.state.sum() - 1.0).abs() < 1e-12);
        assert_eq!(out.mass, 1.0);
        let _ = fs::remove_dir_all(output_path);
    }

    #[test]
    fn spatial_save_intervals_split_signal_and_space_outputs() {
        let shape = vec![2, 2, 1];
        let space = ArrayD::from_elem(IxDyn(&shape), 1.0);
        let gs = SystemState::from_arrays(
            Mode::Population {
                cutoff: None,
                carrying_capacity: None,
            },
            0,
            Array1::zeros(1),
            Some(space),
        );
        let interaction_matrix = Array2::zeros((1, 1));
        let growth_vector = Array1::zeros(1);
        let diffusion = Diffusion::unit_spacing(Array1::zeros(1), 2, Boundary::Neumann);
        let output_path = temp_output_dir("split_save");

        solve(
            gs,
            &interaction_matrix,
            Some(&growth_vector),
            &diffusion,
            0.01,
            5,
            2,
            5,
            &output_path,
            None,
        )
        .expect("solve succeeds");

        let signal_raw =
            fs::read_to_string(output_path.join("signal/1.json")).expect("signal json exists");
        let space_raw =
            fs::read_to_string(output_path.join("space/1.json")).expect("space json exists");
        let signal_json: serde_json::Value =
            serde_json::from_str(&signal_raw).expect("signal json parses");
        let space_json: serde_json::Value =
            serde_json::from_str(&space_raw).expect("space json parses");

        let signal_times: Vec<usize> = signal_json["samples"]
            .as_array()
            .expect("signal samples")
            .iter()
            .map(|sample| sample["time"].as_u64().expect("time") as usize)
            .collect();
        let space_times: Vec<usize> = space_json["samples"]
            .as_array()
            .expect("space samples")
            .iter()
            .map(|sample| sample["time"].as_u64().expect("time") as usize)
            .collect();

        assert_eq!(signal_times, vec![0, 2, 4]);
        assert_eq!(space_times, vec![0, 5]);
        let _ = fs::remove_dir_all(output_path);
    }
}
