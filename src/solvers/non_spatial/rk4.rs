/*!
Well-mixed replicator RK4 solver.

Purpose:
    This module implements the active trajectory solver. It computes the
    replicator right-hand side, advances the state with RK4, restores state
    invariants, applies optional noise, and persists aggregate signal snapshots.

Evolution contract:
    One solver step follows this sequence: RK4 raw step,
    `SystemState::sanitize`, optional noise, then snapshot check.
*/

use std::io::Result;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2};

use rand::SeedableRng;
use rand::rngs::SmallRng;

use super::noise::{Noise, NoiseContext, apply_noise_inplace};
use crate::io::signal::SignalWriter;
use crate::solvers::termination::{
    SolveOutcome, TerminationChecker, TerminationConfig, TerminationReason,
};
use crate::{SIGNAL_OUTPUT_FILE_SIZE, SystemState};

/// Scratch buffers for RK4 (avoid repeated allocations).
///
/// Details:
/// - Purpose: Owns stage derivatives and matrix-vector scratch storage for the
///   hot integration loop.
/// - Parameters:
///   - (none): Construct with `Rk4Scratch::new`.
struct Rk4Scratch {
    k1: Array1<f64>,    // stage 1 derivative
    k2: Array1<f64>,    // stage 2 derivative
    k3: Array1<f64>,    // stage 3 derivative
    k4: Array1<f64>,    // stage 4 derivative
    tmp: Array1<f64>,   // intermediate ν
    w: Array1<f64>,     // w = Vν
    drift: Array1<f64>, // drift = g + w - Υ
}

impl Rk4Scratch {
    #[inline]
    fn new(d: usize) -> Self {
        Self {
            k1: Array1::zeros(d),
            k2: Array1::zeros(d),
            k3: Array1::zeros(d),
            k4: Array1::zeros(d),
            tmp: Array1::zeros(d),
            w: Array1::zeros(d),
            drift: Array1::zeros(d),
        }
    }
}

/// Compute the RHS in-place:
///     out = rhs(ν) = ν ⊙ ( g + Vν - Υ ),
///     where Υ = Σ_i ν_i ( g_i + (Vν)_i ).
///
/// Details:
/// - Purpose: Evaluates the replicator vector field without allocating.
/// - Parameters:
///   - `nu`: Current state.
///   - `growth_vector`: Growth vector `g`.
///   - `interaction_matrix`: Interaction matrix `V`.
///   - `w`: Scratch for `V nu`.
///   - `drift`: Scratch for centered drift.
///   - `out`: Destination derivative.
#[inline]
fn rhs_inplace(
    nu: &Array1<f64>,                 // current state ν (len d)
    growth_vector: &Array1<f64>,      // g (len d)
    interaction_matrix: &Array2<f64>, // V (d×d)
    w: &mut Array1<f64>,              // scratch: w = Vν
    drift: &mut Array1<f64>,          // scratch: drift = g + w - Υ
    out: &mut Array1<f64>,            // output: rhs(ν)
) {
    let d = nu.len();

    // (1) w = V · ν
    for i in 0..d {
        let mut acc = 0.0;
        for j in 0..d {
            acc += interaction_matrix[(i, j)] * nu[j];
        }
        w[i] = acc;
    }

    // (2) Υ = Σ_i ν_i (g_i + w_i)
    let mut upsilon = 0.0;
    for i in 0..d {
        upsilon += nu[i] * (growth_vector[i] + w[i]);
    }

    // (3) drift = g + w - Υ
    for i in 0..d {
        drift[i] = growth_vector[i] + w[i] - upsilon;
    }

    // (4) out = ν ⊙ drift
    for i in 0..d {
        out[i] = nu[i] * drift[i];
    }
}

/// One explicit RK4 step (deterministic) writing into `out`.
///
/// Details:
/// - Purpose: Advances one raw deterministic step without enforcing
///   mode-specific state invariants.
/// - Parameters:
///   - `nu`: Current state.
///   - `growth_vector`: Growth vector `g`.
///   - `interaction_matrix`: Interaction matrix `V`.
///   - `dt`: Step size.
///   - `sc`: Reusable RK4 scratch storage.
///   - `out`: Raw next-state destination.
#[inline]
fn rk4_step_inplace_raw(
    nu: &Array1<f64>,                 // current ν
    growth_vector: &Array1<f64>,      // g
    interaction_matrix: &Array2<f64>, // V
    dt: f64,                          // step size
    sc: &mut Rk4Scratch,              // scratch buffers
    out: &mut Array1<f64>,            // ν_next (raw)
) {
    let d = nu.len();
    let half_dt = 0.5 * dt;
    let dt_over_6 = dt / 6.0;

    // k1 = rhs(ν)
    rhs_inplace(
        nu,
        growth_vector,
        interaction_matrix,
        &mut sc.w,
        &mut sc.drift,
        &mut sc.k1,
    );

    // tmp = ν + 0.5*dt*k1
    for i in 0..d {
        sc.tmp[i] = nu[i] + half_dt * sc.k1[i];
    }
    // k2 = rhs(tmp)
    rhs_inplace(
        &sc.tmp,
        growth_vector,
        interaction_matrix,
        &mut sc.w,
        &mut sc.drift,
        &mut sc.k2,
    );

    // tmp = ν + 0.5*dt*k2
    for i in 0..d {
        sc.tmp[i] = nu[i] + half_dt * sc.k2[i];
    }
    // k3 = rhs(tmp)
    rhs_inplace(
        &sc.tmp,
        growth_vector,
        interaction_matrix,
        &mut sc.w,
        &mut sc.drift,
        &mut sc.k3,
    );

    // tmp = ν + dt*k3
    for i in 0..d {
        sc.tmp[i] = nu[i] + dt * sc.k3[i];
    }
    // k4 = rhs(tmp)
    rhs_inplace(
        &sc.tmp,
        growth_vector,
        interaction_matrix,
        &mut sc.w,
        &mut sc.drift,
        &mut sc.k4,
    );

    // out = ν + dt/6*(k1 + 2k2 + 2k3 + k4)
    for i in 0..d {
        let incr = dt_over_6 * (sc.k1[i] + 2.0 * sc.k2[i] + 2.0 * sc.k3[i] + sc.k4[i]);
        let mut val = nu[i] + incr;
        if !val.is_finite() || val <= 0.0 {
            val = 0.0;
        }
        out[i] = val;
    }
}

/// Integrate a single trajectory and persist aggregate signal output.
///
/// Details:
/// - Purpose: Runs one trajectory and writes signal snapshots to disk.
/// - Parameters:
///   - `gs_i`: Initial state consumed by the solver.
///   - `interaction_matrix`: Square interaction matrix `V`.
///   - `growth_vector`: Optional growth vector `g`; defaults to zero.
///   - `noise`: Optional post-step stochastic update.
///   - `dt`: Step size.
///   - `num_steps`: Number of integration steps.
///   - `save_interval`: Save every Nth step; `t = 0` is always saved.
///   - `output_path`: Directory for signal JSON output.
///   - `progress_counter`: Optional shared progress counter.
pub fn solve(
    gs_i: SystemState<f64>,                 // initial state (consumed)
    interaction_matrix: &Array2<f64>,       // V
    growth_vector: Option<&Array1<f64>>,    // g override
    noise: Noise,                           // noise model
    dt: f64,                                // step size
    num_steps: usize,                       // number of steps
    save_interval: usize,                   // save every N steps
    output_path: &Path,                     // signal output target
    progress_counter: Option<&AtomicUsize>, // optional progress counter
) -> Result<SystemState<f64>> {
    Ok(solve_with_termination(
        gs_i,
        interaction_matrix,
        growth_vector,
        noise,
        dt,
        num_steps,
        save_interval,
        output_path,
        progress_counter,
        TerminationConfig::disabled(),
    )?
    .final_state)
}

/// Integrate a single trajectory with explicit termination configuration.
pub fn solve_with_termination(
    mut gs_i: SystemState<f64>,             // initial state (consumed)
    interaction_matrix: &Array2<f64>,       // V
    growth_vector: Option<&Array1<f64>>,    // g override
    noise: Noise,                           // noise model
    dt: f64,                                // step size
    num_steps: usize,                       // number of steps
    save_interval: usize,                   // save every N steps
    output_path: &Path,                     // signal output target
    progress_counter: Option<&AtomicUsize>, // optional progress counter
    termination: TerminationConfig,         // explicit termination behavior
) -> Result<SolveOutcome> {
    let d = interaction_matrix.nrows(); // assumed square by caller / upstream validation
    if save_interval == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "save_interval must be >= 1",
        ));
    }

    // Own g for inner-loop reuse (avoid Option branches per step).
    let growth_vector_owned: Array1<f64> = growth_vector
        .map(|x| x.to_owned())
        .unwrap_or_else(|| Array1::zeros(d));

    // Enforce invariants at t=0.
    gs_i.sanitize();

    let mut gs_curr = gs_i;
    let mut signal_writer =
        SignalWriter::new(output_path, gs_curr.mode.clone(), SIGNAL_OUTPUT_FILE_SIZE)?;
    signal_writer.push(&gs_curr)?; // t=0 always saved

    if let Some(counter) = progress_counter {
        counter.store(0, Ordering::Relaxed);
    }

    // Pre-allocate the next-state buffer with the same mode as t=0.
    let mode0 = gs_curr.mode.clone();
    let mut gs_next = SystemState::empty(mode0, 0, d, None);

    // Scratch / noise context / RNG for the whole run.
    let mut sc = Rk4Scratch::new(d);
    let mut noise_ctx = NoiseContext::new(d);
    let mut rng = SmallRng::from_rng(&mut rand::rng());
    let mut termination_checker = TerminationChecker::new(termination)?;

    // Main loop: deterministic RK4 -> sanitize -> stochastic -> snapshot.
    let start_time = gs_curr.time;
    let mut steps_run = 0usize;
    let mut termination_reason = TerminationReason::MaxSteps;
    for step in 1..=num_steps {
        rk4_step_inplace_raw(
            &gs_curr.state,
            &growth_vector_owned,
            interaction_matrix,
            dt,
            &mut sc,
            &mut gs_next.state,
        );

        gs_next.sanitize();

        apply_noise_inplace(&mut gs_next, noise, dt, &mut noise_ctx, &mut rng);

        gs_next.time = start_time + step;

        // Advance current state and optionally save a snapshot.
        std::mem::swap(&mut gs_curr, &mut gs_next);
        steps_run = step;

        if step % save_interval == 0 {
            signal_writer.push(&gs_curr)?;
        }

        if let Some(counter) = progress_counter {
            counter.store(step, Ordering::Relaxed);
        }

        if let Some(checker) = termination_checker.as_mut() {
            if let Some(reason) = checker.check(&gs_curr, step) {
                termination_reason = reason;
                if step % save_interval != 0 {
                    signal_writer.push(&gs_curr)?;
                }
                break;
            }
        }
    }

    signal_writer.finish()?;

    Ok(SolveOutcome {
        final_state: gs_curr,
        steps_run,
        reason: termination_reason,
    })
}
