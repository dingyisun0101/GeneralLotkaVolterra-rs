use std::io::Result;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2};

use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::state::{SystemState, SystemStateTimeSeries};
use super::noise::{apply_noise_inplace, Noise, NoiseContext};

/// ==============================================================================================
/// =================================== RK4 Scratch Buffers ======================================
/// ==============================================================================================

/// Scratch buffers for RK4 (avoid repeated allocations).
///     Layout:
///         - k1..k4: RK4 stage derivatives
///         - tmp:   intermediate state
///         - w:     scratch for matrix-vector product (Vν)
///         - drift: scratch for drift term (g + Vν - Υ)
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


/// ==============================================================================================
/// =============================== Replicator RHS + RK4 Integrator ===============================
/// ==============================================================================================

/// Compute the RHS in-place:
///     out = rhs(ν) = ν ⊙ ( g + Vν - Υ ),
///     where Υ = Σ_i ν_i ( g_i + (Vν)_i ).
/// Notes:
///     - Conserves total mass analytically (replicator form).
///     - Numerical drift is corrected by `SystemState::sanitize()`.
#[inline]
fn rhs_inplace(
    nu: &Array1<f64>,                   // current state ν (len d)
    growth_vector: &Array1<f64>,        // g (len d)
    interaction_matrix: &Array2<f64>,   // V (d×d)
    w: &mut Array1<f64>,                // scratch: w = Vν
    drift: &mut Array1<f64>,            // scratch: drift = g + w - Υ
    out: &mut Array1<f64>,              // output: rhs(ν)
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
/// Notes:
///     - Does NOT enforce simplex / capacity constraints.
///     - Clamps non-finite / nonpositive outputs to 0 to prevent NaN propagation.
///     - Caller must call `SystemState::sanitize()` immediately afterward.
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

/// ==============================================================================================
/// ===================================== Top-Level Solve ========================================
/// ==============================================================================================

/// Integrate a single trajectory and persist a `SystemStateTimeSeries`.
///     Pipeline per step:
///         RK4 (raw) -> sanitize -> noise -> record snapshot.
/// Inputs:
///     - `gs_i`: initial state (will be sanitized in-place before t=0 is recorded)
///     - `interaction_matrix`: V (d×d)
///     - `growth_vector`: optional g (defaults to 0)
///     - `noise`: stochastic model applied after sanitize
///     - `save_interval`: save every Nth step (t=0 always saved)

pub fn solve(
    epoch: usize,                           // epoch index
    mut gs_i: SystemState<f64>,            // initial state (consumed)
    interaction_matrix: &Array2<f64>,       // V
    growth_vector: Option<&Array1<f64>>,    // g override
    noise: Noise,                           // noise model
    dt: f64,                                // step size
    num_steps: usize,                       // number of steps
    save_interval: usize,                   // save every N steps
    output_path: &Path,                     // time-series output target
    progress_counter: Option<&AtomicUsize>, // optional progress counter
) -> Result<SystemState<f64>> {
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

    let mut states: Vec<SystemState<f64>> =
        Vec::with_capacity(num_steps / save_interval + 1);
    let mut gs_curr = gs_i;
    states.push(gs_curr.clone()); // t=0 always saved

    if let Some(counter) = progress_counter {
        counter.store(0, Ordering::Relaxed);
    }

    // Pre-allocate the next-state buffer with the same mode as t=0.
    let mode0 = gs_curr.mode.clone();
    let mut gs_next = SystemState::empty(mode0, 0, d, None);

    // Scratch / noise context / RNG for the whole run.
    let mut sc = Rk4Scratch::new(d);
    let mut noise_ctx = NoiseContext::new(d);
    let mut rng = SmallRng::from_os_rng();

    // Main loop: deterministic RK4 -> sanitize -> stochastic -> snapshot.
    let start_time = gs_curr.time;
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
        if step % save_interval == 0 {
            states.push(gs_curr.clone());
        }

        if let Some(counter) = progress_counter {
            counter.store(step, Ordering::Relaxed);
        }
    }

    let mut ts = SystemStateTimeSeries::empty(epoch, gs_curr.mode.clone());
    for gs in &states {
        ts.add(gs);
    }
    ts.save(output_path)?;

    Ok(gs_curr)
}
