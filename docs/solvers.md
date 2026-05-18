# Solvers

## Purpose

The solver layer owns numerical evolution. The implemented solvers are a
well-mixed replicator integrator with optional stochastic updates, an
arbitrary-dimensional spatial GLV reaction-diffusion integrator, and an
arbitrary-dimensional spatial local-simplex replicator reaction-diffusion
integrator.

For application code, prefer the task runners in `src/tasks` as the stable
entrypoints. The lower-level solver functions are public so advanced users can
assemble custom workflows, but they expose more implementation detail, including
state ownership and save-cadence parameters.

## Replicator Step

The deterministic right-hand side is:

```text
d nu_i / dt = nu_i * (g_i + (V nu)_i - upsilon)
upsilon = sum_j nu_j * (g_j + (V nu)_j)
```

One integration step uses RK4 scratch buffers, writes a raw next state, and then
calls `SystemState::sanitize` before optional noise is applied.

The non-spatial solver exposes two entry points:

- `solve`: compatibility wrapper with termination disabled.
- `solve_with_termination`: returns `SolveOutcome` with the final state,
  steps run, `TerminationReason`, and signal/space writer stats.

## Noise Model

Noise is an optional post-step update:

- `NoiseKind::None`: no stochastic update.
- `NoiseKind::ProportionalGaussian`: multiplicative Gaussian perturbation with
  approximate mass projection before sanitization.
- `NoiseKind::DemographicGaussian`: additive Gaussian perturbation scaled by
  `sqrt(nu_i)`.

Every noise update ends at the same state boundary: `SystemState::sanitize`.

## Spatial RK4

The spatial solver evolves fields stored in `SystemState.space`. Spatial arrays
are species-last:

```text
space[x0, x1, ..., x{k-1}, i]
```

The GLV population right-hand side is:

```text
d n_i(x) / dt = n_i(x) * (g_i + sum_j V_ij n_j(x)) + D_i * laplacian(n_i)(x)
```

Diffusion uses a finite-difference Laplacian over all spatial axes with either
periodic or Neumann boundaries. The spatial replicator variant uses the local
replicator right-hand side and normalizes each spatial cell onto the simplex
after every raw RK4 step. The global `SystemState.state` vector is refreshed
from spatial totals or cell averages after each step.

Spatial solves have two save cadences:

- `save_signal_interval` stores the aggregate `SystemState.state` signal.
- `save_space_interval` includes the full `SystemState.space` field.

The initial sample at `t = 0` is written to both streams. Signal samples are
stored under `signal/`; full spatial snapshots are stored independently under
`space/`. The two streams flush JSON chunks independently against
`SIGNAL_OUTPUT_FILE_SIZE` and `SPACE_OUTPUT_FILE_SIZE`.

Spatial solver wrappers mirror the non-spatial termination split:

- `solve` and `solve_replicator`: compatibility wrappers with termination
  disabled.
- `solve_with_termination` and `solve_replicator_with_termination`: return
  `SolveOutcome`.

Task-level spatial runners use one save interval and pass it as both lower-level
spatial save cadences.

## Termination

Termination logic lives in `src/solvers/termination.rs` and is shared by
non-spatial and spatial solvers.

The main types are:

- `TerminationConfig`: user-selected checks, observable, tolerance, and
  `check_interval`.
- `TerminationReason`: `MaxSteps`, `Monoculture`, `FixedPoint`, or
  `OscillatorySteadyState`.
- `SolveOutcome`: final state plus stop metadata.
- `TerminationObservable`: `GlobalState` or `SpatialField`.
- `SteadyStateConfig`: off or adaptive fixed/oscillatory checks.

When termination is disabled, the solver pays only the construction-time check
and no per-step history cost. When enabled, checks run only on
`check_interval`. Monoculture uses `SystemState.state` and stops scanning once
two survivors are found. Steady-state detection stores a bounded history of the
configured observable and uses L-infinity distance.

## File Layout

- `src/solvers/mod.rs`: solver module surface.
- `src/solvers/non_spatial/mod.rs`: non-spatial module surface.
- `src/solvers/non_spatial/rk4.rs`: replicator RHS, RK4 step, and top-level
  trajectory solve.
- `src/solvers/non_spatial/noise.rs`: noise configuration and application.
- `src/solvers/spatial/mod.rs`: spatial module surface.
- `src/solvers/spatial/rk4.rs`: arbitrary-dimensional spatial GLV and
  local-replicator RK4 solvers.
- `src/solvers/termination.rs`: shared early-termination configuration and
  checker.
- `src/io/signal.rs`: aggregate signal output writer.
- `src/io/space.rs`: full spatial snapshot output writer.
