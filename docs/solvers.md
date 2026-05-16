# Solvers

## Purpose

The solver layer owns numerical evolution. The implemented solvers are a
well-mixed replicator integrator with optional stochastic updates, an
arbitrary-dimensional spatial GLV reaction-diffusion integrator, and an
arbitrary-dimensional spatial local-simplex replicator reaction-diffusion
integrator.

## Replicator Step

The deterministic right-hand side is:

```text
d nu_i / dt = nu_i * (g_i + (V nu)_i - upsilon)
upsilon = sum_j nu_j * (g_j + (V nu)_j)
```

One integration step uses RK4 scratch buffers, writes a raw next state, and then
calls `SystemState::sanitize` before optional noise is applied.

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

The initial sample at `t = 0` always includes both signal and space. Later
signal-only samples keep `space = None` to avoid writing full fields at every
signal step.

## File Layout

- `src/solvers/mod.rs`: solver module surface.
- `src/solvers/non_spatial/mod.rs`: non-spatial module surface.
- `src/solvers/non_spatial/rk4.rs`: replicator RHS, RK4 step, and top-level
  trajectory solve.
- `src/solvers/non_spatial/noise.rs`: noise configuration and application.
- `src/solvers/spatial/mod.rs`: spatial module surface.
- `src/solvers/spatial/rk4.rs`: arbitrary-dimensional spatial GLV and
  local-replicator RK4 solvers.
