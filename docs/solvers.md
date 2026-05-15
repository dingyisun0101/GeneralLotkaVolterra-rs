# Solvers

## Purpose

The solver layer owns numerical evolution. The current implemented solver is a
well-mixed replicator integrator with optional stochastic updates.

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

## File Layout

- `src/solvers/non_spatial/rk4.rs`: replicator RHS, RK4 step, and top-level
  trajectory solve.
- `src/solvers/non_spatial/noise.rs`: noise configuration and application.
- `src/solvers/non_spatial.rs`: non-spatial module surface.
- `src/solvers/spatial.rs`: reserved spatial module surface.
