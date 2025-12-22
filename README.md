# Replicator Solver

A Rust implementation of a replicator-style dynamical system solver, designed for deterministic and stochastic (noise-driven) simulations with reproducible JSON outputs suitable for downstream analysis workflows.

This crate targets models of the general form
$$
\frac{d\nu_i}{dt}=\nu_i\left(f_i(\nu,\ldots)-\sum_j \nu_j f_j(\nu,\ldots)\right),
$$
with optional noise processes and (optionally) spatial extensions, while preserving the simplex constraints $\nu_i \ge 0$ and $\sum_i \nu_i = 1$ via standardized sanitization and renormalization routines.

## Highlights

- Deterministic and noisy dynamics (configurable noise models via `Noise` / `NoiseKind`)
- Simplex-safe state handling (sanitization, cutoff, renormalization)
- Structured outputs written as JSON tables for analysis pipelines
- Testable, reproducible simulations (fixed seeds where applicable)
- Designed for both unit-test driven development and parameter sweeps

## Repository layout

Typical layout (adapt as needed):

- `src/`
  - `solver/` or `src/solver.rs`: core time-stepping logic (e.g., `ReplicatorSolver`)
  - `noise/` or `src/noise.rs`: noise models and stochastic update rules
  - `ecosystem/` (optional): tracking tables and IO utilities
  - `tests/` (optional, internal module tests): Rust unit tests compiled with the crate
- `tests/`
  - Rust integration tests (recommended if you want `cargo test --test <name>`)
- `examples/`
  - Small runnable examples for common configurations
- `Cargo.toml`
