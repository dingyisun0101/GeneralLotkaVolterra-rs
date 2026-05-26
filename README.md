# General Lotka-Volterra

General Lotka-Volterra is a Rust crate for small ecological dynamical-system
experiments. The ready paths include well-mixed replicator dynamics,
deterministic spatial replicator reaction-diffusion, deterministic spatial GLV
reaction-diffusion, optional post-step stochasticity for well-mixed replicator
runs, automatic JSON output chunking, and early termination checks for
downstream analysis.

The crate is organized from live state to solver code to IO and runnable task
wrappers:

```text
system_state -> solvers -> io -> tasks -> examples
```

- `system_state` defines `SystemState`, representation modes, and invariants.
- `solvers` defines non-spatial integration, spatial reaction-diffusion
  integration, and stochastic update machinery.
- `io` defines dedicated signal and space JSON streams with automatic chunking.
- `tasks` wires solver calls into total-step experiments.
- `examples` provides minimal executable configurations.

## Examples

Run examples through Cargo:

```bash
cargo run --example replicator_deterministic
cargo run --example replicator_demographic
cargo run --example replicator_diffusive_deterministic
cargo run --example lv_diffusive_deterministic
```

The bundled examples take their settings from `examples/common/constants.rs`.
Users choose `TOTAL_STEPS` and a save interval; writers split output into
numbered JSON files under `output/<example>/signal/` and, for spatial runs,
`output/<example>/space/`:

- `replicator_deterministic`: deterministic well-mixed replicator run.
- `replicator_demographic`: replicator run with demographic Gaussian
  noise after each deterministic step.
- `replicator_diffusive_deterministic`: deterministic spatial
  local-simplex replicator reaction-diffusion run.
- `lv_diffusive_deterministic`: deterministic spatial GLV
  reaction-diffusion run.

Each example renders the full saved simulation history as its final output at
`output/<example>/plot/plot.png`. Activate a Python environment with `numpy`
and `matplotlib` available before running the examples, because Cargo invokes
the bundled Python renderer after the Rust simulation completes.

## Design Rules

Core consistency rules used across the crate:

- Runtime state lives in `SystemState<T>`. Solvers mutate this type and call
  `sanitize` at mode boundaries instead of duplicating feasibility logic.
- `Mode::Frequency` stores simplex states with mass one. `Mode::Population`
  stores absolute counts and may apply a carrying-capacity cap.
- Signal files store `time`, aggregate `state`, and `mass`. Space files store
  `time`, aggregate `state`, full `space`, and `mass`.
- Signal and space output streams chunk independently using the crate-level
  `SIGNAL_OUTPUT_FILE_SIZE` and `SPACE_OUTPUT_FILE_SIZE` budgets. Each stream
  computes a fixed samples-per-chunk count before stepping starts. Signal
  chunks default to 32 MiB; space chunks default to 1 GiB. A single oversized
  space sample is written alone.
- Each task writes `metadata.json` with requested steps, actual steps run,
  termination reason, save cadence, model dimensions, output budgets, and
  signal/space writer stats.
- Spatial task runners use one save interval for signal and space. Lower-level
  spatial solvers still support separate aggregate and full-field save cadences
  for custom workflows.
- Termination checks are explicit. Tasks receive a `TerminationConfig`; the
  examples enable monoculture termination and leave steady-state checks off.
- Non-spatial solvers keep reusable scratch buffers outside hot loops where
  practical.
- Non-spatial GLV task entry points are API placeholders until a dedicated
  non-spatial GLV right-hand side and integrator are introduced.

## State

Purpose:

`SystemState` is the live simulation state used by task runners and solvers.
JSON output lives under `io::signal` and `io::space`.

Core API:

```rust
SystemState::from_arrays(mode, time, state, space)
SystemState::empty(mode, time, num_taxa, space_shape)
SystemState::from_grid(mode, time, grid)
state.get(i)
state.set(i, value)
state.increase(i)
state.decrease(i)
state.sanitize()
SignalWriter::new(path, mode, SIGNAL_OUTPUT_FILE_SIZE)
SpaceWriter::new(path, mode, SPACE_OUTPUT_FILE_SIZE)
```

Core types:

- `Mode<T>`
- `SystemState<T>`
- `Scalar`
- `SignalRecord<T>`
- `SignalSeries<T>`
- `SpaceRecord<T>`
- `SpaceSeries<T>`

## Solvers

Purpose:

`solvers` owns numerical evolution. The non-spatial implementation is the
replicator solver:

```text
RK4 raw step -> sanitize -> optional noise -> snapshot
```

Core API:

```rust
solve(state, interaction_matrix, growth_vector, noise, dt, steps, save_interval, output_path, progress_counter)
solve_with_termination(..., termination)
Noise::none()
Noise::proportional_gaussian(sigma)
Noise::demographic_gaussian(sigma)
```

Core types:

- `Noise`
- `NoiseKind`
- `NoiseContext`
- `TerminationConfig`
- `TerminationReason`
- `SteadyStateConfig`
- `TerminationObservable`
- `SolveOutcome`

The spatial solver evolves `Mode::Population` fields with species stored on the
last axis:

```text
space[x0, x1, ..., x{k-1}, species]
```

Spatial core API:

```rust
solvers::spatial::rk4::solve(..., save_signal_interval, save_space_interval, ...)
solvers::spatial::rk4::solve_with_termination(..., termination)
solvers::spatial::rk4::solve_replicator(..., save_signal_interval, save_space_interval, ...)
solvers::spatial::rk4::solve_replicator_with_termination(..., termination)
solvers::spatial::rk4::Diffusion::unit_spacing(...)
solvers::spatial::rk4::Boundary::Periodic
solvers::spatial::rk4::Boundary::Neumann
```

## Tasks

Purpose:

`tasks` exposes experiment-level entry points. Callers provide `total_steps` and
`save_interval`; IO writers automatically split signal and space streams using
the crate-level `SIGNAL_OUTPUT_FILE_SIZE` and `SPACE_OUTPUT_FILE_SIZE` budgets,
which default to 32 MiB and 1 GiB respectively. Users do not configure
chunking behavior; writers compute fixed chunk sample counts before the solver
loop starts.

Task-level APIs also require an explicit `TerminationConfig`. Use
`TerminationConfig::disabled()` to run exactly to `total_steps`, or
`TerminationConfig::monoculture_only(save_interval)` for the cheap built-in
monoculture stop used by the examples.

Ready task entry points:

```rust
tasks::replicator_deterministic::run(...)
tasks::replicator_demographic::run(...)
tasks::replicator_diffusive_deterministic::run(...)
tasks::lv_diffusive_deterministic::run(...)
```

Ready task runners return `TaskOutcome` and persist the same run summary to
`metadata.json`. Before each task run, stale `signal/`, `space/`, and
`metadata.json` outputs under the target directory are removed so the directory
matches the latest run.

Placeholder task entry points:

```rust
tasks::lv_deterministic::run()
tasks::lv_demographic::run()
```

The GLV placeholders return `ErrorKind::Unsupported` until a dedicated GLV
right-hand side and integrator are introduced for non-spatial tasks.

## Documentation

Additional design notes live under `docs/`:

- [State](docs/state.md)
- [Solvers](docs/solvers.md)
- [Tasks](docs/tasks.md)
