# General Lotka-Volterra

General Lotka-Volterra is a Rust crate for small ecological dynamical-system
experiments. The ready paths include well-mixed replicator dynamics, spatial
replicator reaction-diffusion, spatial GLV reaction-diffusion, optional
post-step stochasticity across supported solver domains, automatic JSON output
chunking, and early termination checks for downstream analysis.

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

The bundled examples are complete, standalone programs under
`examples/<name>/`. Each one keeps settings in its own `constants.rs`,
constructs the initial state and matrices in `main.rs`, calls the public
`solvers::solve` API, writes metadata, and renders a plot.

Run examples from this crate directory:

```bash
cargo run --example replicator_deterministic
cargo run --example replicator_demographic
cargo run --example replicator_diffusive_deterministic
cargo run --example lv_diffusive_deterministic
cargo run --example ground_truth_comparison
```

The examples are:

- `replicator_deterministic`: deterministic well-mixed replicator run.
- `replicator_demographic`: replicator run with demographic Gaussian
  noise after each deterministic step.
- `replicator_diffusive_deterministic`: deterministic spatial
  local-simplex replicator reaction-diffusion run.
- `lv_diffusive_deterministic`: deterministic spatial GLV
  reaction-diffusion run.
- `ground_truth_comparison`: deterministic correctness checks comparing this
  crate's public solver API against SciPy `solve_ivp` on small systems.

To customize an example, edit `examples/<name>/constants.rs`. Common knobs are
`TOTAL_STEPS`, `SAVE_INTERVAL`, `DT`, grid shape, cutoff, diffusion
coefficients, and noise strength. The examples intentionally do not share a Rust
`common` module, so each folder shows the complete end-user flow in one place.

Each run writes numbered JSON chunks under:

```text
output/<example>/signal/
output/<example>/space/   # spatial examples only
output/<example>/metadata.json
output/<example>/plot/plot.png
```

The Rust solve finishes before plotting starts. Plot rendering examples use the
bundled Python module `examples.plotting.render_from_output`, so activate a
Python environment with `numpy` and `matplotlib` before running those examples
end-to-end. Without those packages, the simulation output and metadata may still
be written, but the example exits with a plot-rendering error.

The `ground_truth_comparison` example does not render plots. It runs small
deterministic systems in Rust, generates SciPy references with
`examples/ground_truth_comparison/reference_scipy.py`, and compares final
states within explicit tolerances. It requires Python with `numpy` and `scipy`;
install those with:

```bash
python -m pip install -r examples/ground_truth_comparison/requirements.txt
```

To check that all examples compile without running simulations:

```bash
cargo check --examples
```

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
- Each Cargo example is self-contained and writes the same metadata shape after
  calling `solvers::solve` directly.
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

`solvers` owns numerical evolution. The top-level API is a single dispatcher:

```text
deterministic raw step -> sanitize -> optional noise -> sanitize -> snapshot
```

Core API:

```rust
use general_lotka_volterra_rs::prelude::*;

solve(
    state,
    interaction_matrix,
    growth_vector,
    SolveConfig {
        dynamics: Dynamics::Replicator,
        space: Space::None,
        noise: Noise::none(),
        dt,
        num_steps,
        save_signal_interval,
        output_path,
        termination,
    },
    progress_counter,
)

Space::spatial(&diffusion, save_space_interval)
Noise::none()
Noise::proportional_gaussian(sigma)
Noise::demographic_gaussian(sigma)
```

Applications can use the prelude to import the common solver, state,
termination, noise, spatial diffusion, metadata, output, and ndarray array
types in one line.

Core types:

- `Dynamics`
- `Space`
- `SolveConfig`
- `Noise`
- `NoiseKind`
- `NoiseContext`
- `TerminationConfig`
- `TerminationReason`
- `SteadyStateConfig`
- `TerminationObservable`
- `SolveOutcome`

Spatial runs use `Space::spatial(...)` and evolve fields with species stored on
the last axis:

```text
space[x0, x1, ..., x{k-1}, species]
```

Lower-level non-spatial RK4 and spatial RK2 modules remain public for custom
workflows, but task runners and normal application code use `solvers::solve`.

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
