# General Lotka-Volterra

General Lotka-Volterra is a Rust crate for small ecological dynamical-system
experiments. The current ready path is a well-mixed replicator solver with
deterministic RK4 integration, optional post-step stochasticity, and JSON epoch
outputs for downstream analysis.

The crate is organized from state storage to solver code to runnable task
wrappers:

```text
state -> solvers -> tasks -> examples
```

- `state` defines `SystemState`, representation modes, and epoch time series.
- `solvers` defines non-spatial integration, spatial reaction-diffusion
  integration, and stochastic update machinery.
- `tasks` wires solver calls into repeatable epoch-based experiments.
- `examples` provides minimal executable configurations.

## Examples

Run examples through Cargo:

```bash
cargo run --example replicator_deterministic
cargo run --example replicator_demographic
cargo run --example replicator_diffusive_deterministic
cargo run --example lv_diffusive_deterministic
```

The bundled examples write epoch JSON files under `output/`:

- `replicator_deterministic`: deterministic well-mixed replicator run.
- `replicator_demographic`: replicator run with demographic Gaussian
  noise after each deterministic step.
- `replicator_diffusive_deterministic`: deterministic spatial
  local-simplex replicator reaction-diffusion run.
- `lv_diffusive_deterministic`: deterministic spatial GLV
  reaction-diffusion run.

Each example renders the latest epoch as its final output at
`output/<example>/plot/plot.png`. Activate a Python environment with `numpy`
and `matplotlib` available before running the examples, because Cargo invokes
the bundled Python renderer after the Rust simulation completes.

## Design Rules

Core consistency rules used across the crate:

- Runtime state lives in `SystemState<T>`. Solvers mutate this type and call
  `sanitize` at mode boundaries instead of duplicating feasibility logic.
- `Mode::Frequency` stores simplex states with mass one. `Mode::Population`
  stores absolute counts and may apply a carrying-capacity cap.
- Time series files keep one shared `mode` per epoch and store owned snapshot
  records under `samples`.
- Non-spatial solvers keep reusable scratch buffers outside hot loops where
  practical.
- GLV task entry points are present as API placeholders, but the implemented
  solver stack is currently replicator-form.

## State

Purpose:

`state` is the common data model used by task runners, solvers, and JSON IO.

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
SystemStateTimeSeries::empty(epoch, mode)
time_series.add(&state)
time_series.save(path)
SystemStateTimeSeries::from(path)
```

Core types:

- `Mode<T>`
- `SystemState<T>`
- `SystemStateRecord<T>`
- `SystemStateTimeSeries<T>`
- `Scalar`

## Solvers

Purpose:

`solvers` owns numerical evolution. The non-spatial implementation is the
replicator solver:

```text
RK4 raw step -> sanitize -> optional noise -> snapshot
```

Core API:

```rust
solve(epoch, state, interaction_matrix, growth_vector, noise, dt, steps, save_interval, output_path, progress_counter)
Noise::none()
Noise::proportional_gaussian(sigma)
Noise::demographic_gaussian(sigma)
```

Core types:

- `Noise`
- `NoiseKind`
- `NoiseContext`

The spatial solver evolves `Mode::Population` fields with species stored on the
last axis:

```text
space[x0, x1, ..., x{k-1}, species]
```

Spatial core API:

```rust
solvers::spatial::rk4::solve(...)
solvers::spatial::rk4::Diffusion::unit_spacing(...)
solvers::spatial::rk4::Boundary::Periodic
solvers::spatial::rk4::Boundary::Neumann
```

## Tasks

Purpose:

`tasks` exposes experiment-level entry points that run one or more epochs and
persist each epoch to disk.

Ready task entry points:

```rust
tasks::replicator_deterministic::run(...)
tasks::replicator_demographic::run(...)
tasks::replicator_diffusive_deterministic::run(...)
tasks::lv_diffusive_deterministic::run(...)
```

Placeholder task entry points:

```rust
tasks::lv_deterministic::run()
tasks::lv_demographic::run()
```

The GLV placeholders return `ErrorKind::Unsupported` until a dedicated GLV
right-hand side and integrator are introduced.

## Documentation

Additional design notes live under `docs/`:

- [State](docs/state.md)
- [Solvers](docs/solvers.md)
- [Tasks](docs/tasks.md)
