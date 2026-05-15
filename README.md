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
- `solvers` defines non-spatial integration and stochastic update machinery.
- `tasks` wires solver calls into repeatable epoch-based experiments.
- `examples` provides minimal executable configurations.

## Examples

Run the crate with the selected example in `src/main.rs`:

```bash
cargo run
```

The bundled examples write epoch JSON files under `output/`:

- `examples::replicator_deterministic`: deterministic well-mixed replicator run.
- `examples::replicator_demographic`: replicator run with demographic Gaussian
  noise after each deterministic step.

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

`solvers` owns numerical evolution. The active implementation is the non-spatial
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

## Tasks

Purpose:

`tasks` exposes experiment-level entry points that run one or more epochs and
persist each epoch to disk.

Ready task entry points:

```rust
tasks::replicator_deterministic::run(...)
tasks::replicator_demographic::run(...)
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
