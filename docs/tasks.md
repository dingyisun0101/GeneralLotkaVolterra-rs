# Tasks

## Purpose

Task modules expose experiment-level runners. They configure initial state,
choose a noise policy, and run solvers with dedicated signal/space output
streams.

## Run Contract

Each task run:

1. Creates the task-specific initial condition.
2. Runs the solver for up to `total_steps`.
3. Writes aggregate signal samples under `output_path/signal/`.
4. For spatial runs, writes full spatial snapshots under `output_path/space/`.
5. Writes `metadata.json` beside those output folders.

Callers do not choose `epoch_len` or `num_epochs`. Output writers estimate JSON
sample size and flush numbered files when a stream approaches its crate-level
file-size budget. Signal and space streams are chunked independently.

The default file-size budgets are:

```rust
general_lotka_volterra_rs::SIGNAL_OUTPUT_FILE_SIZE // 128 MiB
general_lotka_volterra_rs::SPACE_OUTPUT_FILE_SIZE  // 1 GiB
```

The estimate is conservative rather than exact. A terminal snapshot or a single
large spatial snapshot may make a file exceed the target size because preserving
the sample is more important than strict byte accounting.

Each ready task returns `TaskOutcome` and writes the same data to
`metadata.json`. The metadata includes requested steps, actual steps run,
termination reason, save cadence, model dimensions, cutoff/capacity settings,
chunk budgets, and signal/space writer stats. Task runners remove stale
`signal/`, `space/`, and `metadata.json` outputs before a run starts.

Well-mixed tasks use a uniform simplex initial condition. Spatial replicator
tasks use a uniform local simplex in every spatial cell. Spatial GLV tasks use a
uniform initial population density in every spatial cell and species.

## Ready Tasks

- `replicator_deterministic::run`: RK4 replicator dynamics without noise.
- `replicator_demographic::run`: RK4 replicator dynamics with demographic
  Gaussian noise.
- `replicator_diffusive_deterministic::run`: spatial local-simplex replicator
  reaction-diffusion without noise.
- `lv_diffusive_deterministic::run`: spatial GLV population
  reaction-diffusion without noise.

All ready task signatures follow the same high-level pattern:

```rust
run(..., dt, total_steps, save_interval, output_path, progress_counter, termination)
```

The return type is:

```rust
Result<TaskOutcome>
```

Spatial tasks include additional spatial setup arguments before `dt`, but still
use the same `total_steps` and single `save_interval` model.

Spatial task runners accept one `save_interval`; each saved spatial sample
writes to both the signal and space streams. Lower-level spatial solver APIs
still expose separate signal and space intervals for custom workflows.

## Termination

Ready task runners require an explicit
`solvers::termination::TerminationConfig`.

Useful constructors:

```rust
TerminationConfig::disabled()
TerminationConfig::monoculture_only(save_interval)
```

`monoculture_only` checks at the provided interval and stops when no more than
one strain/species remains above the cutoff or configured survivor tolerance.

Steady-state checks are opt-in through `SteadyStateConfig::Adaptive`. The
checker keeps bounded history and runs only on `check_interval`, so disabled
checks add essentially no runtime work. The observable can be `GlobalState` or,
for spatial models, `SpatialField`.

When a terminal condition occurs, the current state is saved even if the step is
not aligned with `save_interval`.

## Placeholder Tasks

- `lv_deterministic::run`
- `lv_demographic::run`

These return `ErrorKind::Unsupported` because the implemented solver stack is
replicator-form only for non-spatial tasks.

## File Layout

- `src/tasks/replicator_deterministic.rs`: deterministic well-mixed replicator
  task.
- `src/tasks/replicator_demographic.rs`: well-mixed replicator task with
  demographic Gaussian noise.
- `src/tasks/replicator_diffusive_deterministic.rs`: spatial local-simplex
  replicator task.
- `src/tasks/lv_diffusive_deterministic.rs`: spatial GLV population task.
- `src/tasks/metadata.rs`: task outcome and `metadata.json` writer.
