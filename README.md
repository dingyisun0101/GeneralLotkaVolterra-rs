# General Lotka–Volterra for Rust

`general-lotka-volterra-rs` provides concrete, composable implementations of:

- mean-field replicator dynamics with RK4;
- spatial local-frequency replicator reaction–diffusion with RK2; and
- spatial General Lotka–Volterra population reaction–diffusion with RK2.

Scientific Workflow owns state schemas, simulation time, project
configuration, execution scopes, progress reporting, recording, integrity
checks, and reconstruction. This crate owns the ecological equations,
interaction matrices, deterministic kernels, noise plugins, and invariants.

The minimum supported toolchain is Rust 1.97 with edition 2024.

## Concrete simulation API

Applications normally import one of the three concrete simulations directly:

```rust
use general_lotka_volterra_rs::kernel::{
    InMemorySource, InteractionSource,
};
use general_lotka_volterra_rs::{
    MeanFieldReplicator, MeanFieldReplicatorConfig, TimeStep,
};
use ndarray::{Array1, arr2};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let interaction = InMemorySource::new(arr2(&[
    [0.0, 0.4],
    [-0.3, 0.0],
]))
.resolve(2)?;
let config = MeanFieldReplicatorConfig::new(
    Array1::zeros(2),
    1e-12,
    TimeStep::new(0.005)?,
);
let mut simulation = MeanFieldReplicator::new(
    Array1::from_vec(vec![0.6, 0.4]),
    interaction,
    config,
)?;

simulation.step()?;
assert_eq!(simulation.state().simulation_time().iteration(), 1);
# Ok(())
# }
```

There is intentionally no generic GLV dispatcher. Applications orchestrate a
concrete simulation, which keeps invalid model/plugin combinations out of the
runtime API.

## Canonical state

Every model uses the schema in `schemas/state.json`:

| Field | Rust payload | Meaning |
| --- | --- | --- |
| `abundance` | `Array1<f64>` | Aggregate species abundance |
| `space` | `Option<ArrayD<f64>>` | `None` for mean-field models; a species-last array for spatial models |
| `total` | `f64` | Total abundance; frequency models use `1.0` |

The state also carries `SimulationTime`: an integer iteration and continuous
physical time. A successful `step()` advances each coordinate exactly once,
after kernel, invariant, noise, and final-invariant work succeeds.

Spatial population `total` preserves the historical rounded aggregate
convention. Spatial and aggregate arrays retain full floating-point values.

## Interaction matrices

An interaction matrix is resolved and validated before simulation
construction. Sources can be:

- an in-memory `Array2<f64>`;
- inline JSON already decoded by `TaskConfig`;
- a versioned JSON file resolved through project paths; or
- a typed generator with identity, version, parameters, and an explicit seed
  when stochastic.

Matrices are immutable `Arc<Array2<f64>>` values. For recorded runs,
`persist_interaction_matrix` writes canonical JSON once beneath the execution
scope's `inputs/` directory. Its SHA-256 digest, shape, format, path, and source
kind enter recording metadata; matrix values do not enter evolving states or
checkpoints.

## Workflow project layout

Runnable simulation examples use the conventional project structure:

```text
examples/<model>/
├── config/
│   ├── fixed.json
│   ├── sweep.json
│   ├── paths.json
│   └── state.json
├── inputs/
│   └── interaction.json
└── main.rs
```

`ScientificProject` parses all four documents. `task_configs()` expands the
sweep lazily, and each `TaskConfig` is decoded completely before numerical
construction. Relative paths are resolved against that example's project
root.

Each run creates a new collision-resistant `ExecutionScope`; existing output
is never deleted or overwritten. Examples execute tasks sequentially to make
the orchestration lifecycle explicit. Task-level parallelism can later consume
the same lazy task iterator without changing simulation internals.

Run an example with its checked-in project:

```sh
cargo run --example mean_field_replicator
cargo run --example mean_field_replicator_demographic
cargo run --example spatial_replicator
cargo run --example spatial_general_lotka_volterra
cargo run --example ground_truth_comparison
```

Pass another project root as the first argument to any Workflow-recording
example. The available example directories are the authoritative configuration
references.

## Recording and reading

`GlvRecording` configures one Scientific Workflow writer with three independent
streams:

| Stream | Fields | Intended use |
| --- | --- | --- |
| `signal` | `abundance`, `total` | Frequent aggregate analysis |
| `space` | all canonical fields | Spatial analysis |
| `checkpoint` | all canonical fields | Deterministic restart |

Orchestration observes the initial state and every successful step. Workflow
owns sampling intervals, bounded queues, exact-byte chunk rollover, JSONL
encoding, checksums, timing, and lifecycle metadata. Completion records the
final state exactly once, even when it is not aligned with an interval.

Use `open_completed_glv_recording` to obtain a verified
`StoredStateSeriesReader`. It can reconstruct an entire stream or only its
latest state. The examples reopen their completed checkpoint stream and verify
that it equals the simulation's final state.

The plotting helper validates declared byte counts and SHA-256 checksums before
exporting signal data:

```sh
python tools/plot_workflow_recording.py path/to/task-000000 \
  --csv signal.csv
```

Add `--plot signal.png` when Matplotlib is installed. CSV export uses only the
Python standard library and refuses to overwrite an existing destination.

## Noise and reproducibility

Built-in plugins are `NoNoise`, `DemographicGaussian`, and
`ProportionalGaussian`. Gaussian plugins own a ChaCha12 RNG and reusable
scratch. Their method, implementation version, key encoding, and exact seed
are written once as namespaced Workflow RNG-record metadata.

Workflow records RNG provenance but does not generate random values. Exact
stochastic continuation is deliberately unsupported because checkpoints do
not yet serialize an RNG cursor. Deterministic continuation verifies both the
checkpoint chunk and interaction artifact before reconstruction and appends to
the existing running recording.

## Validation

The direct `ground_truth_comparison` example compares small deterministic
systems against a dependency-free high-resolution reference integrator without
using persistence.

`validation/run.sh` runs every example and compares the refactored code against
legacy commit `5ad7cad1ade361e4ee40e540db72d602565e15e8`. Deterministic state
comparisons and same-seed demographic-noise comparisons use `1e-12` absolute
and relative tolerances. Validation outputs are written to new directories and
never replace earlier evidence.

## License

Licensed under either MIT or Apache-2.0, at your option.
