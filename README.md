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

The current Workflow-native implementation and documentation live on the
[`sw-version` GitHub branch](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version).

## Ordinary users: one entry point

Ordinary applications import a deliberately two-item prelude, select a built-in
template, and point it at a conventional Workflow `config` directory:

```rust,no_run
use general_lotka_volterra_rs::prelude::*;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let execution = run(
    GlvTemplate::MeanFieldReplicator,
    "examples/mean_field_replicator/config",
)?;
println!("results: {}", execution.directory().display());
# Ok(())
# }
```

The ordinary prelude exports only `run` and `GlvTemplate`. `run` loads the
Workflow project whose root contains that configuration folder, validates its
state schema, expands every task, creates a collision-resistant output scope,
constructs the selected model, evolves it, records it, verifies its final
checkpoint, and completes progress reporting.

Built-in templates are:

| Template | Scientific composition |
| --- | --- |
| `MeanFieldReplicator` | deterministic mean-field replicator with RK4 |
| `MeanFieldReplicatorDemographic` | RK4 mean-field replicator with demographic Gaussian noise |
| `SpatialReplicator` | local-frequency reaction–diffusion with midpoint RK2 |
| `SpatialGeneralLotkaVolterra` | absolute-population reaction–diffusion with midpoint RK2 |

All scientific values and output locations remain in the Workflow project
documents. There is no separate output-path argument and therefore no second
path authority outside `config/paths.json`.

## Advanced users: custom templates

Authors who need a new scientific composition use the explicitly separate
advanced layer:

```rust,ignore
use general_lotka_volterra_rs::advanced::prelude::*;

struct MyTemplate;

impl GlvProjectTemplate for MyTemplate {
    fn name(&self) -> &str {
        "my_template"
    }

    fn run_task(
        &mut self,
        scope: &ExecutionScope,
        reporter: &ProgressReporter,
        task: TaskConfig,
    ) -> Result<(), TemplateTaskError> {
        // Decode the task, compose model/kernel/noise/invariant building
        // blocks, and use Workflow recording and progress directly.
        todo!()
    }
}

run(MyTemplate, "path/to/project/config")?;
```

`advanced::prelude` exposes the concrete models, kernels, noise plugins,
invariants, interaction sources, recording adapter, ndarray boundary types,
PiP spatial/RNG configuration, Workflow prelude, and the
`GlvProjectTemplate` contract. Built-in templates are assembled from these same
components; they have no privileged model API.

The outer `run` function always owns project loading, task iteration, execution
scope creation, and the project-level reporter. A custom template owns only its
per-task scientific composition: it decodes the supplied `TaskConfig`, starts
and completes that task's `TaskProgress`, resolves scientific inputs, constructs
and steps the model, and delegates recording to `GlvRecording`. It should not
create another project, scope, reporter, task wrapper, or output-path system.

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
- a typed generator with identity, version, parameters, and a resolved PiP
  `RngConfig` when stochastic.

Matrices are immutable `Arc<Array2<f64>>` values. For recorded runs,
`persist_interaction_matrix` writes canonical JSON once beneath the execution
scope's `inputs/` directory. Its SHA-256 digest, shape, format, path, and source
kind enter recording metadata; matrix values do not enter evolving states or
checkpoints. Scientific Workflow now owns the generic atomic publication,
content reuse, path containment, and digest-verification mechanics;
`scientific-interaction` owns only the matrix document and provenance schema.

## Workflow project layout

Runnable simulation examples use the conventional project structure:

```text
examples/<model>/
├── Cargo.toml
├── README.md
├── config/
│   ├── fixed.json
│   ├── sweep.json
│   ├── paths.json
│   └── state.json
├── inputs/
│   └── interaction.json
└── src/
    └── main.rs
```

Each directory is a complete crate and Workflow project that can be copied and
run independently. Its `main.rs` selects one `GlvTemplate` and passes its
`config` directory to `run`; it contains no model construction, task loop,
recording code, or custom configuration struct. Relative paths are resolved
against that example's project root.

Spatial models use PiP's `SquareLatticeConfig` as the sole owner of shape,
boundary condition, spacing, neighbor lookup, and Laplacian behavior. GLV's
`Diffusion` adds only the model-specific per-species coefficients. The
species-last ndarray state shape is derived from that lattice configuration and
the growth-vector length.

Each run creates a new collision-resistant `ExecutionScope`; existing output
is never deleted or overwritten. Built-in templates execute tasks sequentially.
Task-level parallelism can later consume the same lazy task iterator without
changing simulation internals.

The example READMEs focus on the governing equations, parameter meanings,
state interpretation, and usage:

- [mean-field replicator](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version/examples/mean_field_replicator)
- [mean-field replicator with demographic noise](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version/examples/mean_field_replicator_demographic)
- [spatial replicator reaction–diffusion](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version/examples/spatial_replicator)
- [spatial General Lotka–Volterra populations](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version/examples/spatial_general_lotka_volterra)

Run any copied example from its own directory:

```sh
cd examples/mean_field_replicator
cargo run --release
```

Within this repository, all examples can be checked together with
`cargo check --workspace`. Pass another compatible `config` directory as the
first argument to any example binary.

## Recording and reading

`GlvRecording` consumes Workflow's `Vec<StateStreamConfig>` directly. The
checked-in projects configure three independent streams without a GLV mirror
configuration type:

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
latest state. `verify_completed_glv_checkpoint` performs the standard final
checkpoint comparison used by every example.

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
`ProportionalGaussian`. Gaussian constructors accept only PiP's `RngConfig`.
They use PiP's `TensorRandFiller`, defaulting to ChaCha12 and one stream, and
retain reusable GLV proposal scratch. The fully resolved method, implementation
version, stream count, key encoding, and exact seed are written once as
namespaced Workflow `RngRecord` metadata.

Workflow records RNG provenance but does not generate random values. Exact
stochastic continuation is deliberately unsupported because checkpoints do
not yet serialize an RNG cursor. Deterministic continuation verifies both the
checkpoint chunk and interaction artifact before reconstruction and appends to
the existing running recording.

## Validation

Normal Rust tests compare deterministic mean-field and spatial trajectories
with and without diffusion against checked-in values from an independent
high-resolution RK4 reference implementation:

```sh
cargo test --test ground_truth
```

The dependency-free reference generator remains under `tests/ground_truth/`
for transparent fixture regeneration. Routine tests do not require Python.
Seeded-noise tests separately cover RNG provenance and exact same-seed
reproducibility.

## License

Licensed under either MIT or Apache-2.0, at your option.
