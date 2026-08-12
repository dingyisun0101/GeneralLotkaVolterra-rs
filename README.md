# General Lotka–Volterra for Rust

`general-lotka-volterra-rs` provides concrete, composable implementations of:

- mean-field replicator dynamics with RK4;
- spatial local-frequency replicator reaction–diffusion with RK2; and
- spatial General Lotka–Volterra population reaction–diffusion with RK2.

Scientific Workflow owns the schema representation, simulation time, project
configuration, execution scopes, progress reporting, recording, integrity
checks, and reconstruction. This crate owns its canonical schema content, the
ecological equations, interaction matrices, deterministic kernels, noise
plugins, and invariants.

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
Workflow project whose root contains that configuration folder, supplies GLV's
crate-owned canonical state schema, expands every task, creates a
collision-resistant output scope,
constructs the selected model, evolves it, records it, verifies its final
checkpoint, and completes progress reporting.

Built-in templates are:

| Template | Scientific composition |
| --- | --- |
| `MeanFieldReplicator` | deterministic mean-field replicator with RK4 |
| `MeanFieldReplicatorDemographic` | RK4 mean-field replicator with demographic Gaussian noise |
| `SpatialReplicator` | local-frequency reaction–diffusion with midpoint RK2 |
| `SpatialGeneralLotkaVolterra` | absolute-population reaction–diffusion with midpoint RK2 |

Spatial templates consume categorical ecological lattices through the public
`ecological-initial-state` crate. GLV owns only their explicit conversion to a
species-last continuous field: spatial replicator sites become one-hot
frequency cells, while population GLV requires `initial_population_per_site`.
An initialization may be generated from shared configuration or loaded from a
verified content-addressed artifact produced by an earlier execution.

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
invariants, interaction matrices, recording adapter, ndarray boundary types,
PiP matrix/spatial/RNG configuration, Workflow prelude, and the
`GlvProjectTemplate` contract. Built-in templates are assembled from these same
components; they have no privileged model API.

The outer `run` function always owns project loading, task iteration, execution
scope creation, and the project-level reporter. A custom template owns only its
per-task scientific composition: it decodes the supplied `TaskConfig`, starts
and completes that task's `TaskProgress`, resolves scientific inputs, constructs
and steps the model, and delegates recording to `GlvRecording`. It should not
create another project, scope, reporter, task wrapper, or output-path system.

### Deterministic termination monitoring

Deterministic tasks may opt into synchronous early termination with a
`termination` object in `fixed.json` or `sweep.json`. The checker samples after
completed solver steps and is independent of recording cadence. It accepts a
fixed point only when every sample in each configured, non-overlapping
confirmation window has stable support, confined Jensen–Shannon composition,
optional stable mass, and an authoritative model RHS residual within the
configured absolute/state-relative scale. A separate optional detector can
classify a recurrent orbit only after multiple matching cycles with a
nontrivial within-cycle amplitude.

```json
"termination": {
  "start_after_iteration": 1000,
  "sample_interval_iterations": 10,
  "observable": "global_state",
  "fixed_point": {
    "base_window_samples": 16,
    "confirmation_window_multipliers": [1, 2, 4],
    "composition_tolerance": 1e-7,
    "relative_mass_tolerance": 1e-7,
    "mass_floor": 1e-12,
    "support_threshold": 1e-10,
    "residual_tolerance": {"absolute": 1e-10, "relative": 1e-8}
  },
  "oscillation": {
    "minimum_period_samples": 2,
    "maximum_period_samples": 128,
    "repeated_cycles": 3,
    "recurrence_tolerance": 1e-6,
    "minimum_cycle_amplitude": 1e-4
  }
}
```

Either detector may be omitted, but at least one is required. The observable
may be `global_state` or `spatial_field`. Monitoring stochastic simulations is
rejected rather than assigning deterministic convergence semantics to noise.
The public `termination` submodule contains the same no-I/O, bounded-history
`TerminationMonitor` used by the built-in runner, so downstream runners can use
the identical decision rule synchronously in their own step loop.

## Supported public API

This is the exhaustive GLV API allowlist. Ordinary users may use only `run`
and `GlvTemplate`, both exported by `prelude`. Advanced template authors may
also use the items below through `advanced::prelude`. Their public enum
variants and documented public methods are part of the same supported surface;
other compiler-visible implementation paths are not compatibility promises.

- State: `ABUNDANCE_FIELD`, `SPACE_FIELD`, `TOTAL_FIELD`, `SIGNAL_STREAM`,
  `SPACE_STREAM`, `CHECKPOINT_STREAM`, `AbundanceRepresentation`,
  `AggregateAbundance`, `SpatialAbundance`, `TotalAbundance`, `TimeStep`,
  `TimeStepError`, `load_state_schema`, and `state_schema_path`.
- Interaction: `InteractionMatrix`, `InteractionMatrixError`,
  `InteractionProvenance`, `InteractionSourceKind`, `GeneratorProvenance`,
  `InteractionArtifactDescriptor`, `InteractionArtifactError`,
  `InteractionArtifactLoadError`, `PersistedInteraction`,
  `ArtifactDisposition`, `INTERACTION_MATRIX_FORMAT`,
  `INTERACTION_MATRIX_METADATA_KEY`, `INTERACTION_GENERATOR_RNG_NAMESPACE`,
  `persist_interaction_matrix`, and `load_verified_interaction_matrix`.
- Spatial initialization: `SpatialInitialStateSource`,
  `ResolvedSpatialInitialState`, `SpatialInitializationError`, and
  `categorical_to_species_field`, plus the curated
  `ecological-initial-state` types reexported by `advanced::prelude`.
- Kernels: `Kernel`, `KernelAlgorithm`, `KernelCore`, `KernelResidual`,
  `KernelResidualError`, `KernelStateView`,
  `KernelUpdate`, `KernelCoreError`, `KernelStepError`, `KernelUpdateError`,
  `KernelAlgorithmError`, `MeanFieldReplicatorRk4`, `Diffusion`,
  `BoundaryCondition`, `SpatialReplicatorRk2`, and
  `SpatialGeneralLotkaVolterraRk2`.
- Noise: `Noise`, `NoiseAlgorithm`, `NoiseDomain`, `NoNoise`,
  `DemographicGaussian`, `ProportionalGaussian`, `NoisePluginError`,
  `NoiseStepError`, `DEMOGRAPHIC_GAUSSIAN_RNG_NAMESPACE`, and
  `PROPORTIONAL_GAUSSIAN_RNG_NAMESPACE`.
- Invariants: `InvariantPolicy`, `InvariantError`, `InvariantPolicyError`,
  `FrequencyInvariant`, `LocalFrequencyInvariant`, `PopulationInvariant`,
  `INVARIANT_TOLERANCE`, `validate_state`, and `enforce_state`.
- Models: `MeanFieldReplicator`, `MeanFieldReplicatorConfig`,
  `SpatialReplicator`, `SpatialReplicatorConfig`,
  `SpatialGeneralLotkaVolterra`, `SpatialGeneralLotkaVolterraConfig`,
  `SimulationKind`, `SimulationBuildError`, `DefaultSimulationBuildError`,
  and `StateAssemblyError`.
- Project execution: `GlvProjectTemplate`, `TemplateTaskError`, `GlvRunError`,
  `GlvProjectError`, `load_glv_project`, and `validate_glv_project`.
- Recording and reading: `GlvRecording`, `GlvRecordingError`,
  `GlvRecordingMetadata`, `RecordingMetadataError`, `TerminationReason`,
  `GlvCheckpointVerificationError`, `glv_json_decoders`,
  `open_completed_glv_recording`, `verify_completed_glv_checkpoint`,
  `ABUNDANCE_REPRESENTATION_METADATA_KEY`,
  `COMPLETED_ITERATION_METADATA_KEY`, `MODEL_KIND_METADATA_KEY`,
  `TASK_ORDINAL_METADATA_KEY`, `TERMINATION_REASON_METADATA_KEY`, and
  `TERMINATION_DIAGNOSTICS_METADATA_KEY`.
- Termination: `TerminationPolicy`, `TerminationMonitor`,
  `TerminationObservable`, `FixedPointTerminationConfig`,
  `OscillationTerminationConfig`, `ResidualTolerance`, `ConvergenceReason`,
  `FixedPointDiagnostics`, `OscillationDiagnostics`, `TerminationError`, and
  `jensen_shannon_distance`.
- Deliberate upstream reexports: ndarray's `Array1`, `Array2`, `ArrayD`,
  `Axis`, `IxDyn`, `ShapeError`, `arr1`, and `arr2`; PiP's `DenseMatrix`,
  `RngConfig`, `RngMethod`, and `SquareLatticeConfig`; and the complete
  `scientific_workflow::prelude` allowlist documented by Workflow.

Generated crate documentation is the exact signature reference for every item
in this list.

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
Individual projects do not copy this file and cannot override the model's
state contract. Workflow embeds the resolved schema in every recording.

## Interaction matrices

An interaction matrix is resolved and validated before simulation
construction. It can be constructed from:

- an in-memory PiP `DenseMatrix<f64>` or ndarray `Array2<f64>`;
- inline JSON already decoded by `TaskConfig`;
- a versioned JSON file resolved through project paths; or
- externally generated coefficients with explicit identity, version,
  parameters, and a resolved PiP `RngConfig` when stochastic.

Matrices are immutable `Arc<DenseMatrix<f64>>` values and use PiP's standard
versioned matrix JSON. For recorded runs,
`persist_interaction_matrix` writes canonical JSON once beneath the execution
scope's `inputs/` directory. Its SHA-256 digest, shape, format, path, and source
kind enter recording metadata; matrix values do not enter evolving states or
checkpoints. Scientific Workflow now owns the generic atomic publication,
content reuse, path containment, and digest-verification mechanics. GLV's
`interaction` module owns only ecological validation and provenance.

## Workflow project layout

Runnable simulation examples use the conventional project structure:

```text
examples/<model>/
├── Cargo.toml
├── README.md
├── config/
│   ├── fixed.json
│   ├── sweep.json
│   └── paths.json
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

## Dispatcher handoff

Dispatcher should treat GLV as an opaque project runner. Its adapter selects a
`GlvTemplate`, passes the stage's `config` directory to `run`, and retains the
returned `ExecutionScope` as the source of verified task recordings. It does
not supply a state schema, assemble models, interpret checkpoint internals, or
create another output path. Advanced GLV composition remains available through
`advanced::prelude`, but it is not part of the ordinary Dispatcher path.

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
