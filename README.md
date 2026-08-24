# General Lotka–Volterra for Rust

> This crate is pre-1.0 alpha software. Every release may contain intentional
> breaking changes; older releases should be treated as incompatible.

`general-lotka-volterra-rs` provides ecological models and their scientific I/O
as workloads for an application-owned Scientific Study. GLV handles
model construction, evolution, recording, terminal-state production, and final
integrity checks; the application owns task expansion, phases, scheduling, and
study execution.

The included models are:

- mean-field replicator dynamics with RK4;
- spatial local-frequency replicator reaction–diffusion with RK2; and
- spatial General Lotka–Volterra population reaction–diffusion with RK2.

## Quick start

Add the crate:

```toml
[dependencies]
general-lotka-volterra-rs = "0.13.0"
scientific-workflow = "0.7.2"
```

Copy the configuration, inputs, and `main.rs` from the example closest to your
study. Applications select a built-in GLV template and register its generated
tasks in a phase:

```rust,no_run
use general_lotka_volterra_rs::prelude::*;
use scientific_workflow::prelude::study::{Phase, Study};

# fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
let template = GlvTemplate::MeanFieldReplicator;
let workload = GlvWorkload::load("examples/mean_field_replicator", template)?;
let simulation = workload
    .register(Phase::builder(1, "mean-field replicator"))
    .build()?;
Study::builder(workload.record_path())
    .phase(simulation)
    .build()?
    .run_phases([1])?;
println!("results: {}", workload.execution().directory().display());
# Ok(())
# }
```

The GLV prelude exports the built-in template and GLV-aware study loader.
Scientific Workflow types are imported from their owning crate, making ownership explicit.
Paths, model parameters, sweeps, and recording settings remain in study files
rather than in Rust code.

Custom template authors may import GLV extension contracts from
`advanced::prelude`. Generic arrays, matrices, randomness, ecological data, and
Study types remain available from their owning crates rather than
being duplicated through GLV.

The minimum supported toolchain is Rust 1.97 with edition 2024.

## Choose a model

Built-in templates are:

| Template | Scientific composition |
| --- | --- |
| `MeanFieldReplicator` | deterministic mean-field replicator with RK4 |
| `MeanFieldReplicatorDemographic` | RK4 mean-field replicator with demographic Gaussian noise |
| `SpatialReplicator` | local-frequency reaction–diffusion with midpoint RK2 |
| `SpatialGeneralLotkaVolterra` | absolute-population reaction–diffusion with midpoint RK2 |

Start from one of the complete runnable examples:

- [mean-field replicator](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version/examples/mean_field_replicator)
- [mean-field replicator with demographic noise](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version/examples/mean_field_replicator_demographic)
- [spatial replicator reaction–diffusion](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version/examples/spatial_replicator)
- [spatial General Lotka–Volterra populations](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version/examples/spatial_general_lotka_volterra)

Each example is an independent crate and complete study. Run one from its
directory:

```sh
cd examples/mean_field_replicator
cargo run --release
```

Pass another compatible configuration directory as the optional argument:

```sh
cargo run --release -- /path/to/my-study/config
```

Spatial templates consume categorical ecological lattices through the public
`ecological-model-core` crate. GLV owns only their explicit conversion to a
species-last continuous field: spatial replicator sites become one-hot
frequency cells, while population GLV requires `initial_population_per_site`.
An initialization may be generated from shared configuration or loaded from a
verified content-addressed artifact produced by an earlier execution.

## Study configuration

A runnable study has this layout:

```text
my-study/
├── config/
│   ├── fixed.json
│   ├── sweep.json
│   └── paths.json
└── inputs/
    └── interaction.json
```

- `fixed.json` contains common model, observation, and recording settings.
- `sweep.json` defines task-varying parameter values.
- `paths.json` names interaction inputs and the recording output root. The
  `interaction.path_key` value in each resolved configuration selects which named
  interaction path it uses, so a sweep may pair different model sizes with
  different matrices inside one study.
- Path keys, path-resolved `initial_abundance`, and `recording_name` may contain
  scalar configuration placeholders such as `{sys_idx}` or `{energy}`. For
  example, `interaction_sys_{sys_idx}` selects `interaction_sys_0` for the
  configuration whose `sys_idx` is `0`. A `paths.json` entry may itself retain
  the same templated key and use placeholders in its path, covering any number
  of systems with one declaration. This keeps correlated artifact routing
  compatible with ordinary Cartesian sweep axes.
- `interaction.json` is the versioned PiP interaction matrix.

The examples are the configuration reference for their respective models.
Their READMEs explain every model-specific field. Common fields include
`maximum_iterations`, `physical_time_increment`, `observation`, and
`recording`. Spatial examples additionally show lattice shape,
initialization, diffusion, spacing, and boundary conditions.

The interaction matrix is the sole authority for species count. GLV ignores a
study-level `K` value, allowing it to remain useful metadata without becoming
duplicated model configuration. A scalar `growth` or `diffusion` value applies
to every inferred species. Mean-field models default to uniform initial
abundance when `initial_abundance` is absent. Spatial models default to unit
spacing when `spacing` is absent, and population GLV applies no global capacity
when `carrying_capacity` is absent.

Mean-field `initial_abundance` accepts either an inline vector or
`{ "path_key": "<key>" }`. The latter resolves a JSON vector through the
study's `paths.json`, allowing an upstream scientific-input phase to provide
the exact same aggregate frequencies used to construct a categorical lattice.
GLV validates and normalizes the resolved vector through its ordinary invariant
boundary; it does not infer or regenerate the lattice initialization.

Relative paths are resolved from the study root. There is no separate output
argument: `config/paths.json` is the sole path authority.

### Trajectory observation

Built-in deterministic templates expose trajectory-observation modes and
detector toggles:

```json
"observation": {
  "mode": "detect",
  "equilibrium": true,
  "periodic_orbit": false
}
```

Deterministic GLV defaults to `detect`. The stochastic demographic model must
explicitly select `terminal_only` or `disabled`. Observation cadence defaults
to the canonical `signal` writer cadence.

GLV owns the detector's bounded windows, tolerances, and confirmation schedule.
These are intentionally not ordinary study settings. The synchronous checker
runs inside the solver loop after complete steps, at the signal writer cadence.
Equilibrium acceptance requires stable
support, confined Jensen–Shannon composition, stable mass, and an authoritative
model RHS residual. A support change immediately starts a fresh confirmation
window and retains the transition sample. Periodic-orbit acceptance is separate
and requires repeated matching cycles with nontrivial amplitude.

Replicator dynamics receive one safe fast path: GLV checks iteration zero and
every later single-species state, including the last allowed step. A
single-species state is accepted immediately only when the model invariant
makes that support absorbing and the authoritative RHS residual also passes.
This shortcut applies to mean-field and spatial replicators, not generic
population GLV.

Unless observation is disabled, terminal-state production is independent of
automatic termination. Every successful observed built-in run embeds one
normalized `ecological.terminal-state.v1`
product in completed recording metadata and publishes the same document as
`terminal-state.json` inside that task's recording directory. GLV
samples global composition in a bounded internal window starting at iteration
zero and always forces the final state into that window. If GLV
accepted an equilibrium, the product contains the exact normalized final state,
has one represented sample, and is marked `equilibrium`. For every
other completion reason—including an iteration cap, periodic orbit, a request, or
a stochastic run—the product contains the normalized mean of the internal
trailing samples and is marked `trailing_average`. The classification is the
authoritative distinction; downstream code must not infer fixed-point status
from the vector alone. Disabled observation allocates no observer and emits no
terminal-state metadata or artifact.

The internal details are recorded for auditability but are not study
parameters. End users choose which outcomes to detect; GLV defines and applies
their evidence policy consistently.

## Outputs and terminal states

Every invocation creates a collision-resistant execution directory beneath the
configured recording root. Recording streams are explicit and model-specific:

| Stream | Intended use |
| --- | --- |
| `signal` | frequent aggregate abundance and total |
| `space` | optional full spatial state for spatial analysis |
| `checkpoint` | complete state for integrity checks and deterministic restart |

Every successful task also publishes a canonical terminal composition. Read it
through GLV's public API:

```rust,no_run
use ecological_model_core::terminal_state::TerminalClassification;
use general_lotka_volterra_rs::open_terminal_state;

# fn inspect() -> Result<(), Box<dyn std::error::Error>> {
let terminal = open_terminal_state("path/to/task-recording")?;
println!("composition: {:?}", terminal.composition());

match terminal.classification() {
    TerminalClassification::Equilibrium => {
        println!("GLV accepted an equilibrium");
    }
    TerminalClassification::AbsorbingState => {
        println!("the dynamics reached an absorbed state");
    }
    TerminalClassification::PeriodicOrbit => {
        println!("GLV detected a periodic orbit");
    }
    TerminalClassification::TrailingAverage => {
        println!("terminal vector is a trailing estimate");
    }
}
# Ok(())
# }
```

The execution directory also contains a directly inspectable
`terminal-state.json` inside the recording directory. This is a validated export of the canonical
recording metadata, not a separately computed result.

An accepted fixed point contains the exact normalized final composition. Any
other successful completion contains a bounded trailing average, including an
iteration cap, detected orbit, requested stop, or stochastic run. Always check
the classification; do not infer fixed-point status from the vector alone.

Use `open_accepted_fixed_point` when a downstream study must reject anything
except a GLV-accepted fixed point. Use `open_completed_glv_recording` for the
verified state streams.

## Model data

Every model uses the schema in `schemas/state.json`:

| Field | Rust payload | Meaning |
| --- | --- | --- |
| `abundance` | `Tensor<f64>` | Rank-one PiP tensor of aggregate species abundance |
| `space` | `Option<Tensor<f64>>` | `None` for mean-field models; a species-last PiP tensor for spatial models |
| `total` | `f64` | Total abundance; frequency models use `1.0` |

The state also carries an integer iteration and continuous physical time.

Spatial population `total` preserves the historical rounded aggregate
convention. Spatial and aggregate tensors retain full floating-point values.
Individual studies do not copy this file and cannot override the model's
state contract. Workflow embeds the resolved schema in every recording.

## Interaction matrices

Interaction matrices use PiP's standard versioned matrix JSON. Rows are
affected species and columns are contributing species. Matrix construction,
transformation, persistence, and multiplication are owned by
`ecological-model-core` and PiP; GLV consumes their `InteractionMatrix`
directly and adds no competing matrix wrapper. The matrix dimensions determine
the species count, so callers never supply it separately. GLV validates the
matrix against the selected model before evolving a task, preserves it as a
verified input artifact, and records its identity and digest with the output.

Spatial interaction is evaluated as one species-last batch. PiP owns the
bounded internal parallel work, and GLV compares this path against independent
per-cell scalar references in its correctness tests. Matrix application,
lattice Laplacians, and large elementwise tensor transforms use PiP's adaptive
Rayon execution; small workloads stay serial to avoid parallel overhead.

## Threading

GLV operations use the current Rayon pool through PiP and never create a pool
implicitly. Workflow applications should normally configure their existing
execution pool and call GLV normally, allowing task-level and numerical work to
share the same workers.

For direct use, GLV re-exports PiP's explicit convenience pool:

```rust,no_run
use general_lotka_volterra_rs::ComputePool;

let pool = ComputePool::new(8)?;
pool.install(|| {
    // Construct and step one or more GLV simulations here.
});
# Ok::<(), Box<dyn std::error::Error>>(())
```

`with_threads` is available for a single operation; a reusable `ComputePool`
avoids recreating workers across simulation steps. Neither control is part of
scientific inputs or serialized Workflow state.

Gaussian noise has a separate instance-local control:
`DemographicGaussian::with_max_threads` and
`ProportionalGaussian::with_max_threads` bound how many workers that one random
filler may occupy. They do not resize the surrounding pool or change seeded
random values.

## Recording and reading

Use `open_completed_glv_recording` to obtain a verified
`StoredStateSeriesReader`. It can reconstruct an entire stream or only its
latest state. `verify_completed_glv_checkpoint` performs the standard final
checkpoint comparison used by every example.

`open_accepted_fixed_point` succeeds only when the shared observer accepted an
`equilibrium`. It verifies that the terminal
diagnostics, completed iteration, and final checkpoint agree, then returns the
normalized final state that directly passed the configured residual test. It
does not apply an unrelated tail fraction or downstream extinction cutoff.

The plotting helper validates declared byte counts and SHA-256 checksums before
exporting signal data:

```sh
python tools/plot_workflow_recording.py path/to/task-000000 \
  --csv signal.csv
```

Add `--plot signal.png` when Matplotlib is installed. CSV export uses only the
Python standard library and refuses to overwrite an existing destination.

The separately packaged `general-lotka-volterra-reader` distribution composes
Workflow's official reader with GLV-owned NumPy decoders. It validates the
versioned PiP dense-tensor representation, model identity, abundance
interpretation, rank, shape, finiteness, and nonnegativity before exposing
contiguous arrays. Its payload fixture is serialized by a Rust conformance
test.

## Advanced composition

Most users should stay with `prelude::{GlvWorkload, GlvTemplate}` and copy
a complete study integration example. Researchers who genuinely need a new
model composition can import `advanced::prelude` and assemble workloads using
the same concrete models, kernels, noise plugins, invariants, interaction
facilities, recording adapter, and Workflow types used by the built-in
templates.

The advanced API exposes the shared `TrajectoryObserver` policies for custom
synchronous runners. Generated crate documentation is the signature reference
for this larger API; it is kept out of the quick-start path intentionally.

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

Every built-in simulation composition is compared at every step with an
allocation-naive scalar reference solver. These tests independently implement
matrix-vector products, RK4 or midpoint RK2, row-major lattice neighbors,
diffusion, cutoff and carrying-capacity invariants, aggregate synchronization,
and demographic-noise application. The stochastic comparison shares only the
documented seeded normal stream as external forcing:

```sh
cargo test --test naive_solver
```

Deterministic mean-field and spatial GLV trajectories are additionally
compared against checked-in values from an independent high-resolution RK4
reference implementation:

```sh
cargo test --test ground_truth
```

The dependency-free reference generator remains under `tests/ground_truth/`
for transparent fixture regeneration. Routine tests do not require Python.
Seeded-noise tests separately cover RNG provenance and exact same-seed
reproducibility.

## License

Licensed under either MIT or Apache-2.0, at your option.
