# GLV Scientific Workflow Design

This document is the architectural authority for the completed clean-slate GLV
implementation on `sw-version`.

## Goals

- Make Scientific Workflow the sole owner of generic scientific state,
  simulation time, configuration, recording, reconstruction, execution scope,
  and progress behavior.
- Keep GLV responsible for ecological equations, numerical algorithms,
  invariants, stochastic updates, validation, and model-specific assembly.
- Provide concrete simulations as the primary user API while allowing kernels,
  noise algorithms, invariant policies, and interaction-matrix sources to be
  composed without changing the shared engine.
- Validate deterministic numerical behavior against independent
  high-resolution ground truth before changing scientific behavior.
- Make every resolved interaction matrix independently inspectable and exactly
  reproducible.
- Keep numerical loops allocation-aware and free of configuration parsing,
  filesystem IO, recording decisions, and terminal rendering.

## Non-goals

- No parallel GLV state, solver dispatcher, task API, or recording format.
- No second GLV storage, configuration, progress, or execution abstraction
  beside Scientific Workflow.
- No process-global runtime or writer manager.
- No interaction matrix, representation label, or immutable model parameter in
  the evolving Workflow state.
- No exact stochastic-continuation claim until RNG state has an explicit,
  serializable restart contract.

## Toolchain and package layout

The package uses Rust 2024 and Rust 1.97.1, the current stable toolchain when
this design was approved on 2026-08-10:

```toml
[package]
edition = "2024"
rust-version = "1.97"
```

`rust-toolchain.toml` pins `1.97.1` with the `rustfmt` and `clippy` components.
The pinned patch release makes local development and CI repeatable; the Cargo
`rust-version` records the language-level minimum within the 1.97 release.

The repository adopts Cargo's conventional package layout and Rust's modern
file-plus-directory module layout. Nested modules use `kernel.rs` plus
`kernel/*.rs`, never `kernel/mod.rs`. The production and publication layout is:

```text
glv/
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
├── README.md
├── design.md
├── schemas/
│   └── state.json
├── src/
│   ├── lib.rs
│   ├── core.rs
│   ├── engine.rs
│   ├── kernel.rs
│   ├── kernel/
│   │   ├── algorithms.rs
│   │   ├── algorithms/
│   │   │   ├── mean_field_replicator_rk4.rs
│   │   │   ├── spatial.rs
│   │   │   ├── spatial_general_lotka_volterra_rk2.rs
│   │   │   └── spatial_replicator_rk2.rs
│   │   ├── artifact.rs
│   │   ├── core.rs
│   │   └── source.rs
│   ├── noise.rs
│   ├── prelude.rs
│   ├── project.rs
│   ├── noise/
│   │   ├── algorithms.rs
│   │   ├── algorithms/
│   │   │   ├── demographic_gaussian.rs
│   │   │   ├── none.rs
│   │   │   └── proportional_gaussian.rs
│   │   └── core.rs
│   ├── invariant.rs
│   ├── invariant/
│   │   ├── core.rs
│   │   ├── policies.rs
│   │   └── policies/
│   │       ├── frequency.rs
│   │       ├── local_frequency.rs
│   │       └── population.rs
│   ├── reading.rs
│   ├── recording.rs
│   ├── simulation.rs
│   └── simulation/
│       ├── mean_field_replicator.rs
│       ├── spatial_general_lotka_volterra.rs
│       └── spatial_replicator.rs
├── examples/
│   ├── mean_field_replicator/
│   ├── mean_field_replicator_demographic/
│   ├── spatial_replicator/
│   └── spatial_general_lotka_volterra/
│       └── each example contains Cargo.toml, README.md, src/main.rs,
│           config/, and inputs/
├── tools/
│   └── plot_workflow_recording.py
└── tests/
    ├── continuation.rs
    ├── engine.rs
    ├── fixtures/
    │   └── ground_truth.json
    ├── ground_truth.rs
    ├── interaction_matrix.rs
    ├── invariants.rs
    ├── kernel_algorithms.rs
    ├── kernel_evolution.rs
    ├── noise_plugins.rs
    ├── plugin_contracts.rs
    ├── recording.rs
    ├── simulations.rs
    └── state_schema.rs
```

Independent reference-generation tooling remains beneath `tests/ground_truth/`;
routine Rust tests consume checked-in ground truth without requiring Python.

## Dependency boundary

Scientific Workflow owns:

- `SystemState`, `SystemStateSchema`, and `SimulationTime`;
- validation and persistence of RNG-agnostic `RngRecord` metadata;
- `ScientificProject`, `TaskConfig`, and project path resolution;
- `ExecutionScope` and task recording paths;
- `SystemStateWriter`, stream sampling, queues, chunking, checksums, and
  recording lifecycle;
- `StoredStateSeriesReader`, typed reconstruction, and state series;
- creation-time and terminal recording metadata; and
- `ProgressReporter` and `TaskProgress`.

GLV owns:

- engine composition and scientific step ordering;
- interaction, growth, diffusion, cutoff, carrying-capacity, and stochastic
  configuration;
- deterministic kernels and their reusable scratch storage;
- noise algorithms and RNG ownership;
- RNG selection, key creation, sampling, distribution transforms, and cursors;
- frequency, local-frequency, and population invariant policies;
- termination checks and model-specific outcomes;
- concrete simulation constructors and validation; and
- interaction-matrix resolution, validation, application, and provenance.

Recording and reporting remain outside the engine. A simulation exposes its
current state by immutable borrow; orchestration decides when to observe it.

## Workflow state contract

Every simulation uses the one canonical schema in `schemas/state.json`:

| Field | Rust payload | Meaning |
| --- | --- | --- |
| `abundance` | `Array1<f64>` | Aggregate abundance ordered by species index |
| `space` | `Option<ArrayD<f64>>` | Optional spatial abundance (`None` when non-spatial) |
| `total` | `f64` | Total abundance synchronized by the invariant policy |

All three slots are always populated. A non-spatial state stores a concrete
`Option<ArrayD<f64>>::None` rather than leaving the `space` slot empty. A full
checkpoint can therefore select every schema field and encode non-spatial
space as JSON `null`.

### Authoritative state and numerical scratch

The engine's `SystemState` is the only authoritative model state. Kernels and
noise plugins never retain another semantic snapshot of `abundance`, `space`,
`total`, or simulation time.

Numerical algorithms do own reusable scratch arrays. Scratch is uncommitted
working memory for Runge–Kutta stages, matrix products, random samples, and a
proposed next value. It may have the same shape as one state payload, but it is
not observable model history and is invalid as a state until its phase
succeeds. A `KernelStateView` only borrows the authoritative payloads for one
method call, and a `KernelUpdate` only borrows the algorithm's scratch until
`Kernel::step` validates and commits it.

Consequently, ordinary kernels do not duplicate every payload:

- non-spatial kernels propose `abundance` only;
- spatial kernels normally propose `space` only, after which an invariant
  refreshes aggregate `abundance` and `total`; and
- the coordinated `Both` update is reserved for algorithms that truly compute
  both abundance representations together.

The immutable abundance representation is creation-time configuration and
recording metadata:

- `relative_frequency`; or
- `absolute_count`.

It is not repeated in every evolving state. Concrete simulations validate that
their representation, space presence, dimensions, and invariant policy agree.

## Shared engine

`engine.rs` contains the implementation-level generic owner used by every
concrete simulation:

```rust,ignore
pub struct Engine<A, N, I> {
    state: scientific_workflow::SystemState,
    kernel: Kernel<A>,
    noise: Noise<N>,
    invariant: I,
    time_step: TimeStep,
}
```

The module is hidden from generated public documentation while integration
tests exercise it directly. Concrete simulations remain the intended user API;
they will wrap this generic owner and can later permit its visibility to be
tightened without changing model-facing behavior.

`Engine::new` takes the state and all three plugins by value, validates their
compatibility, and validates that the physical coordinate can advance. It is
the only constructor. These ownership rules may not change:

- One engine owns exactly one authoritative Workflow state.
- A kernel owns deterministic numerical scratch, not another complete state.
- A noise plugin owns its RNG and stochastic scratch.
- An invariant plugin synchronizes the three canonical payloads.
- Immutable configuration is not copied into state payloads.
- `state()` yields only an immutable borrow.
- `into_state()` deliberately transfers the sole state owner.

One successful step has this order:

```text
deterministic kernel
        ↓
enforce invariant
        ↓
apply noise
        ↓
enforce invariant
        ↓
advance iteration and physical time
```

The kernel and noise phases do not advance time. Time advances exactly once,
only after all scientific payload mutations for the step succeed. Numerical
algorithms must calculate fallible work into owned scratch before committing
state mutations wherever partial failure would otherwise leave an invalid
state.

The engine does not clone the entire Workflow state to make a multi-phase step
globally transactional. Each kernel, noise, and invariant phase is responsible
for completing its fallible calculations before committing its own mutation.
If a later phase fails, time is not advanced and no later phase runs, but an
earlier successful phase is not rolled back. Orchestration treats that result
as a failed simulation step and must not record it as a successful state.

Before the kernel runs, the engine checks iteration overflow, physical-time
presence, and finite physical-time addition without mutating the state. After
all four scientific phases succeed, it delegates the actual single advance to
`SystemState::advance_simulation_time`. Plugins never receive Workflow time, so
the preflight result cannot be invalidated during the step.

## Kernel subsystem

The public `kernel` module is a plugin subsystem. `kernel/core.rs` contains the
behavior shared by all kernels rather than duplicating matrix ownership,
validation, or multiplication in each integration algorithm.

### Kernel composition

The intended shape is:

```rust,ignore
pub struct Kernel<A> {
    core: KernelCore,
    algorithm: A,
}
```

`A` is a deterministic algorithm such as `MeanFieldReplicatorRk4`. The narrow
algorithm contract receives validated kernel facilities and a read-only view
of the authoritative abundance payloads. It computes one proposed transition
into its own reusable scratch and never receives mutable Workflow state,
configuration IO, stochastic updates, recording, progress reporting, total
abundance, or simulation time.

The implemented evolution boundary is:

```rust,ignore
pub trait KernelAlgorithm {
    type Error: Error + Send + Sync + 'static;

    fn validate(
        &self,
        core: &KernelCore,
        state: KernelStateView<'_>,
    ) -> Result<(), Self::Error>;

    fn compute<'algorithm>(
        &'algorithm mut self,
        core: &KernelCore,
        state: KernelStateView<'_>,
        time_step: TimeStep,
    ) -> Result<KernelUpdate<'algorithm>, Self::Error>;
}
```

`KernelUpdate` is an abundance-only, space-only, or coordinated update whose
views borrow the algorithm's scratch. `Kernel::step` computes from immutable
state, validates every proposed shape and finite value, and only then commits
all selected payloads. A rejected computation or update therefore leaves the
authoritative state unchanged. The kernel does not update `total` or advance
Workflow time. `TimeStep` structurally guarantees a finite, strictly positive
physical-time increment.

`KernelCore` owns:

- the resolved immutable interaction matrix;
- its dimensions and validated species count;
- its complete provenance descriptor; and
- shared zero-allocation matrix application into caller-provided output.

The interaction matrix is stored as an immutable `Arc<Array2<f64>>`. This lets
independent task kernels share one large matrix without cloning its allocation,
while generated per-task matrices retain the same ownership API.

Matrix construction validates:

- square shape;
- exact agreement with the simulation's species dimension;
- finite entries;
- checked element counts and shape conversion; and
- provenance sufficient to locate and verify the resolved artifact.

No matrix is read or generated in a hot loop.

### Deterministic algorithms

Concrete numerical implementations live privately under
`kernel/algorithms/`; `kernel` re-exports their public types so reorganizing
implementation files does not leak into user imports. The corresponding
concrete noise implementations live under `noise/algorithms/`, while invariant
implementations use `invariant/policies/` because they enforce state policy
rather than evolve it. Shared traits and wrappers remain in each subsystem's
`core.rs`.

`MeanFieldReplicatorRk4` owns four species-sized stage vectors, one temporary
state, matrix-product and drift scratch, and one proposed output. Its right-hand
side is the mean-field replicator equation
`nu_i * (g_i + (V nu)_i - sum_j nu_j * (g_j + (V nu)_j))`. One step uses
classical RK4 and proposes aggregate abundance only.

Both spatial algorithms share a species-last `SpatialLayout`, checked
row-major strides, `Diffusion`, and midpoint RK2 machinery. `Boundary` supports
periodic wrapping and zero-flux Neumann edges. Grid spacing is validated once
and cached as inverse squared spacing. Explicit diffusion steps expose and
enforce the conservative bound
`dt <= 1 / (2 * max(D) * sum_axis(1 / dx_axis^2))` when diffusion is nonzero.

`SpatialReplicatorRk2` evolves local replicator reaction plus diffusion;
`SpatialGeneralLotkaVolterraRk2` evolves local GLV population reaction plus
diffusion. They own
fixed-shape first-stage, midpoint, and proposed-output arrays, and propose only
the spatial payload. The following invariant phase refreshes aggregate
abundance and `total`. All inner sums, RK stages, neighbor visits, and final
updates retain a stable documented numerical operation order. Integration
tests compare deterministic endpoints to independently generated
high-resolution ground truth within reviewed method-specific tolerances.

### Interaction sources

`kernel/source.rs` defines the source contract. A source is consumed to produce
one resolved matrix and provenance:

```rust,ignore
pub trait InteractionSource {
    fn resolve(self, species: usize)
        -> Result<InteractionMatrix, InteractionSourceError>;
}
```

Implemented source families are:

- in-memory: an owned or already shared `Array2<f64>`;
- JSON: exact inline rows decoded by Workflow or a versioned matrix file at an
  already resolved project path; and
- generated: a typed generator with explicit algorithm identity, version,
  serializable parameters, and a randomness enum whose stochastic variant
  structurally requires a seed.

Scientific Workflow remains the configuration parser. `ScientificProject` and
`TaskConfig` decode `fixed.json`, `sweep.json`, and `paths.json` into typed GLV
source configuration. A kernel source consumes that resolved configuration; it
does not independently parse the project files.

Direct programmatic callers may supply an already-owned matrix through a
checked in-memory source. Tests use this path without filesystem setup.

### Resolved matrix persistence

The exact resolved matrix is persisted once even when it originated inline in
configuration or from a deterministic generator. Configuration describes the
request; the artifact proves the exact coefficients used.

The preferred location is a content-addressed execution input artifact:

```text
execution-.../
├── inputs/
│   └── interaction-<sha256>.json
├── task-000000/
└── task-000001/
```

The artifact format begins as versioned, row-major JSON:

```json
{
  "format": "glv.interaction-matrix.v1",
  "rows": 2,
  "columns": 2,
  "layout": "row_major",
  "values": [-0.4, 0.1, 0.05, -0.3]
}
```

The SHA-256 digest covers the exact artifact bytes. Persistence writes and
synchronizes a process-unique temporary file, then atomically publishes it as
a hard link without overwriting an existing digest path. Existing artifacts
are reused only when their exact bytes match; a mismatched digest-named file is
a hard collision error. Alternate encodings are deferred until JSON
performance or size measurements justify them.

Stage 4 provides a compact descriptor intended for each task recording's
creation-time metadata:

- format version;
- shape;
- SHA-256 digest;
- execution-relative artifact path;
- source kind; and
- for generators, generator identity, generator version, parameters, and
  seed.

The matrix is not a Workflow state field and is not repeated in checkpoints.
Its coefficients are also absent from task metadata; only the compact
descriptor is inserted. The kernel exposes borrowed matrix values and
provenance. Execution orchestration persists the shared artifact and, during
the later recording-integration stage, inserts its descriptor under the stable
`interaction_matrix` creation-metadata key before passing metadata to the
Workflow writer. The kernel does not write into a task recording directory.

## Noise subsystem

The public `noise` module contains interchangeable stochastic plugins.
`noise/core.rs` defines their shared contract and validated configuration.
Noise receives the same validated `TimeStep` as the kernel and cannot receive
a raw, zero, negative, or non-finite increment. A noise implementation must
complete fallible sampling in its owned scratch before mutating authoritative
payloads.
Implemented algorithms are:

- `NoNoise`, a zero-sized deterministic default;
- demographic Gaussian noise; and
- proportional Gaussian noise.

A Gaussian plugin owns a seeded `ChaCha12Rng`, its standard-normal
distribution, one species-sized normal-sample buffer, and one target-sized
proposal buffer. Its aggregate or exact spatial domain is fixed at
construction, so stepping neither resizes nor allocates scratch. It validates
the complete input and sampling scale before advancing the RNG, computes every
cell into the proposal buffer, and commits only after the complete proposal
succeeds.

Every `NoiseAlgorithm` explicitly returns either an immutable Workflow
`RngRecord` or `None` for deterministic behavior. Workflow provides only
the validated record format and metadata insertion/read interface; it contains
no RNG implementation. The built-in Gaussian plugins record distinct GLV
namespaces, method `chacha12+standard_normal`, implementation version
`rand_chacha-0.10+rand_distr-0.6`, key encoding `u64_be_hex`, and the exact
fixed-width seed value. Concrete simulations expose the selected plugin's
record to orchestration.

Proportional noise applies a mass-projected update proportional to local
abundance. Demographic noise scales fluctuations by square-root local
abundance and removes the weighted Gaussian mean. Both clamp nonpositive or
non-finite proposed values to zero and leave final feasibility restoration to
the following invariant phase. A noise plugin mutates only its selected
abundance payload, does not update `total`, does not advance time, and does not
enforce final invariants itself.

Stochastic checkpoint continuation is not exact until the noise subsystem
defines a serializable RNG cursor or adopts an equivalently reproducible
counter-based design. Until then, continuation tests and documentation are
explicitly deterministic-only.

## Invariant subsystem

The public `invariant` module contains policies independent of integration
algorithms:

- aggregate frequency normalization;
- per-cell local-frequency normalization plus aggregate refresh; and
- spatial population feasibility, optional carrying-capacity enforcement,
  aggregate refresh, and total synchronization.

`invariant/core.rs` defines the narrow policy contract. Policies use typed
Workflow tuple borrowing when `abundance`, `space`, and `total` must change
together. Configuration fixes species count, a finite nonnegative cutoff, and
an optional finite nonnegative carrying capacity before evolution. Spatial
policies require standard contiguous species-last storage and own fixed
species-sized aggregation scratch.

Aggregate frequency enforcement removes non-finite, nonpositive, and
below-cutoff entries, normalizes the remainder, and falls back to a uniform
simplex when nothing remains. Local-frequency enforcement performs the same
operation independently in every spatial cell, then stores the cell-average
species frequency in `abundance` and sets `total` to one. Population
enforcement treats spatial storage as authoritative, sanitizes it, applies an
optional global capacity scale, refreshes per-species aggregate abundance, and
synchronizes `total`.

Absolute population `total` uses an explicit rounded-sum convention. Spatial
values and per-species aggregate abundance remain exact
floating-point sums; only `total` is `round(exact_sum).max(0)`. Changing this
to an exact sum is a future scientific-behavior change requiring fixture and
design review.

## Concrete simulations

The public `simulation` module contains the important final API:

- `MeanFieldReplicator`;
- `SpatialReplicator`; and
- `SpatialGeneralLotkaVolterra`.

`lib.rs` re-exports these types at the crate root so normal orchestration uses:

```rust,ignore
use general_lotka_volterra_rs::MeanFieldReplicator;
```

Concrete simulation modules contain only:

- typed model-specific configuration;
- constructor and reconstruction validation;
- legal kernel, noise, and invariant composition;
- default plugin selections; and
- model-specific convenience accessors.

They do not contain matrix loading logic, numerical kernel implementations,
recording, progress reporting, or duplicated state lifecycle code.

Each concrete type has defaulted kernel and noise parameters while fixing its
invariant policy in the type itself:

```rust,ignore
pub struct MeanFieldReplicator<A = MeanFieldReplicatorRk4, N = NoNoise> {
    engine: Engine<A, N, FrequencyInvariant>,
}
```

`SpatialReplicator` similarly fixes `LocalFrequencyInvariant`, and
`SpatialGeneralLotkaVolterra` fixes `PopulationInvariant`. Callers therefore
cannot substitute
an invariant belonging to a different abundance domain. `from_plugins`
accepts alternate statically dispatched kernel and noise implementations; the
shared engine validates their state domains before ownership is accepted.

Typed `MeanFieldReplicatorConfig`, `SpatialReplicatorConfig`, and
`SpatialGeneralLotkaVolterraConfig` values group immutable growth, layout,
diffusion, cutoff, capacity, and `TimeStep` inputs. Resolved `InteractionMatrix` values remain
separate because their provenance and content-addressed persistence belong to
the shared kernel/input workflow rather than model-specific scalar
configuration.

Every concrete simulation provides:

- `new(initial, interaction, config)`, which moves initial arrays into a
  canonical iteration-zero Workflow state and wires the built-in deterministic
  kernel with `NoNoise`;
- `from_state`, which reconstructs built-in scratch around an existing
  Workflow state and validates recorded representation metadata; and
- immutable `state`, `time_step`, model-kind, representation, `step`, and
  consuming `into_state` accessors.

Spatial `new` constructors derive aggregate abundance and rounded `total` from
the authoritative initial spatial allocation. Construction checks
representation, space presence and exact shape, species dimensions, matrix
dimension, plugin domain, invariant consistency, physical time, and the
explicit diffusion stability limit before evolution. State insertion errors
retain ownership of any rejected payload through typed Workflow
`PayloadInsertError` variants.

`SimulationKind` supplies stable `mean_field_replicator`,
`spatial_replicator`, and `spatial_general_lotka_volterra` metadata values.

## Orchestration and recording

An application `main.rs` directly orchestrates concrete simulations:

```text
ScientificProject
        ↓
resolved TaskConfig
        ↓
interaction source → resolved matrix → content-addressed input artifact
        ↓
kernel + noise + invariant → concrete simulation
        ↓
ExecutionScope task path → GlvRecording → SystemStateWriter
        ↓
observe initial state
        ↓
simulation.step → writer.observe_state → progress update
        ↓
terminal decision
        ↓
complete with final state and terminal metadata
```

Every runnable model example supplies conventional
`config/{fixed,sweep,paths,state}.json` inputs. Scientific parameters are
decoded from `TaskConfig`; interaction files are resolved before construction
and persisted once per execution scope. Examples iterate lazy task
configurations sequentially and report through `ProgressReporter` and
`TaskProgress`. No replacement GLV dispatcher exists.

After completion, examples reopen the checkpoint stream through
`StoredStateSeriesReader` and compare it with the in-memory final state. The
plotting tool independently verifies chunk byte counts and SHA-256 digests
before CSV export or optional rendering. Ground-truth comparison remains
separate from persistence and steps concrete simulations directly.

`GlvRecording::start` creates exactly one Workflow writer and immediately
offers the initial state. Orchestration calls `observe_state` after every
successful simulation step without checking intervals. The Workflow writer
owns all sampling decisions, borrowed encoding, bounded queues, chunk rollover,
checksums, atomic metadata, and operational timing; GLV implements none of
those mechanisms.

Named streams are:

| Stream | Fields | Purpose |
| --- | --- | --- |
| `signal` | `abundance`, `total` | Frequent aggregate analysis |
| `space` | `abundance`, `space`, `total` | Spatial analysis |
| `checkpoint` | all three fields | Complete deterministic restart |

Each stream owns an independent `StreamRecordingConfig` containing a typed
`SamplingInterval`, nonzero chunk target, and nonzero queue-byte budget.
`GlvRecordingConfig` requires all three streams. Non-spatial `space` and
`checkpoint` records encode the populated `Option<ArrayD<f64>>::None` payload
as JSON `null`.

`GlvRecordingMetadata` combines stable `SimulationKind` and
`AbundanceRepresentation` values, exact resolved `TaskParameters` plus
`task_ordinal`, and the content-addressed `InteractionArtifactDescriptor` in
Workflow creation-time `user_metadata`. It also accepts the concrete
simulation's optional `RngRecord`, stores stochastic method/version/key
identity beneath Workflow's reserved `rng_records` object, and permits
additional non-colliding component namespaces. RNG identity is written
once in creation metadata, never in sampled state streams. Reserved-key or
RNG-namespace collisions and a representation incompatible with the
selected model fail before recording creation.

Successful completion consumes the writer, records the final state exactly
once through Workflow's terminal deduplication, and commits typed
`TerminationReason` plus `completed_iteration` as terminal metadata. It returns
`CompletedRecording` for Workflow-owned timing and stream record, chunk, and
exact-byte summaries. Intentional simulation failures transition to failed
metadata without recording an invalid state; dropping an interrupted
`GlvRecording` invents no terminal result, leaving the Workflow recording
running and recoverable. The recording directory contains Workflow's sole
`metadata.json` and no GLV sidecar.

## Reconstruction and continuation

`reading.rs` is a thin adapter over Scientific Workflow storage. It registers
direct Serde decoders for:

- `Array1<f64>` under `abundance`;
- `Option<ArrayD<f64>>` under `space`; and
- `f64` under `total`.

`open_completed_glv_recording` supplies that registry to
`StoredStateSeriesReader`. Scientific Workflow remains responsible for
completed-recording metadata validation, chunk byte-count and SHA-256
verification, JSON decoding, exact `SimulationTime` reconstruction, and typed
state-series assembly. GLV defines no separate signal reader, space reader,
record parser, or completed-recording integrity checker. Integration tests
cover aggregate and populated spatial round trips, non-spatial `None`, and
exact iteration and physical-time coordinates.

A deterministic continuation requires:

1. a verified complete checkpoint;
2. the original resolved task configuration;
3. the verified interaction-matrix artifact and descriptor;
4. reconstruction through the matching concrete simulation constructor; and
5. freshly allocated numerical scratch.

The latest selected sealed checkpoint chunk must pass byte-count and SHA-256
verification before reconstruction. Continuation appends to the same running
recording and must produce the same final state as uninterrupted deterministic
execution.

Scientific Workflow owns and enforces continuation integrity. Its
`continue_recording_from_latest_checkpoint` path verifies the selected latest
sealed checkpoint chunk's declared byte count and SHA-256 checksum before
decoding its final record or returning an append-capable writer. A recovered
open tail is decoded only after Workflow validates its complete JSONL prefix.
GLV neither selects integrity policy nor duplicates these checks.

`GlvRecording::continue_from_latest_checkpoint` rebuilds the exact original
writer configuration and delegates recovery, checkpoint reconstruction, and
append ownership to Workflow. Workflow rejects any difference in time-axis
metadata, resolved task/GLV metadata, or stream configuration. The returned
checkpoint is not observed a second time.

`load_verified_interaction_matrix` resolves the descriptor's normalized path
beneath the execution directory, verifies SHA-256 over the exact artifact
bytes, and only then decodes and validates the matrix. The matching concrete
simulation's existing `from_state` constructor consumes the complete Workflow
checkpoint, verified matrix, abundance representation, and original typed task
configuration. Reconstructing its kernel and plugins allocates fresh numerical
scratch while the checkpoint remains the sole authoritative scientific state.

An end-to-end deterministic test interrupts after a successfully observed
state, reopens the same recording, reconstructs the simulation, and confirms
that its final state and signal, space, and checkpoint sample sequences exactly
match uninterrupted execution. Exact stochastic continuation remains
explicitly unsupported until RNG restart state has an approved serializable or
counter-based contract.

## Error and validation principles

- Public constructors return typed, contextual errors rather than relying on
  debug assertions.
- Ownership-preserving Workflow insertion errors are not flattened before the
  rejected payload can be recovered or deliberately dropped.
- Matrix source, artifact, kernel, noise, invariant, state, and recording
  errors retain distinct context.
- Invalid dimensions, non-finite configuration, zero sampling intervals, and
  incompatible plugin combinations fail before evolution begins.
- Numerical hot loops contain no repeated validation already guaranteed by
  construction unless checking is required for scientific correctness.

## Verification gates

The checked-in ground-truth fixture is generated by a dependency-free,
fine-step classical RK4 reference implementation independent of production
kernels. Routine Rust tests compare mean-field and spatial GLV trajectories
with and without diffusion to those values. The generator is retained for
transparent regeneration but is not a test-time Python dependency.

- Deterministic abundance and space comparisons use reviewed tolerances tied
  to the reference step size and production integration order.
- Iterations and interval-selected sample coordinates are exact.
- Workflow completion always includes a non-aligned final state.
- Every production module receives integration-test coverage outside the
  production source file.
- Formatting, all targets, Clippy with warnings denied, rustdoc with warnings
  denied, recording integrity, interruption, continuation, and package-content
  checks are release gates.

## Design-change rule

Changes to ownership, module boundaries, step ordering, state fields, plugin
contracts, matrix provenance, persistence layout, or continuation guarantees
must update this document in the same reviewed change.
