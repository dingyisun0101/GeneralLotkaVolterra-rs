# GLV Scientific Workflow Design

This document is the architectural authority for the clean-slate GLV
implementation on `sw-version`. `todo.md` records execution status; it must not
silently redefine this design.

The old implementation is preserved beneath `legacy/` and on the read-only
`legacy` branch. It is evidence for numerical and behavioral comparison, not a
source tree to incrementally modernize.

## Goals

- Make Scientific Workflow the sole owner of generic scientific state,
  simulation time, configuration, recording, reconstruction, execution scope,
  and progress behavior.
- Keep GLV responsible for ecological equations, numerical algorithms,
  invariants, stochastic updates, validation, and model-specific assembly.
- Provide concrete simulations as the primary user API while allowing kernels,
  noise algorithms, invariant policies, and interaction-matrix sources to be
  composed without changing the shared engine.
- Preserve deterministic legacy numerics within explicit tolerances before
  changing scientific behavior.
- Make every resolved interaction matrix independently inspectable and exactly
  reproducible.
- Keep numerical loops allocation-aware and free of configuration parsing,
  filesystem IO, recording decisions, and terminal rendering.

## Non-goals

- No compatibility aliases for the legacy GLV state, solver dispatcher, task
  API, or recording format.
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
`kernel/*.rs`, never `kernel/mod.rs`. The tree below is the planned layout
through the concrete-model stages; files beyond the completed stage may not
exist yet.

```text
glv/
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
├── design.md
├── todo.md
├── schemas/
│   └── state.json
├── src/
│   ├── lib.rs
│   ├── core.rs
│   ├── engine.rs
│   ├── kernel.rs
│   ├── kernel/
│   │   ├── artifact.rs
│   │   ├── core.rs
│   │   ├── source.rs
│   │   ├── source/
│   │   │   ├── generated.rs
│   │   │   └── json.rs
│   │   ├── spatial_glv_rk2.rs
│   │   ├── spatial_replicator_rk2.rs
│   │   └── well_mixed_replicator_rk4.rs
│   ├── noise.rs
│   ├── noise/
│   │   ├── core.rs
│   │   ├── demographic_gaussian.rs
│   │   ├── none.rs
│   │   └── proportional_gaussian.rs
│   ├── invariant.rs
│   ├── invariant/
│   │   ├── core.rs
│   │   ├── frequency.rs
│   │   ├── local_frequency.rs
│   │   └── population.rs
│   ├── simulation.rs
│   └── simulation/
│       ├── spatial_glv.rs
│       ├── spatial_replicator.rs
│       └── well_mixed_replicator.rs
└── tests/
    ├── engine.rs
    ├── fixtures/
    ├── interaction_matrix.rs
    ├── invariants.rs
    ├── kernel_evolution.rs
    ├── legacy_baseline.rs
    ├── noise_plugins.rs
    ├── plugin_contracts.rs
    └── state_schema.rs
```

`legacy/` remains outside the package and is excluded from packaging and normal
Cargo target discovery.

## Dependency boundary

Scientific Workflow owns:

- `SystemState`, `SystemStateSchema`, and `SimulationTime`;
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

`A` is a deterministic algorithm such as `WellMixedReplicatorRk4`. The narrow
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

Proportional noise applies the legacy mass-projected update proportional to
local abundance. Demographic noise scales fluctuations by square-root local
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

Absolute population `total` explicitly preserves the legacy rounded-sum
convention. Spatial values and per-species aggregate abundance remain exact
floating-point sums; only `total` is `round(exact_sum).max(0)`. Changing this
to an exact sum is a future scientific-behavior change requiring fixture and
design review.

## Concrete simulations

The public `simulation` module contains the important final API:

- `WellMixedReplicator`;
- `SpatialReplicator`; and
- `SpatialGlv`.

`lib.rs` re-exports these types at the crate root so normal orchestration uses:

```rust,ignore
use general_lotka_volterra_rs::WellMixedReplicator;
```

Concrete simulation modules contain only:

- typed model-specific configuration;
- constructor and reconstruction validation;
- legal kernel, noise, and invariant composition;
- default plugin selections; and
- model-specific convenience accessors.

They do not contain matrix loading logic, numerical kernel implementations,
recording, progress reporting, or duplicated state lifecycle code.

Default type parameters or builders may expose alternative compatible kernel
and noise plugins without exposing the crate-internal engine as the primary
user API. Illegal combinations must fail at construction, preferably through
types and otherwise through descriptive validation errors.

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
ExecutionScope task path → SystemStateWriter
        ↓
observe initial state
        ↓
simulation.step → writer.observe_state → progress update
        ↓
terminal decision
        ↓
complete with final state and terminal metadata
```

The writer owns sampling. Evolution code offers the initial state and every
successfully evolved state unconditionally.

Named streams are:

| Stream | Fields | Purpose |
| --- | --- | --- |
| `signal` | `abundance`, `total` | Frequent aggregate analysis |
| `space` | `abundance`, `space`, `total` | Spatial analysis |
| `checkpoint` | all three fields | Complete deterministic restart |

Each stream owns an independent typed `SamplingInterval`. Completion records a
non-aligned final state exactly once. Termination reason and completed
iteration are terminal metadata in the Workflow recording; GLV creates no
second metadata sidecar.

## Reconstruction and continuation

Readers register direct Serde decoders for:

- `Array1<f64>` under `abundance`;
- `Option<ArrayD<f64>>` under `space`; and
- `f64` under `total`.

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

The fixed legacy reference is commit
`5ad7cad1ade361e4ee40e540db72d602565e15e8`. The checked-in fixture covers
well-mixed replicator, spatial replicator, spatial GLV, early termination, and
legacy sampling coordinates.

- Deterministic abundance and space comparisons use `1e-12` absolute and
  relative tolerances unless a reviewed kernel-specific tolerance replaces
  them.
- Iterations, termination decisions, and interval-selected sample coordinates
  are exact.
- Workflow completion intentionally adds a non-aligned final state that legacy
  max-step recording omitted.
- Every production module receives integration-test coverage outside the
  production source file.
- Formatting, all targets, Clippy with warnings denied, rustdoc with warnings
  denied, recording integrity, interruption, continuation, and package-content
  checks are release gates.

## Design-change rule

Changes to ownership, module boundaries, step ordering, state fields, plugin
contracts, matrix provenance, persistence layout, or continuation guarantees
must update this document and `todo.md` in the same reviewed change.
