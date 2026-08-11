# GLV Scientific Workflow Refactor TODO

`design.md` is the architectural authority. This file records implementation
order and completion status for the clean-slate GLV crate on `sw-version`.

- Development branch: `sw-version`.
- Read-only reference branch: `legacy`.
- Read-only moved reference tree: `legacy/`.
- Never develop on, merge from, or retrofit the legacy branch or tree.

## Stage 0: lock the legacy reference

- [x] Rename the old local `master` branch to `legacy`.
- [x] Make `sw-version` the only development branch for the refactor.
- [x] Verify the moved `legacy/` tree byte-for-byte against the reference
  branch.
- [x] Fix reference commit
  `5ad7cad1ade361e4ee40e540db72d602565e15e8` in the baseline fixture.
- [x] Pass all 10 legacy library tests.
- [x] Compile all five legacy Cargo examples.
- [x] Record deterministic well-mixed replicator, spatial replicator, and
  spatial GLV final states.
- [x] Record monoculture termination and exact legacy signal/space sample
  iterations and counts.
- [x] Define `1e-12` absolute and relative comparison tolerances; keep
  iterations, termination decisions, and sample coordinates exact.
- [x] Record the intentional recording difference: Workflow completion adds a
  non-aligned final state exactly once, while legacy max-step recording omitted
  it.
- [ ] Move meaningful legacy production-module tests into integration tests as
  each corresponding subsystem is reimplemented.

## Stage 1: adopt the modern Rust package layout

- [x] Move `Cargo.toml` and `Cargo.lock` from the temporary `src/` package root
  to the repository root.
- [x] Use Cargo's conventional `src/lib.rs`, root `tests/`, and root
  `schemas/state.json` locations.
- [x] Use Rust's file-plus-directory module layout; do not introduce any
  `mod.rs` files.
- [x] Add `rust-toolchain.toml` pinned to Rust `1.97.1` with `rustfmt` and
  `clippy`.
- [x] Set Cargo `edition = "2024"` and `rust-version = "1.97"`.
- [x] Change the local Scientific Workflow dependency path to
  `../workflow/dev` after moving the package root.
- [x] Update package include/exclude rules so schemas and tests are included
  while `legacy/`, generated outputs, and targets are excluded.
- [x] Verify the relocated crate with formatting, tests, Clippy, and rustdoc
  under the pinned toolchain.

## Stage 2: preserve the canonical Workflow state foundation

- [x] Define one canonical schema with `abundance`, `space`, and `total`.
- [x] Use concrete payload types `Array1<f64>`, `Option<ArrayD<f64>>`, and
  `f64`.
- [x] Populate `space` for every state: `None` for non-spatial simulations and
  `Some(ArrayD<f64>)` for spatial simulations.
- [x] Define canonical field constants for `abundance`, `space`, and `total`.
- [x] Define canonical stream constants for `signal`, `space`, and
  `checkpoint`.
- [x] Test exact schema JSON round trips.
- [x] Test non-spatial and spatial state assembly through the shared schema.
- [x] Test coordinated tuple mutation across all three fields.
- [x] Test zero-copy ndarray insertion and extraction.
- [x] Relocate the implemented schema, constants, aliases, fixtures, and tests
  into the Stage 1 package layout without changing behavior.
- [x] Define the typed abundance representation label with
  `relative_frequency` and `absolute_count` metadata values.

## Stage 3: establish plugin module contracts

- [x] Add top-level `kernel.rs` and its `kernel/` directory.
- [x] Add `kernel/core.rs` for shared kernel and interaction behavior.
- [x] Define the `Kernel<A>` composition with private fields and an explicit
  deterministic algorithm contract.
- [x] Require kernels to own scratch, perform exactly one deterministic
  transition, and never perform configuration IO, noise, recording, progress,
  or time advancement.
- [x] Add top-level `noise.rs` and `noise/core.rs` for the swappable noise
  contract.
- [x] Add top-level `invariant.rs` and `invariant/core.rs` for the swappable
  invariant contract.
- [x] Keep plugin contracts narrow enough for static generic dispatch and
  external implementation without exposing internal state ownership.
- [x] Add compile-time and integration tests proving compatible plugins compose
  and incompatible state domains fail before evolution.

## Stage 4: implement interaction-matrix resolution and provenance

- [x] Define `InteractionMatrix` as a validated immutable
  `Arc<Array2<f64>>` plus provenance.
- [x] Define `KernelCore` ownership of the resolved matrix, species dimension,
  and provenance descriptor.
- [x] Validate square shape, species dimension, checked element count, and
  finite entries at construction.
- [x] Implement shared zero-allocation matrix application into caller-provided
  output storage.
- [x] Define the kernel evolution API around immutable `KernelStateView`,
  scratch-backed `KernelUpdate`, and a validated positive `TimeStep`.
- [x] Validate every proposed update before atomically committing any payload.
- [x] Add `kernel/source.rs` and the consuming `InteractionSource` contract.
- [x] Add a checked in-memory source for direct callers and tests.
- [x] Add a JSON source for inline values or a named file resolved from
  Scientific Workflow project configuration.
- [x] Add a typed generated source contract with generator identity, version,
  parameters, and explicit seed when stochastic.
- [x] Ensure `ScientificProject` and `TaskConfig` remain the only parsers of
  `fixed.json`, `sweep.json`, and `paths.json`; kernel sources consume resolved
  typed configuration.
- [x] Define versioned `glv.interaction-matrix.v1` row-major JSON artifacts.
- [x] Compute SHA-256 over exact artifact bytes and name artifacts by digest.
- [x] Persist each resolved matrix once beneath the execution scope's
  `inputs/` directory using atomic, collision-safe creation.
- [x] Reuse one artifact when multiple tasks resolve the same matrix.
- [x] Record format, shape, digest, execution-relative path, source kind, and
  generator provenance in task creation metadata.
- [x] Keep matrix values out of Workflow state payloads, checkpoints, and
  repeated task metadata.
- [x] Test inline, file, generated, shared, malformed, non-finite, wrong-shape,
  digest, and artifact-collision cases.

## Stage 5: implement the shared engine

- [x] Add top-level `engine.rs`; retain the agreed root `core.rs` for shared
  primitives and do not add `model.rs`.
- [x] Make the engine the sole owner of one authoritative Workflow
  `SystemState`.
- [x] Compose one kernel, one noise plugin, and one invariant policy through
  static generic dispatch.
- [x] Store the validated physical-time increment outside state payloads.
- [x] Expose immutable `state()` access and deliberate consuming
  `into_state()` transfer internally to concrete simulations.
- [x] Implement the shared step order: kernel, invariant, noise, invariant,
  then time advancement.
- [x] Advance iteration and physical time exactly once and only after all
  scientific mutations succeed.
- [x] Require fallible numerical work to complete in scratch before committing
  mutations that cannot be rolled back safely.
- [x] Add integration tests with minimal fake plugins that prove exact call
  order, single time advancement, error behavior, and sole state ownership.

## Stage 6: implement invariant and noise plugins

- [x] Implement aggregate frequency cutoff and normalization.
- [x] Implement per-cell local-frequency cutoff and normalization plus
  aggregate refresh.
- [x] Implement population feasibility, optional carrying-capacity enforcement,
  aggregate refresh, and total synchronization.
- [x] Use tuple mutable borrowing whenever an invariant coordinates
  `abundance`, `space`, and `total`.
- [x] Decide explicitly whether absolute population `total` preserves legacy
  rounding or becomes the exact aggregate sum; update `design.md` and fixtures
  if scientific behavior changes.
- [x] Implement zero-sized `NoNoise` as the deterministic default.
- [x] Implement demographic Gaussian noise with owned RNG and reusable scratch.
- [x] Implement proportional Gaussian noise with owned RNG and reusable
  scratch.
- [x] Keep noise independent from final invariant enforcement and time
  advancement.
- [x] Test invariant boundaries, non-finite inputs, cutoff, capacity, local
  simplex behavior, seeded noise reproducibility, and allocation reuse.
- [x] Keep exact stochastic continuation explicitly unsupported until a
  serializable RNG cursor or equivalent counter-based design is approved.

## Stage 7: implement deterministic kernel algorithms

- [x] Implement `MeanFieldReplicatorRk4` with model-owned RK4 stages and
  matrix-vector scratch.
- [x] Verify the well-mixed deterministic trajectory against the legacy
  fixture before proceeding.
- [x] Implement the shared spatial layout and boundary facilities required by
  spatial kernels.
- [x] Implement `SpatialReplicatorRk2` with reusable spatial scratch.
- [x] Verify spatial replicator abundance and space against the legacy fixture.
- [x] Implement `SpatialGeneralLotkaVolterraRk2` with reusable spatial scratch.
- [x] Verify spatial GLV abundance, space, and reviewed total semantics against
  the legacy fixture.
- [x] Preserve the legacy numerical operation order unless an intentional
  scientific change is separately reviewed and recorded.
- [x] Keep all kernel tests under integration-test targets rather than
  production files.

## Stage 8: expose concrete simulation APIs

- [x] Add top-level `simulation.rs` and `simulation/` implementation files.
- [x] Implement `MeanFieldReplicator` as a legal engine composition.
- [x] Implement `SpatialReplicator` as a legal engine composition.
- [x] Implement `SpatialGeneralLotkaVolterra` as a legal engine composition.
- [x] Keep concrete simulations limited to typed configuration, constructors,
  reconstruction validation, plugin wiring, defaults, and convenience
  accessors.
- [x] Provide construction from initial values and from reconstructed Workflow
  state.
- [x] Validate abundance representation, space presence, dimensions, kernel
  matrix dimension, and plugin compatibility before evolution.
- [x] Allow compatible alternate kernel and noise plugins through builders or
  defaulted generic parameters without making the internal engine the primary
  API.
- [x] Re-export all three concrete simulations at the crate root.
- [x] Verify normal user code imports concrete simulations directly from
  `general_lotka_volterra_rs`.

## Stage 9: integrate Workflow recording and terminal metadata

- [x] Define `signal` as `abundance` plus `total`.
- [x] Define `space` as `abundance`, `space`, and `total`.
- [x] Define `checkpoint` as all three canonical fields; non-spatial records
  encode `space` as `null`.
- [x] Build one `SystemStateWriter` per simulation run with independent typed
  stream intervals and nonzero byte limits.
- [x] Include model kind, abundance representation, resolved task parameters,
  and interaction artifact descriptor in creation-time metadata.
- [x] Observe the initial state once and every successful step
  unconditionally; remove all solver-side modulo sampling.
- [x] Complete successful recordings with the final state exactly once.
- [x] Commit termination reason and completed iteration as terminal metadata;
  create no GLV sidecar metadata file.
- [x] Use `CompletedRecording` for timing and stream record, chunk, and byte
  summaries.
- [x] Mark intentional simulation failures and leave unexpected interruption
  recoverable as a running recording.
- [x] Test independent intervals, final-state behavior, exact-byte chunks,
  bounded backpressure, checksums, failure lifecycle, matrix descriptors, and
  the single-metadata-file rule.

## Stage 10: implement reading and deterministic continuation

- [ ] Register direct Serde decoders for `Array1<f64>`,
  `Option<ArrayD<f64>>`, and `f64`.
- [ ] Replace legacy signal/space readers with `StoredStateSeriesReader`.
- [ ] Verify typed signal and space round trips with exact simulation times.
- [ ] Close Scientific Workflow's mandatory-integrity gap by verifying the
  selected latest sealed checkpoint chunk's byte count and SHA-256 checksum
  before enabling GLV continuation.
- [ ] Verify the referenced interaction artifact's exact bytes and SHA-256
  digest before reconstructing a kernel.
- [ ] Reconstruct the concrete simulation from complete checkpoint, original
  task configuration, and resolved interaction artifact.
- [ ] Reallocate numerical scratch without creating a second scientific state.
- [ ] Continue the same running recording without overwriting existing data.
- [ ] Verify uninterrupted and resumed deterministic runs produce the same
  final state and sample sequence.
- [ ] Do not claim exact stochastic continuation until RNG restart is part of
  the state or reproducible configuration contract.

## Stage 11: replace project orchestration and examples

- [ ] Give each runnable example conventional
  `config/{fixed,sweep,paths,state}.json` inputs.
- [ ] Replace Rust runtime constants with `ScientificProject` and complete
  `TaskConfig` decoding.
- [ ] Resolve or generate interaction matrices before simulation construction
  and persist their shared execution artifacts.
- [ ] Use `ExecutionScope`; never delete old output to make room for a run.
- [ ] Replace raw progress atomics with `ProgressReporter` and `TaskProgress`.
- [ ] Demonstrate sequential execution before enabling task-level Rayon
  parallelism over lazy task configurations.
- [ ] Keep each `main.rs` limited to project loading, scope creation, task
  preparation, concrete simulation orchestration, recording completion,
  validation, and plotting.
- [ ] Keep ground-truth comparison independent of persistence so it tests
  simulation steps directly.
- [ ] Update plotting and analysis tools for Workflow recordings and readers.
- [ ] Do not create a replacement GLV dispatcher while Scientific Workflow's
  generic dispatcher remains deferred.

## Stage 12: remove superseded infrastructure and release

- [ ] Confirm every replacement and equivalence test passes before deleting
  legacy-native concepts from the new crate API.
- [ ] Provide no `SignalWriter`, `SpaceWriter`, native series container,
  `TaskOutcome`, destructive output preparation, free `solve` dispatcher,
  generic `SystemState<T>`, `Mode<T>`, or `Scalar` compatibility layer.
- [ ] Add a symbol/search gate proving no legacy storage, metadata, progress,
  dispatcher, or generic-state API reappears.
- [ ] Remove dependencies unused by the clean implementation.
- [ ] Update README and rustdoc to the strict
  step/iteration/physical-time/sampling-interval vocabulary.
- [ ] Document configuration, matrix provenance, simulation construction,
  evolution, recording, reading, and deterministic continuation.
- [ ] Document the stochastic-continuation limitation.
- [ ] Run formatting, all targets, Clippy and rustdoc with warnings denied,
  numerical equivalence, artifact integrity, recording integrity,
  interruption, failure lifecycle, backpressure, and continuation tests.
- [ ] Run package verification and inspect the packaged file set.
- [ ] Replace the local Scientific Workflow path with its published version
  only when preparing the breaking release.
- [ ] Publish the breaking GLV version before migrating dispatcher.
- [ ] Reuse the proven engine/plugin/recording pattern for simulator.

## Project rules

- Work only on `sw-version`; use `legacy` and `legacy/` as read-only evidence.
- Follow `design.md`; update both documents when an architectural decision
  changes.
- Use the modern file-plus-directory module layout and no `mod.rs` files.
- Work on one production file at a time and wait for review before the next
  production file unless a larger batch is explicitly approved.
- Keep tests in integration-test targets, never in production module files.
- Preserve one authoritative Workflow state and one Workflow writer per
  running simulation.
- Keep configuration parsing, filesystem IO, recording, and progress out of
  numerical hot loops.
- Preserve exact interaction-matrix provenance without storing immutable
  matrices in evolving state or checkpoints.
- Do not add compatibility aliases or legacy recording readers.
- Use `step` for one numerical evolution action, `iteration` for the integer
  coordinate, `physical_time` for continuous modeled time, and
  `sampling_interval` for recording selection.
