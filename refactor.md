# GLV Refactor for Ecological Model Core

## Goal

Refactor GLV onto the clean-slate `ecological-model-core` crate. GLV keeps its
native ndarray computational state and model-specific ODE/PDE behavior. It no
longer owns shared categorical initialization, ecological interactions,
trajectory classification, or terminal scientific products.

No legacy GLV interaction, termination-monitor, terminal-state, configuration,
recording, or reader API will be retained.

## Dependency and ownership changes

- Add the new public `ecological-model-core` dependency.
- Remove the `ecological-initial-state` dependency.
- Remove shared interaction implementation from GLV.
- Remove shared terminal-state implementation from GLV.
- Remove GLV's separate `TerminationMonitor` and `TerminalStateMonitor`.
- Do not re-export legacy module paths as compatibility aliases.
- Keep ndarray as GLV's native ODE/PDE storage.
- Keep PiP interaction storage behind the shared core matrix.

GLV owns a canonical workload directory containing its fixed values, sweeps,
paths, scientific inputs, writer settings, and output instructions. Loading a
workload directory resolves configuration once and returns runtime-ready GLV
tasks. Both the standalone examples and Dispatcher call this same public entry
point; neither duplicates GLV decoding or writer construction.

One project represents every model in a GLV phase. A singleton `K` may live in
`fixed.json`; multiple values belong in `sweep.json`. Each resolved task also
selects its interaction input through `interaction.path_key`, allowing the
project to pair each `K` with its correctly dimensioned matrix without
per-`K` project directories.

Dispatcher treats the directory as opaque. It supplies the path, receives the
tasks, and registers them. GLV continues to own model assembly, recording,
terminal products, checkpoint validation, and task progress.

GLV continues to own:

- canonical GLV Workflow state and schema;
- ndarray conversion from categorical initial states;
- equation kernels, integrators, invariants, and noise;
- authoritative RHS/PDE residual computation;
- model/template selection;
- recording streams, checkpoint continuation, and runtime task behavior;
- model-specific stop reasons and fixed-point extraction policy.

Recording construction uses Workflow's unified schema-source builder. New
recordings pass their constructed live state; continuation passes the canonical
schema before checkpoint reconstruction. GLV never asks an orchestrator to
construct its writer.

## Initialization migration

- Replace imports with `ecological_model_core::initial_state`.
- Use the shared generated-or-verified source contract directly.
- Keep categorical-to-species-last ndarray conversion in GLV.
- Persist the new core descriptor and RNG provenance in GLV recording metadata.
- Remove old source-shape compatibility code and configuration forms.

## Interaction migration

- Replace `crate::interaction` ownership with
  `ecological_model_core::interaction`.
- Move any ndarray convenience conversion into a small GLV-local adapter if it
  remains necessary.
- Keep all equation interpretation and kernel validation in GLV.
- Update recording metadata and public APIs to use the core descriptor type.
- Update examples and tests to import the core type where appropriate.
- Consume core matrix recipes in tests/examples that generate ecological
  interactions; do not recreate recipe mathematics in GLV.

## Trajectory configuration

GLV's default is `Detect`. Omitted trajectory configuration resolves to the
standard equilibrium and periodic-orbit policies migrated from the existing
scientific detector.

The default observation interval is derived from the resolved canonical
abundance/signal stream. It is not hard-coded separately. An explicit policy
may override it.

Supported explicit modes are:

- `disabled`;
- `terminal_only`;
- `detect`.

Stochastic GLV cannot use equilibrium or periodic-orbit detection. A
stochastic template/project must explicitly select `terminal_only` or
`disabled`; task preparation fails clearly otherwise. No silent fallback is
allowed.

`Disabled` must avoid observer allocation, state normalization, residual
evaluation, terminal metadata, and terminal artifact publication.

## Observation adapter

After initial state construction and after every completed GLV step, the task
builds a borrowed core observation:

- `Continuous` for relative-frequency, density, and continuous-population
  models;
- aggregate abundance as terminal composition input;
- aggregate abundance as the default detector observable;
- a borrowed contiguous flattened spatial field when spatial detection is
  configured;
- current iteration and physical time.

No ndarray or `SystemState` is cloned. Off-cadence calls must not copy arrays.
Retained samples are copied once into core-owned reusable history.

## Equilibrium evidence

GLV supplies authoritative deterministic evidence only when the observer says
the current retained sample requires it.

The first implementation should use the existing allocation-free
`maximum_scaled_residual` calculation through
`MaximumScaledResidual`. It must use the exact observer tolerances and exact
submitted state. A later kernel may expose a borrowed residual vector without
changing the observer contract.

The core observer owns window, support, mass, and residual acceptance. GLV owns
the vector field and whether a residual is available.

Replicator single-support shortcuts may submit `AbsorbingState` only where the
model invariant proves no future evolution. Population GLV must not inherit
that assertion automatically.

## Periodic-orbit terminology

Replace `Oscillation` public terminology with `PeriodicOrbit` throughout
configuration, diagnostics, stop reasons, metadata, documentation, and tests.
Do not call the numerical result a limit cycle or neutral cycle.

GLV task execution stops when the observer emits an accepted equilibrium or
periodic orbit. Workflow target iteration is shortened to the detected
iteration exactly as today.

## Finalization and recording

Every successful observed run calls the consuming core `finish` method:

- detected equilibrium;
- detected periodic orbit;
- absorbing state where applicable;
- maximum iterations;
- requested/model-specific successful stop.

The returned core `TerminalState` is placed in GLV terminal metadata and may be
published as the task terminal artifact. Periodic-orbit output uses complete
detected cycles. Capped output uses the bounded trailing composition window.

When observation is disabled, GLV still records completed iteration and its
model-specific stop reason but omits terminal-state metadata and terminal
artifact publication.

Fixed-point extraction is updated to consume the new equilibrium terminal
classification and diagnostics. It remains GLV-specific because it validates
the GLV checkpoint and representation.

## Public API cleanup

- Remove GLV-owned interaction and terminal-state definitions.
- Remove `TerminationMonitor`, `TerminationObservable`, and oscillation-named
  exports.
- Export GLV trajectory configuration only where it adds model-specific
  defaults or validation.
- Refer users to `ecological-model-core` for shared scientific types.
- Update the ordinary prelude, advanced API, README, architecture docs, and
  all examples without compatibility aliases.

## Python and persisted output

- Update GLV's reader for the new terminal product and metadata shape.
- Remove old terminal-format decoding.
- Preserve native GLV ndarray payload decoding and checkpoints.
- Add reader coverage for equilibrium, periodic orbit, trailing average, and
  absent terminal product in disabled mode.

## Tests and verification

- Port all existing equilibrium and recurrence ground-truth tests to the core
  observer or GLV adapter as appropriate.
- Verify default deterministic GLV uses `Detect`.
- Verify cadence derives from the abundance writer.
- Verify stochastic GLV rejects implicit/default detection and accepts
  explicit terminal-only/disabled modes.
- Verify disabled mode performs no residual call and publishes no terminal
  product.
- Verify global and spatial detector observables remain distinct.
- Verify no ndarray/SystemState clone is introduced.
- Verify equilibrium still requires the authoritative residual.
- Verify periodic orbit is not emitted for a constant or transient history.
- Verify runtime progress and early target shortening.
- Verify maximum-iteration terminal trailing output.
- Verify fixed-point extraction against the new product.
- Run formatting, strict Clippy, all Rust tests/doctests, Python tests, and all
  GLV examples.

## Sequencing

1. Land and publish the approved core crate.
2. Update GLV dependencies and adapters.
3. Replace termination and terminal monitors in one task path.
4. Port tests and verify scientific equivalence.
5. Apply the path uniformly to every template and example.
6. Remove all superseded GLV modules and APIs.
7. Complete the full verification matrix before release.

## Canonical workload contract

- One directory is independently runnable by a GLV example.
- It contains or resolves every scientific input and all writer settings.
- Paths are resolved relative to that workload, never by Dispatcher.
- Configuration and sweeps are expanded before task execution; execution does
  not repeatedly parse JSON.
- The loader returns tasks carrying stable ordinals and concise display data.
- Output files live in the declared output root, not inside immutable input
  documents.
