# General Lotka–Volterra for Rust

> **Breaking 0.17 update:** GLV now uses `ecological-state-toolkit` 0.12.1 and
> its renamed Rust import. There is no compatibility dependency on
> `ecological-model-core`; consumers that exchange toolkit-owned types must use
> the new crate.

`general-lotka-volterra-rs` provides allocation-conscious GLV and replicator
dynamics as a Scientific Workflow `ExecutionUnit`. Workflow registers the unit
under `glv`; the selected model remains an internal scientific composition.

## Architecture

```text
application / Dispatcher
├── Workflow Config ──> GlvConstants ───────────────────────┐
└── Ecological State Toolkit ──> interaction + initial-state artifacts ─┐   │
                                                        v   v
                                                  EcologicalInputs
                                                        │
                                                        v
                                  ExecutionUnit("glv"): GlvUnit
                                                        │
                     ┌──────────────────────────────────┼──────────────┐
                     v                                  v              v
                 numerical kernel                  invariant      observer
                     │                                  │              │
                     └──────────────> SystemState <─────┘              │
                                           │                           │
                                           v                           v
                                    Workflow writers          completion reason
                            signal / space / checkpoint        + terminal_state
```

Ownership is strict:

- Workflow alone reads `wf_configs`, expands sweeps, schedules units, owns
  state/time containers, derives requested runtime seeds, and writes results.
- The application or Dispatcher prepares shared scientific inputs through Eco
  Core and supplies their immutable references.
- Ecological State Toolkit owns ecological artifacts, the canonical ecological state schema,
  validation, resolution, and the common terminal-state format. It is
  model-neutral and has no dependency on GLV, Simulator, Dispatcher, or
  another private application crate.
- GLV advertises Ecological State Toolkit's standard schema to Workflow, resolves the prepared
  inputs, assembles its mathematical payloads into that schema, owns numerical
  evolution and invariants, and reports completion.
- Physics in Parallel owns tensors, matrices, lattice geometry, diffusion
  primitives, random engines, and bounded numerical parallelism.

GLV never reads project JSON or writes recordings directly.

## Models

`GlvModelConfig` supports:

- `mean_field_replicator`: deterministic mean-field RK4;
- `mean_field_replicator_demographic`: mean-field RK4 with demographic Gaussian
  noise;
- `spatial_replicator`: species-last local-frequency reaction–diffusion RK2;
- `spatial_general_lotka_volterra`: species-last absolute-population
  reaction–diffusion RK2.

All models consume the same `EcologicalInputs` envelope. The interaction
artifact determines species count. The initial-state artifact supplies one
canonical categorical lattice:

- mean-field models use its exact global frequencies;
- spatial replicator maps each site to a one-hot frequency cell;
- spatial population GLV maps each site to the selected
  `initial_population_per_site`.

This guarantees that GLV and Simulator can start from the same ecological
realization while retaining model-specific state representations.

## Project configuration

A Workflow project root must contain `wf_configs/study.json` and
`wf_configs/parameters.json`. GLV does not require a local state-schema file:
its `ExecutionUnit` supplies Ecological State Toolkit's canonical schema through Workflow's
standard provider API. Accordingly, a GLV task omits both `paths.states` and
the task-level `state` key.

```text
my-study/
├── prepared/inputs/...
├── src/main.rs
└── wf_configs/
    ├── study.json
    └── parameters.json
```

The minimal `study.json` boundary is:

```json
{
  "seed": 2001,
  "phases": {
    "simulate": {
      "tasks": [{"execution_unit": "glv"}]
    }
  }
}
```

Workflow asks `GlvUnit::standard_state_schema()` for the provider, records the
provider identity `ecological-state-toolkit.ecological-state.v1`, and resolves a
fresh schema for the task. GLV then follows the same lifecycle vocabulary as
Simulator: `validate_constants` → `resolve_inputs` → `assemble_state` →
`build_member` → step/observe. The two crates share this orchestration shape,
while each retains its own payload types and mathematical update rules.

The executable only links GLV's registration and enters Workflow:

```rust,no_run
use general_lotka_volterra_rs as _;

fn main() -> Result<(), scientific_workflow::WorkflowError> {
    scientific_workflow::run(std::path::Path::new(env!("CARGO_MANIFEST_DIR")))
}
```

`parameters.json["glv"]` deserializes directly as `GlvConstants`:

```json
{
  "glv": {
    "identity": "mean-field-replicator",
    "inputs": {
      "interaction": {
        "format": "ecological.interaction-artifact-reference.v2",
        "artifact_root": "/resolved/prepared",
        "descriptor": {"...": "InteractionArtifactDescriptor"}
      },
      "initial_state": {
        "format": "ecological.initial-state-artifact-reference.v2",
        "artifact_root": "/resolved/prepared",
        "descriptor": {"...": "InitialStateArtifactDescriptor"}
      }
    },
    "model": {
      "kind": "mean_field_replicator",
      "growth": 0.0,
      "extinction_cutoff": 1e-10,
      "time_step": 0.01
    },
    "recording": {
      "signal_interval": 1,
      "space_interval": 10,
      "checkpoint_interval": 100
    },
    "observation": {
      "mode": "detect",
      "equilibrium": true,
      "periodic_orbit": true
    },
    "maximum_iterations": 10000
  }
}
```

`growth` and spatial `diffusion` accept either one scalar or one value per
species. Lattice shape, spacing, boundary, species count, and initial
frequencies are not repeated in GLV configuration; they come from the prepared
initial artifact.

Relative artifact roots use the process working directory. Dispatcher should
normally pass resolved roots. A standalone project may set its working
directory to the project root before entering Workflow, as the checked-in
[mean-field example](examples/mean_field_replicator/README.md) does.

Current runnable examples cover
[deterministic mean field](examples/mean_field_replicator/README.md),
[demographic noise](examples/mean_field_replicator_demographic/README.md),
[spatial replicator](examples/spatial_replicator/README.md), and
[spatial population GLV](examples/spatial_general_lotka_volterra/README.md).

## Seeds and reproducibility

Input-generation seeds belong to the immutable Ecological State Toolkit artifacts and are not
requested again by GLV. Deterministic models need no runtime seed. The
demographic model requests one member-scoped `noise` seed from Workflow only
when its nested `rng.seed` is absent. Workflow records the actual derived seed
with the member metadata.

An explicit `rng.seed` is an advanced override. In that case GLV uses the
supplied value and does not make a Workflow seed request.

## Recording and completion

Every GLV member exposes the same three streams used by other ecological units:

| Stream | Fields | Purpose |
| --- | --- | --- |
| `signal` | `abundance`, `total` | frequent global behavior |
| `space` | `abundance`, `space`, `total` | spatial analysis |
| `checkpoint` | all fields | complete integrity/restart state |

Mean-field models retain `space = null`, preserving one stable schema across
all built-in models.

`observation.mode` is either `terminal_only` or `detect`. Detection applies
GLV's bounded equilibrium and periodic-orbit evidence policy. Stochastic GLV
requires `terminal_only`, because a noisy instantaneous residual is not valid
deterministic equilibrium evidence.

Every successful completion includes Ecological State Toolkit's common `terminal_state` inside
Workflow's completion metadata. Its classification distinguishes an accepted
equilibrium or periodic orbit from a trailing terminal estimate.

## Installation

Use the coordinated crates.io releases for application development:

```toml
[dependencies]
general-lotka-volterra-rs = "0.18.0"
scientific-workflow = "0.13.3"
```

A local clone remains appropriate when changing numerical methods, invariants,
payloads, or sibling crates together. The minimum toolchain is Rust 1.97,
edition 2024.

## Public API

Ordinary Workflow users need:

- `GlvConstants`: complete constants for one unit;
- `GlvModelConfig`: built-in model selection and numerical parameters;
- `SpeciesValues`: scalar-or-vector species parameter;
- `GlvObservationConfig`: `signal`, `space`, and `checkpoint` cadence;
- `ObservationConfig`: terminal-only or deterministic detection mode;
- `GlvUnit`: registered `ExecutionUnit` implementation;
- `GlvExecutionError`: boundary validation and construction errors.

Direct scientific users may construct `MeanFieldReplicator`,
`SpatialReplicator`, or `SpatialGeneralLotkaVolterra` and their typed configs.
`advanced::prelude` additionally exposes kernels, noise plugins, invariants,
interaction types, and categorical conversion. These APIs evolve numerical
state only; Workflow remains the project configuration and persistence owner.

## Scientific verification

The numerical core retains three independent layers of regression evidence:

- every built-in composition is compared step-for-step with an allocation-naive
  scalar solver;
- deterministic trajectories are compared with checked-in independent
  high-resolution ground truth;
- plugin and invariant tests verify failure-before-mutation, exact step order,
  shape contracts, and seeded noise reproducibility.

The Workflow integration test separately verifies shared-input resolution,
exact initial frequency conversion, runtime seed provenance, uniform streams,
and common terminal metadata.

## License

Licensed under either MIT or Apache-2.0, at your option.
