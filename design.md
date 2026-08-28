# GLV execution-unit design

## Decision

GLV implements the universal Workflow `ExecutionUnit` directly. One `GlvUnit`
owns one independently stateful numerical simulation and exposes it through one
stable `MemberView`. Workflow does not know which GLV model is selected.

The application supplies one `EcologicalInputs` value containing a verified
interaction artifact reference and a verified canonical initial-state
reference. GLV does not own shared-input generation, configuration loading, or
recording persistence.

Eco Core also supplies the canonical ecological state schema. `GlvUnit`
advertises that standard provider, so GLV projects declare neither a copied
schema file nor a task-level state name.

## Input conversion

```text
EcologicalInputs
├── InteractionMatrix ─────────────────────────────┐
└── categorical InitialState                       │
    ├── frequencies -> mean-field                  │
    ├── one-hot cells -> spatial replicator        ├──> typed simulation
    └── scaled one-hot cells -> spatial population │
                                                   v
                                               SystemState
```

The same categorical realization can therefore feed discrete Simulator space,
GLV aggregate frequencies, or GLV continuous spatial fields without competing
input generators.

## Lifecycle

`preflight` calls `validate_constants` for descriptor dimensions, schema order,
GLV parameters, observation policy, and stochastic/detector compatibility
without reading artifacts. `initialize` follows `resolve_inputs` →
`assemble_state` → `build_member` using the exact schema instance supplied by
Workflow. Simulator uses the same architectural vocabulary while retaining
different payload types and mathematics. `step` performs one complete
numerical transition and then updates the bounded observer. `member` is
side-effect free.

Workflow owns all stream sampling and persistence. GLV declares uniform
`signal`, `space`, and `checkpoint` streams and returns structured completion
metadata containing Eco Core's `terminal_state`.

## Reproducibility

Prepared-input RNG provenance remains in the artifacts. Only demographic noise
may require a runtime seed. When its explicit seed is absent, GLV requests the
member-scoped purpose `noise` from `InitializationContext`; Workflow records the
actual derived value.

## Compatibility and efficiency

Direct simulation types, kernels, plugins, and invariants remain independent of
Workflow scheduling. The Workflow adapter adds no per-step full-state copy.
Numerical scratch is allocated at construction and reused. Trajectory
observation retains bounded windows, and disabled terminal production is not an
option at this boundary because ecological executions must expose a common
terminal product.
