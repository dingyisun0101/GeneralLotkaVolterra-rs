# Spatial replicator

This Workflow 0.11 project converts the canonical Eco Core categorical lattice
to a species-last one-hot frequency field. Lattice shape, boundary, spacing,
species count, and initial realization all come from `EcologicalInputs`; GLV
configuration adds only growth, diffusion, cutoff, and time step.

The task declares no local state JSON; `GlvUnit` supplies Eco Core's canonical
ecological schema through Workflow's standard provider API.

```bash
cargo run --manifest-path examples/spatial_replicator/Cargo.toml
```
