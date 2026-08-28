# Spatial replicator

This Workflow 0.11 project converts the canonical Eco Core categorical lattice
to a species-last one-hot frequency field. Lattice shape, boundary, spacing,
species count, and initial realization all come from `EcologicalInputs`; GLV
configuration adds only growth, diffusion, cutoff, and time step.

```bash
cargo run --manifest-path examples/spatial_replicator/Cargo.toml
```
