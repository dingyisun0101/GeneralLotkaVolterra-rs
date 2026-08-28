# Spatial General Lotka–Volterra populations

This Workflow 0.11 project converts the canonical Eco Core categorical lattice
to a species-last population field. `initial_population_per_site` is the sole
GLV-specific scale applied during conversion. The source realization and
interaction artifact are shared with the other examples.

The task declares no local state JSON; `GlvUnit` supplies Eco Core's canonical
ecological schema through Workflow's standard provider API.

```bash
cargo run --manifest-path examples/spatial_general_lotka_volterra/Cargo.toml
```
