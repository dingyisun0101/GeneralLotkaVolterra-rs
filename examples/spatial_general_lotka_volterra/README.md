# Spatial General Lotka–Volterra populations

This Workflow 0.13.1 project converts the canonical Ecological State Toolkit categorical lattice
to a species-last population field. `initial_population_per_site` is the sole
GLV-specific scale applied during conversion. The source realization and
interaction artifact are shared with the other examples.

The task declares no local state JSON; `GlvUnit` supplies Ecological State Toolkit's canonical
ecological schema through Workflow's standard provider API.

```bash
python -m pip install "scientific-workflow-reader[npy]==0.4.0"
cargo run --manifest-path examples/spatial_general_lotka_volterra/Cargo.toml
```

The final reserved `$npy` phase converts the completed member recording into
C-contiguous arrays.
