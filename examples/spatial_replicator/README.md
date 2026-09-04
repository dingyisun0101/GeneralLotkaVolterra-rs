# Spatial replicator

This Workflow 0.13.3 project converts the canonical Ecological State Toolkit categorical lattice
to a species-last one-hot frequency field. Lattice shape, boundary, spacing,
species count, and initial realization all come from `EcologicalInputs`; GLV
configuration adds only growth, diffusion, cutoff, and time step.

The task declares no local state JSON; `GlvUnit` supplies Ecological State Toolkit's canonical
ecological schema through Workflow's standard provider API.

```bash
python -m pip install \
  "scientific-workflow-reader[npy] @ git+https://github.com/dingyisun0101/Scientific-Workflow.git@6c8e7d4#subdirectory=python"
cargo run --manifest-path examples/spatial_replicator/Cargo.toml
```

The final reserved `$npy` phase converts the completed member recording into
C-contiguous arrays.
