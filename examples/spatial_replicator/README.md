# Spatial replicator

This Workflow 0.13.7 project converts the canonical Ecological State Toolkit categorical lattice
to a species-last one-hot frequency field. Lattice shape, boundary, spacing,
species count, and initial realization all come from `EcologicalInputs`; GLV
configuration adds only growth, diffusion, cutoff, and time step.

The task declares no local state JSON; `GlvUnit` supplies Ecological State Toolkit's canonical
ecological schema through Workflow's standard provider API.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install \
  "scientific-workflow[npy] @ git+https://github.com/dingyisun0101/Scientific-Workflow.git@v0.13.7#subdirectory=python"
cargo run --manifest-path examples/spatial_replicator/Cargo.toml
```

The final reserved `$npy` phase converts the completed member recording into
C-contiguous arrays.

Linux and Python 3.14+ are required. Activate `.venv` before every launch,
including in each new shell; Cargo does not install or activate Python. Keep
`wf_configs/study.json` and `wf_configs/parameters.json` in their standard locations.
