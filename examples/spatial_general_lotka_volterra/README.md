# Spatial General Lotka–Volterra populations

This Workflow 0.13.5 project converts the canonical Ecological State Toolkit categorical lattice
to a species-last population field. `initial_population_per_site` is the sole
GLV-specific scale applied during conversion. The source realization and
interaction artifact are shared with the other examples.

The task declares no local state JSON; `GlvUnit` supplies Ecological State Toolkit's canonical
ecological schema through Workflow's standard provider API.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install \
  "scientific-workflow[npy] @ git+https://github.com/dingyisun0101/Scientific-Workflow.git@v0.13.5#subdirectory=python"
cargo run --manifest-path examples/spatial_general_lotka_volterra/Cargo.toml
```

The final reserved `$npy` phase converts the completed member recording into
C-contiguous arrays.

Linux and Python 3.14+ are required. Activate `.venv` before every launch,
including in each new shell; Cargo does not install or activate Python. Keep
`wf_configs/study.json` and `wf_configs/parameters.json` in their standard locations.
