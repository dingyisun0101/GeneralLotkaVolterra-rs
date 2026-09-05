# Mean-field replicator with demographic noise

This Workflow 0.13.7 project selects `mean_field_replicator_demographic`. It
consumes the same checked-in `EcologicalInputs` fixture as the deterministic and
spatial examples, then requests one member-scoped runtime seed named `noise`.
Workflow records the actual derived seed with the member output.

The task declares no local state JSON; `GlvUnit` supplies Ecological State Toolkit's canonical
ecological schema through Workflow's standard provider API.

Stochastic GLV uses `observation.mode = terminal_only`; deterministic
equilibrium evidence is intentionally unavailable under active noise.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install \
  "scientific-workflow[npy] @ git+https://github.com/dingyisun0101/Scientific-Workflow.git@v0.13.7#subdirectory=python"
cargo run --manifest-path examples/mean_field_replicator_demographic/Cargo.toml
```

The final reserved `$npy` phase converts the completed member recording into
C-contiguous arrays.

Linux and Python 3.14+ are required. Activate `.venv` before every launch,
including in each new shell; Cargo does not install or activate Python. Keep
`wf_configs/study.json` and `wf_configs/parameters.json` in their standard locations.
