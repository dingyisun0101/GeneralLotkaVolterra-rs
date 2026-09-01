# Mean-field replicator with demographic noise

This Workflow 0.13.1 project selects `mean_field_replicator_demographic`. It
consumes the same checked-in `EcologicalInputs` fixture as the deterministic and
spatial examples, then requests one member-scoped runtime seed named `noise`.
Workflow records the actual derived seed with the member output.

The task declares no local state JSON; `GlvUnit` supplies Ecological State Toolkit's canonical
ecological schema through Workflow's standard provider API.

Stochastic GLV uses `observation.mode = terminal_only`; deterministic
equilibrium evidence is intentionally unavailable under active noise.

```bash
python -m pip install "scientific-workflow-reader[npy]==0.4.0"
cargo run --manifest-path examples/mean_field_replicator_demographic/Cargo.toml
```

The final reserved `$npy` phase converts the completed member recording into
C-contiguous arrays.
