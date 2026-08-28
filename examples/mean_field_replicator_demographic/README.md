# Mean-field replicator with demographic noise

This Workflow 0.11 project selects `mean_field_replicator_demographic`. It
consumes the same checked-in `EcologicalInputs` fixture as the deterministic and
spatial examples, then requests one member-scoped runtime seed named `noise`.
Workflow records the actual derived seed with the member output.

The task declares no local state JSON; `GlvUnit` supplies Eco Core's canonical
ecological schema through Workflow's standard provider API.

Stochastic GLV uses `observation.mode = terminal_only`; deterministic
equilibrium evidence is intentionally unavailable under active noise.

```bash
cargo run --manifest-path examples/mean_field_replicator_demographic/Cargo.toml
```
