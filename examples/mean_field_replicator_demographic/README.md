# Mean-field replicator with demographic noise

This crate augments mean-field replicator dynamics with seeded demographic
Gaussian fluctuations. The deterministic drift is

```text
dx_i/dt = x_i [f_i(x) - f̄(x)]
f_i(x)  = r_i + Σ_j A_ij x_j.
```

For one time increment `dt`, the noise plugin proposes

```text
x_i <- x_i + σ sqrt(dt) sqrt(x_i) (η_i - η̄),   η_i ~ Normal(0, 1),
```

where the weighted common component `η̄` is removed. The frequency invariant
then applies the extinction cutoff and renormalizes the community to unit mass.
The `sqrt(x_i)` amplitude makes fluctuations relatively stronger for rare
species, as expected for demographic rather than proportional noise.

The plugin accepts PiP's universal `RngConfig` and uses PiP's random filler,
defaulting here to ChaCha12 with one stream. Workflow records the resolved
method, implementation version, stream count, key encoding, and exact `seed` as
provenance; Workflow does not generate random numbers itself.

## Run and sweep seeds

Copy the entire directory and run:

```sh
cargo run --release
```

The binary delegates the complete project to
`GlvTemplate::MeanFieldReplicatorDemographic`. The checked-in
`config/sweep.json` runs the same scientific parameters with multiple complete
PiP `RngConfig` values. Edit `sigma` in `config/fixed.json` to control noise
strength, or pass another compatible configuration folder:

```sh
cargo run --release -- /path/to/project/config
```

Each task receives its own recording directory. Signal, spatial, and checkpoint
streams have independent sampling intervals and storage budgets. The final
checkpoint is integrity-checked and compared with the final in-memory state.

Checkpoints currently preserve state but not the RNG cursor, so exact
stochastic continuation is intentionally unsupported.

See the [GLV `sw-version` branch](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version)
for the plugin API and additional complete examples.
