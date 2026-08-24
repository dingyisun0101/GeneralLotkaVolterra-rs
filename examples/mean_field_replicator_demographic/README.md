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

The binary dispatches the replicate declared in `study.json`, passes its
Workflow execution scope to `GlvWorkload`, and registers the tasks in an
application-owned Study. In `config/parameters.json`, `rng.seed` uses
`{"$sweep": [...]}` to run the same scientific parameters with multiple PiP
seeds. Edit the shared `sigma` value there to control noise strength, or pass
another compatible configuration folder:

```sh
cargo run --release -- /path/to/inputs/config
```

Other main values in `config/parameters.json` are optional `initial_condition`, scalar or
per-species `growth`, `extinction_cutoff`, `time_step`,
`maximum_iterations`, and the recording streams. Each sweep RNG configuration creates a reproducible task with its own
recorded seed provenance.

Each task receives its own recording directory. Signal and checkpoint streams
have independent sampling intervals and storage budgets. The final
checkpoint is integrity-checked and compared with the final in-memory state.

The configuration explicitly selects `terminal_only`. Equilibrium and periodic-
orbit detection are intentionally disabled for this noisy model; every successful run still
publishes GLV's canonical trailing terminal-state estimate.

Checkpoints currently preserve state but not the RNG cursor, so exact
stochastic continuation is intentionally unsupported.

See the [GLV `sw-version` branch](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version)
for the plugin API and additional complete examples.
