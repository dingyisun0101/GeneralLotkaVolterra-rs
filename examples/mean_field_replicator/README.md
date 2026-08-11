# Mean-field replicator dynamics

This complete crate evolves a non-spatial community whose state
`x = (x_1, ..., x_S)` is a vector of relative species frequencies:

```text
dx_i/dt = x_i [f_i(x) - f̄(x)]
f_i(x)  = r_i + Σ_j A_ij x_j
f̄(x)   = Σ_i x_i f_i(x)
```

`A` is the interaction matrix and `r` is the intrinsic-growth vector. The
replicator subtraction removes common fitness and preserves total frequency.
After every deterministic or stochastic phase, the GLV frequency invariant
sets values below `cutoff` to zero and normalizes the surviving frequencies to
sum to one.

The built-in model uses classical fourth-order Runge–Kutta integration and no
stochastic noise. Its Workflow state has aggregate `abundance`, `space = None`,
and `total = 1`.

## Run

Install Rust 1.97 or newer, copy this entire directory, and run:

```sh
cargo run --release
```

An optional first argument selects another compatible Workflow project:

```sh
cargo run --release -- /path/to/project
```

The default project is this directory. `config/fixed.json` contains common
parameters, `config/sweep.json` varies `cutoff`, and `config/paths.json` names
the interaction matrix and output root. The matrix file uses rows as affected
species and columns as contributing species.

## Outputs

Every invocation creates a new collision-resistant execution directory beneath
`output/`. Each task records:

- `signal`: aggregate abundance and total;
- `space`: the complete canonical state; and
- `checkpoint`: restart-quality complete states.

Sampling and storage limits are configured independently under `recording`.
The program verifies the final checkpoint before reporting success.

See the [GLV `sw-version` branch](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version)
for the model API and additional complete examples.
