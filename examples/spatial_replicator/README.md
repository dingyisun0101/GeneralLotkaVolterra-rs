# Spatial replicator reaction–diffusion

This crate evolves a local frequency vector in every grid cell. Arrays are
species-last: a two-dimensional system has shape `[rows, columns, species]`.
The continuous model is

```text
∂x_i/∂t = x_i [f_i(x) - f̄(x)] + D_i ∇²x_i
f_i(x)  = r_i + Σ_j A_ij x_j.
```

`D_i` is the diffusion coefficient of species `i`. The finite-difference
Laplacian uses the configured grid `spacing` and either:

- `periodic` boundaries, which wrap to the opposite edge; or
- `neumann` boundaries, which impose zero flux.

The built-in kernel uses explicit midpoint RK2. Construction checks a
conservative diffusion stability limit. After kernel and noise phases, the
local-frequency invariant applies `cutoff` and normalizes every cell to sum to
one. Aggregate abundance is the mean composition across cells and `total = 1`.

## Run

Copy this complete directory and run:

```sh
cargo run --release
```

The default `fixed.json` uses `ecological-initial-state` to sample one
categorical taxon per site, then converts each site to a one-hot frequency
cell. Its explicit PiP RNG configuration is recorded as provenance.
Change `spatial_shape`, per-species `diffusion`, per-axis `spacing`, or
`boundary` to define another lattice. The final array axis is inferred from the
length of `growth`; the interaction matrix and initialization distribution
must have the same dimension.

Both deterministic automatic-termination detectors are enabled: fixed-point
convergence and nontrivial recurrent oscillation. GLV owns their evidence
policy and records which outcome, if any, ended the run.

The binary selects `GlvTemplate::SpatialReplicator`; all construction and
Workflow orchestration are supplied by that built-in template. An optional
configuration-folder path allows the same binary to run another project:

```sh
cargo run --release -- /path/to/project/config
```

Outputs are written to a fresh execution scope. The `space` stream retains the
full field, while the smaller `signal` stream is suited to aggregate analysis.
The `checkpoint` stream supports deterministic continuation.

See the [GLV `sw-version` branch](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version)
for the numerical API and additional complete examples.
