# Spatial General Lotka–Volterra populations

This crate models absolute population counts on a grid. For species `i`,

```text
∂n_i/∂t = n_i [r_i + Σ_j A_ij n_j] + D_i ∇²n_i.
```

`r_i` is intrinsic growth, `A_ij` is the effect of species `j` on species `i`,
and `D_i` controls diffusion. Negative diagonal interaction terms commonly
represent self-limitation; off-diagonal signs encode competition, facilitation,
or predator–prey effects.

Arrays are species-last. The kernel applies finite-difference diffusion with
periodic or zero-flux Neumann boundaries and integrates the combined reaction–
diffusion equation using explicit midpoint RK2. Construction rejects time steps
above a conservative diffusion stability limit.

The population invariant removes counts at or below `cutoff`, recomputes
aggregate abundance, and optionally rescales the field to `carrying_capacity`.
Unlike frequency models, aggregate abundance and `total` represent absolute
population rather than a unit simplex.

## Run

Copy this complete directory and run:

```sh
cargo run --release
```

Important values in `config/fixed.json` are:

- `spatial_shape`: grid axes, excluding species;
- `initial_cell`: initial count of each species in every cell;
- `growth` and `diffusion`: one value per species;
- `spacing`: one value per spatial axis;
- `boundary`: `periodic` or `neumann`;
- `carrying_capacity`: a positive total capacity or `null`; and
- `physical_time_increment`: RK2 time increment.

The binary selects `GlvTemplate::SpatialGeneralLotkaVolterra` and contains no
model assembly or recording code. Pass another compatible Workflow `config`
folder as the optional first argument.
Every invocation creates new output and records aggregate, spatial, and
checkpoint streams. The program verifies the final checkpoint before success.

See the [GLV `sw-version` branch](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version)
for the model API and additional complete examples.
