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
- `initialization`: shared categorical ecological initial-state source;
- `initial_population_per_site`: explicit population assigned to the selected
  taxon at each categorical site;
- `growth` and `diffusion`: one value per species or one shared scalar;
- `spacing`: optional values per spatial axis; omission uses unit spacing;
- `boundary`: `periodic` or `neumann`;
- `carrying_capacity`: an optional positive total capacity; and
- `physical_time_increment`: RK2 time increment.

Both deterministic automatic-termination detectors are enabled: equilibrium
convergence and a nontrivial periodic orbit. Their evidence policy is
internal to GLV.

The binary loads a `GlvWorkload` and registers its tasks in an application-owned
Workflow phase. It contains no model assembly or recording code. Pass another
compatible workload directory as the optional first
argument.
Every invocation creates new output and records aggregate, spatial, and
checkpoint streams. The program verifies the final checkpoint before success.
Every successful task also records a classified terminal composition. For
population GLV, single-species support alone is not accepted as a fixed point;
the population RHS and ordinary convergence evidence must pass.

See the [GLV `sw-version` branch](https://github.com/dingyisun0101/GeneralLotkaVolterra-rs/tree/sw-version)
for the model API and additional complete examples.
