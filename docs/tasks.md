# Tasks

## Purpose

Task modules expose experiment-level runners. They configure initial state,
choose a noise policy, run sequential epochs, and persist epoch files.

## Epoch Contract

Each task run:

1. Creates the task-specific initial condition.
2. Runs epochs in order.
3. Carries the final state from epoch `k` into epoch `k + 1`.
4. Writes each epoch's `SystemStateTimeSeries` under the caller-provided output
   path.

Well-mixed tasks use a uniform simplex initial condition. Spatial replicator
tasks use a uniform local simplex in every spatial cell. Spatial GLV tasks use a
uniform initial population density in every spatial cell and species.

## Ready Tasks

- `replicator_deterministic::run`: RK4 replicator dynamics without noise.
- `replicator_demographic::run`: RK4 replicator dynamics with demographic
  Gaussian noise.
- `replicator_diffusive_deterministic::run`: spatial local-simplex replicator
  reaction-diffusion without noise.
- `lv_diffusive_deterministic::run`: spatial GLV population
  reaction-diffusion without noise.

Spatial task runners accept separate save intervals for aggregate signal
samples and full spatial-field samples.

## Placeholder Tasks

- `lv_deterministic::run`
- `lv_demographic::run`

These return `ErrorKind::Unsupported` because the implemented solver stack is
replicator-form only for non-spatial tasks.
