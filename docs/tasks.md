# Tasks

## Purpose

Task modules expose experiment-level runners. They configure initial state,
choose a noise policy, run sequential epochs, and persist epoch files.

## Epoch Contract

Each task run:

1. Creates a well-mixed initial condition.
2. Runs epochs in order.
3. Carries the final state from epoch `k` into epoch `k + 1`.
4. Writes each epoch's `SystemStateTimeSeries` under the caller-provided output
   path.

## Ready Tasks

- `replicator_deterministic::run`: RK4 replicator dynamics without noise.
- `replicator_demographic::run`: RK4 replicator dynamics with demographic
  Gaussian noise.

## Placeholder Tasks

- `lv_deterministic::run`
- `lv_demographic::run`

These return `ErrorKind::Unsupported` because the implemented solver stack is
replicator-form only.
