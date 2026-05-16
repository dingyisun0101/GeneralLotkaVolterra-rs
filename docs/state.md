# State

## Purpose

The state layer owns the in-memory and persisted representation shared by
solvers and task runners. It stores one global taxon vector, an optional spatial
field, a mode, integer time, and cached mass.

## Data Model

- `Mode::Frequency` treats `state` as a simplex vector and keeps `mass = 1`.
- `Mode::Population` treats `state` as absolute counts and keeps `mass` close
  to the rounded sum after cutoff and optional capacity enforcement.
- `SystemStateRecord` stores owned snapshot data without repeating the mode.
  Spatial signal-only records may omit `space` while retaining aggregate
  `state`.
- `SystemStateTimeSeries` stores one shared mode plus an ordered sample list for
  one epoch.

## Validation Contract

`SystemState::sanitize` is the boundary that restores mode-specific invariants:

1. Remove non-finite, nonpositive, and below-cutoff entries.
2. Renormalize frequency states onto the simplex, falling back to uniform mass
   when all entries are removed.
3. Cap population states when `carrying_capacity` is present.
4. Refresh cached `mass`.

## File Layout

- `src/state/system_state.rs`: one state snapshot and mode-specific invariant
  handling.
- `src/state/time_series.rs`: epoch-level save/load container.
- `src/state/mod.rs`: public state module surface.
