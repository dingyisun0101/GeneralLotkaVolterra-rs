# State

## Purpose

The system-state layer owns the live in-memory representation shared by solvers
and task runners. It stores one global taxon vector, an optional spatial field,
a mode, integer time, and cached mass.

## Data Model

- `Mode::Frequency` treats `state` as a simplex vector and keeps `mass = 1`.
- `Mode::Population` treats `state` as absolute counts and keeps `mass` close
  to the rounded sum after cutoff and optional capacity enforcement.
Persisted JSON records are owned by `src/io/signal.rs` and `src/io/space.rs`,
not by `SystemState`.

## Validation Contract

`SystemState::sanitize` is the boundary that restores mode-specific invariants:

1. Remove non-finite, nonpositive, and below-cutoff entries.
2. Renormalize frequency states onto the simplex, falling back to uniform mass
   when all entries are removed.
3. Cap population states when `carrying_capacity` is present.
4. Refresh cached `mass`.

## File Layout

- `src/system_state.rs`: one live state snapshot and mode-specific invariant
  handling.
- `src/io/signal.rs`: aggregate signal JSON output.
- `src/io/space.rs`: full spatial snapshot JSON output.
