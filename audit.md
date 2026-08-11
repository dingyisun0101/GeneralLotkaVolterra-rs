# GLV Audit

Date: 2026-08-11

Scope
- Reviewed `glv` crate core modules: `engine`, `core`, `kernel`, `noise`, `invariant`,
  `reading`, and `recording` and relevant `workflow` integration points.

Summary
- Design is clear and well-layered: a single authoritative `SystemState` owned by
  `Engine`, deterministic `Kernel` then `Invariant` then stochastic `Noise` ordering,
  and strict validation at boundaries.

Simplification opportunities
- Remove thin wrapper duplication: `GlvRecordingConfig` / `StreamRecordingConfig`
  largely mirror `scientific_workflow::storage::StateStreamConfig`. Provide a small
  adapter constructor rather than maintaining parallel wrapper types.
- Micro-refactor: bind the matrix view once in hot loops (e.g. `KernelCore::apply_interaction`) to
  avoid repeated `self.interaction.values()` calls and improve clarity.
- Consider unifying common code between proportional and demographic Gaussian noise
  as a parameterized strategy to reduce duplication.

Logical/semantic observations and edge cases
- The engine requires a present physical-time coordinate (engine validates `physical_time.is_some()`).
  This is intentional but important: iteration-only runs will be rejected at build/step time.
- Several commit paths use `expect(...)` after prior validation (e.g. applying a spatial update
  or copying into contiguous slices). If validation ever misses a case, these `expect`s can panic.
  Tests currently exercise many failure modes; convert to explicit error returns if you want
  a no-panic policy.
- Many routines assume standard contiguous row-major `ndarray` layout (use of `as_slice()`).
  Ensure callers construct arrays accordingly or add explicit layout-validation where needed.

Correctness checks
- Extensive numeric and shape validations are present (finite checks, non-negativity,
  overflow guards). I found no obvious algorithmic bug in the inspected files.

Tests & recommendations
- Keep and expand tests for the following:
  - non-contiguous ndarray layouts (ensure rejections are explicit),
  - missing physical-time states (confirm intended rejection),
  - recovery/resume behavior (already covered),
  - RNG provenance round-trips and determinism.
- If zero-panic guarantees are required, replace a few `expect` usages in commit paths
  with explicit error propagation and add tests to assert no panics in failure paths.

Next steps (optional)
- I can prepare a tiny patch demonstrating the `apply_interaction` micro-refactor and
  one `expect` → error conversion as a safe example. I can also run targeted tests.
