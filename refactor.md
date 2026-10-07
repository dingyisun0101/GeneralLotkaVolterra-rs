# GLV coordinated release and migration notes

## Current dependency patch: 0.20.1

This patch consumes registry Workflow 0.16.1 and Ecological State Toolkit 0.15.1,
retaining the exact PiP 4.1.1-alpha pin. Workflow accepts stable Python
companions `>=0.6,<0.7` during conversion and reuse. The private Python reader
remains 0.5.2; its installation guide selects the published companion 0.6.1.
Numerical algorithms, scientific APIs, schemas, and raw recordings remain
unchanged from 0.20.0. There are no code fingerprints or source overrides.

Upstream dependency minimums and lockfiles are updated only after both patches
are available online. GLV's default branch and release tag must be pushed before
Cargo publication. Private examples adopt the new registry patches only after
GLV itself is online; the private Python reader is never published.

Patch qualification against the online dependencies passed all 41 scientific
Rust/PTY tests, the source-only example configuration check, and one doctest.
Formatting, Clippy and rustdoc with warnings denied passed. Cargo packaged and
verified 58 files; that extracted crate independently passed the same 41 tests,
with the source-only example check explicitly ignored. Dependency-tree checks
found one registry Workflow 0.16.1 and one registry Ecological State Toolkit
0.15.1, with PiP unchanged. The private reader's four raw-v7/v8 tests passed
against public Python companion 0.6.1; its 0.5.2 wheel/sdist built locally and
remain unpublished. Evidence is in `/tmp/glv-0201-*.log`; these bounded/local
checks do not establish full-scale convergence or resource requirements.

Release commit `e5e8a6b` was merged into `sw-version` and pushed with annotated
tag `v0.20.1` before Cargo publication. Cargo upload and the crates.io
availability wait both succeeded. Only afterward were the four private example
manifests and lockfiles refreshed to registry GLV 0.20.1, Workflow 0.16.1,
Ecological State Toolkit 0.15.1, and unchanged PiP 4.1.1-alpha.

All four example pipelines then completed through real dashboards in temporary
copies with authored science unchanged and exact terminal restoration. Mean
field reached iteration 300 with 301 signal records; demographic mean field and
both spatial examples reached iteration 100 with 101 records. Their NPY v3
scalar-total arrays match verified raw values and coordinates, and every stream
reads through the private domain adapter. Evidence:
`/tmp/glv-0201-examples-qualification.log` and
`/tmp/glv-0201-examples-enk5x2ae/qualification.json`. Production outputs and
authored parameters were not modified.

## Initial generation and qualification: 0.20.0

GLV 0.20.0 consumes published Workflow 0.16.0, Ecological State Toolkit 0.15.0,
and exactly PiP 4.1.1-alpha. GLV numerical compositions, canonical ecological
schema meaning, and periodic raw recording formats are unchanged. This is a
breaking dependency generation: exchange Workflow/Eco-owned Rust types only
within the coordinated versions; no old-generation compatibility aliases are
provided.

The private Python reader remains private at 0.5.2 and consumes published
Workflow companion >=0.6,<0.7. Current NPY output is v3; Workflow-controlled reuse
requires current v2 receipts with durable captured-input checksum evidence.
Historical result interpretation is downstream-owned.

The four independent example crates stay private at 0.1.0. Their dependency
refresh follows GLV publication so they consume the online 0.20.0 release,
never a local source override or unpublished upstream change.

Validation for this pass:

- All 41 packaged Rust unit/integration tests and the separately invoked
  source-only private-example configuration test passed with published dependencies,
  including independent ground truth, stepwise naive solvers, seeded noise,
  failure-before-mutation, schema/provider checks, and a real-terminal Workflow run.
- One Rust doctest passed; formatting, Clippy with warnings denied, and rustdoc
  with warnings denied passed.
- All four private Python reader tests passed against installed published
  Workflow 0.6.0, including completed raw v7/v8 records; `pip check` passed.
- Dependency-tree inspection confirms one Workflow 0.16.0, one Ecological State
  Toolkit 0.15.0, and exactly PiP 4.1.1-alpha, with no source overrides.
- The Cargo package includes the JSON fixtures and release documentation needed
  by its shipped tests. Cargo excludes the independent private example workspaces;
  their configuration test is explicitly ignored in the ordinary package suite
  and invoked separately in source qualification.

Release commit `09f9227` was merged into default branch `sw-version` and pushed.
The annotated `v0.20.0` tag was pushed before `cargo publish --locked` successfully
published Rust GLV 0.20.0; a registry download independently confirmed availability.
Private Python 0.5.2 wheel/sdist artifacts built successfully and remain unpublished.
The packaged crate passed all 41 scientific/PTY tests with the one explicitly
source-only private-example configuration check ignored; that check passed
separately in the source checkout.

All four private example manifests/lockfiles now consume registry GLV 0.20.0,
Eco Core 0.15.0, Workflow 0.16.0, and exactly PiP 4.1.1-alpha. Complete pipelines
ran in temporary copies with their authored science unchanged and restored the
terminal exactly. Deterministic mean field reached iteration 300 with 301 signal
records; demographic mean field and both spatial examples reached iteration 100
with 101 signal records each. Their verified NPY v3 `total` series are rank one
and match the verified raw values/iteration coordinates exactly. Every raw stream
also reconstructed through the private GLV reader.
Evidence: `/tmp/glv-020-examples-qualification.log`,
`/tmp/glv-020-examples-ga_fbuy4/qualification.json`, and
`/tmp/glv-020-packaged-tests.log`. Temporary paths are local validation artifacts,
not production-scale convergence claims.

## Historical release and migration evidence

The following sections describe earlier generations, including their package
versions, installation methods, and validation environments. Current contracts
above supersede their old NPY/reuse/dependency guidance.


## Historical dependency update: 0.19.0

Version 0.19.0 consumes published Workflow 0.15.0 and Eco Core 0.14.0. Public
Workflow types require coordinated downstream migration; PiP remains
4.1.0-alpha. Rust tests, Clippy/rustdoc with warnings denied, doctests, three
Python decoder tests, and Rust/Python package checks passed. Commit and tag were
pushed before crates.io publication. The private Python reader is 0.5.1 and
consumes Workflow 0.5 from PyPI.

All four private examples consume the online releases and completed through a
real dashboard terminal, including NPY conversion. Users launch in screen/tmux;
disk recovery requires typing `resume`, and NPY auto admission ramps gradually.

## Previous dependency update: 0.18.8

Version 0.18.8 consumes published Workflow 0.14.2 and Eco Core 0.13.10.
Public APIs and PiP 4.1.0-alpha are unchanged. Python remains on companion
0.4.5, now pinned through Workflow's public `v0.14.2` tag. All 42 Rust tests,
one doctest, three Python decoder tests, formatting, Clippy with warnings
denied, and registry package verification passed.

All four private example crates now consume registry GLV 0.18.8, Eco Core
0.13.10, and Workflow 0.14.2. Their lockfiles were updated after publication,
and all four passed `cargo check --locked --all-targets`.

The earlier migration and validation history follows.

Initial audit on 2026-09-05: local `sw-version` based on
`f472c6de81b6fb4ca7e50f3288e829a9de8a5554`, including existing uncommitted Cargo,
README, and example manifest/lockfile changes. Those edits were preserved. The
validation copy used the working-tree files; this is not a claim about an
unmodified remote GLV commit. The subsequent 0.18.1 implementation and validation
are recorded below.

## Coordinated release requirements

[Workflow v0.13.5](https://github.com/dingyisun0101/Scientific-Workflow/releases/tag/v0.13.5)
publishes Rust `scientific-workflow` **0.13.5** and Python companion
`scientific-workflow` **0.4.3** (import `scientific_workflow`). Macros stay 0.2.1.
**The requested patch release contains breaking changes; no old Python aliases exist.**

**LINUX ONLY. Workflow's Python utilities require Python 3.14+. Upgrade older
Python versions. ACTIVATE THE ENVIRONMENT BEFORE EVERY LAUNCH, INCLUDING EACH
NEW SHELL. Cargo does not install or activate Python.**

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install \
  'scientific-workflow[npy] @ git+https://github.com/dingyisun0101/Scientific-Workflow.git@v0.13.5#subdirectory=python'
```

Omit `[npy]` for core recording readers without conversion/NPY views. Python is
published through the Git tag and GitHub wheel/source assets, not claimed on PyPI.
Rust 0.13.5 is published on crates.io.

**REQUIRED LAYOUT: `<study>/wf_configs/study.json` and `parameters.json`.
DO NOT RENAME OR RELOCATE THEM.** Accessors assume the standard layout. Programs
should read resolved parameter snapshots through the supported accessors, so
sweeps and overrides are respected.

Periodic recordings remain format 7; explicit `initial_and_final` recordings use
format 8. Both new readers accept 7 and 8. **NPY remains v2** and project schema
remains 1. Read the [upstream migration guide](https://github.com/dingyisun0101/Scientific-Workflow/blob/v0.13.5/docs/migration-0.13.5.md)
and [API references](https://github.com/dingyisun0101/Scientific-Workflow/blob/v0.13.5/README.md#subsystem-contracts).

## Migration contract

Update Workflow from 0.13.4 to 0.13.5 in the root and independent example Cargo
manifests/lockfiles. Current GLV execution units, schemas, state ownership, and
observations compile and run without Rust source changes. No new context object
or pause methods are needed. Keep scientific stepping and ecological payload
semantics in GLV.

In `python/pyproject.toml`, retain the `general-lotka-volterra-reader` package name,
raise Python to >=3.14, and replace the old `scientific-workflow-reader` Git pin
with the companion 0.4.3 install target above. In
`python/src/general_lotka_volterra_reader/decoders.py`, replace
`from scientific_workflow_reader ...` with `from scientific_workflow ...`.
`RecordingReader`, `open_completed_recording`, and the per-field decoder call
shapes remain available. Update Python README claims and installation instructions.

GLV continues to own abundance/space decoding and scientific interpretation.
Optional NPY consumers can use Workflow's whole-series views without inventing
GLV-specific manifest traversal. If checkpoint streams intend only initial/final
records, `.initial_and_final()` is available; use it deliberately because it
requires format 8 readers. Do not change ordinary periodic sampling incidentally.

## Validation evidence and limits

An isolated copy with only the Workflow dependency overridden to 0.13.5 passed
**42 Rust tests**, with no GLV Rust source edits. After the Python namespace
replacement, **all three Python decoder tests passed** against the installed
companion wheel. Independent example workspaces were reviewed for dependency
updates but were not all executed.

## Implemented in 0.18.1

The public Rust crate is bumped from 0.18.0 to 0.18.1. It now uses published
Eco Core 0.13.2, Workflow 0.13.5, and exactly PiP 4.1.0-alpha. Earlier PiP
versions have been yanked; applications exchanging PiP types with GLV must
upgrade their direct dependencies too. Dependency-tree checks confirm a single
version of PiP and Workflow through Eco Core and GLV. The four independent
example crates retain version 0.1.0 and `publish = false`; their GLV and Workflow
dependencies target the new releases.

The Python distribution retains its name and local version. Its Python minimum
is now 3.14, its Workflow companion pin is the v0.13.5 Git tag (Python 0.4.3),
and its decoder imports use `scientific_workflow`. Reader and example READMEs
document Linux, environment creation, activation before every launch, and the
optional NPY extra. Periodic sampling and the scientific Rust implementation
are unchanged.

Running the examples exposed a stale checked-in interaction fixture: its flat
matrix JSON was rejected by current Eco Core/PiP with `missing field backend`.
The same matrix `[[0, 1], [-1, 0]]` was regenerated with Eco Core 0.13.2's
`persist_interaction_matrix`. All four parameter files now reference its new
content-addressed filename and checksum. The existing seed-777 initial state
remains valid. The example preflight test now resolves and checks the actual
inputs; it failed on the old fixture and passed after regeneration.

Validation on 2026-09-05:

- All 42 Rust tests passed against the released dependencies with
  `cargo test --locked --workspace --all-targets`. Both Workflow integration
  tests passed again after strengthening the example fixture regression check.
- `cargo test --locked --doc` passed its one documentation test, and formatting
  passed `cargo fmt --all -- --check`.
- All three Python decoder tests passed in an activated Python 3.14 environment
  with the companion installed from the release tag. The local reader package
  also built and installed successfully, and `python -m pip check` passed.
- All four independent example studies completed in isolated copies using a
  temporary GLV source override, including their reserved `$npy` phases. The
  deterministic mean-field run reached 300 iterations; the other three reached
  100 iterations. Each produced a verified NPY v2 batch from format-7 recordings.
- The installed GLV reader reconstructed every stream in those recordings.
  Official NPY whole-series views matched all coordinates and 1,382 numeric
  field records exactly, including spatial tensors. GLV's tensor fields are
  structured projections at `/tensor`; `total` is a numeric series with one
  value per record. No application-side manifest path synthesis was required.
- `cargo publish --dry-run --locked --allow-dirty` verified the release package.
  `cargo publish --locked --allow-dirty` published GLV 0.18.1 to crates.io and
  confirmed registry availability; `cargo info` verified the published version.
- After publication, all four example lockfiles were refreshed against crates.io
  GLV 0.18.1 and passed `cargo check --locked --all-targets`. All five lockfiles
  select Eco Core 0.13.2, PiP 4.1.0-alpha, and Workflow 0.13.5, with no source
  overrides in the checked-in example manifests or lockfiles.

Existing local edits were preserved. Downstream applications must refresh their
own lockfiles to consume this coordinated release.
