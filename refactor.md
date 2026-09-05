# GLV migration to Workflow 0.13.5

Reviewed on 2026-09-05: local `sw-version` based on
`f472c6de81b6fb4ca7e50f3288e829a9de8a5554`, including existing uncommitted Cargo,
README, and example manifest/lockfile changes. Those edits were preserved. The
validation copy used the working-tree files; this is not a claim about an
unmodified remote GLV commit.

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

## Changes needed here

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

After migration run `cargo test --workspace --all-targets` and
`PYTHONPATH=python/src python -m unittest discover -s python/tests -v`, then update
and exercise relevant independent example lockfiles. Preserve existing user edits
when preparing the GLV change; this document is the only file added by this task.
