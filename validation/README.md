# Legacy validation

`run.sh` executes every published example and compares the refactored
deterministic kernels and seeded demographic-noise update against legacy commit
`5ad7cad1ade361e4ee40e540db72d602565e15e8`.

The script extracts that immutable commit into the ignored
`validation/legacy-source/` directory. Each invocation creates a new directory
beneath `validation/runs/`; previous validation evidence is never deleted or
overwritten. The machine-readable comparison is
`legacy-comparison.json`, and each runnable example has a separate log.
The run also verifies and exports one completed Workflow signal stream to
`mean-field-signal.csv` through the publication plotting/analysis helper.

Run from any directory:

```sh
bash /path/to/glv/validation/run.sh
```

Absolute and relative comparison tolerances are both `1e-12`. The stochastic
comparison supplies the same explicit ChaCha12 seed to the legacy noise kernel
and the refactored plugin; this avoids comparing unrelated entropy-seeded
trajectories.

`report.json` is the checked-in summary of the latest complete publication
validation. Detailed logs and numerical arrays remain in its immutable,
ignored run directory.
