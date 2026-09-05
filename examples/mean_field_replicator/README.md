# Mean-field replicator Workflow project

This is the current GLV configuration boundary in one runnable project:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install \
  "scientific-workflow[npy] @ git+https://github.com/dingyisun0101/Scientific-Workflow.git@v0.13.5#subdirectory=python"
cargo run --manifest-path /path/to/glv/examples/mean_field_replicator/Cargo.toml
```

The final reserved `$npy` phase converts the completed member recording into
C-contiguous arrays.

The application prepares a model-ready interaction matrix and one canonical
categorical initial state through Ecological State Toolkit. Both references are grouped as
`EcologicalInputs` in `parameters.json`. GLV resolves that same envelope,
converts categorical counts to exact aggregate frequencies, and runs the
selected mean-field model. It does not read project JSON, generate an initial
condition, or copy input artifacts into its output.

```text
wf_configs/ ──Workflow Config──> GlvConstants
prepared/ ──Ecological State Toolkit references──> EcologicalInputs
                                      │
                                      v
                    GlvUnit -> SystemState -> Workflow writers
                                      │
                    signal / space / checkpoint + terminal_state
```

`wf_configs/` is required for this directory to qualify as a Workflow project
root. No state-schema file or `study.json` state key is needed: `GlvUnit`
provides Ecological State Toolkit's canonical ecological schema directly to Workflow. The tiny
`main.rs` only links GLV's `glv` registration, sets the working directory for
the relative artifact roots, and enters Workflow.

```text
wf_configs/
├── study.json
└── parameters.json
```

The example uses the deterministic RK4 mean-field replicator:

```text
dx_i/dt = x_i [f_i(x) - f̄(x)]
f_i(x)  = r_i + Σ_j A_ij x_j
```

`growth` may be one scalar or one value per species. Species count, initial
frequencies, and lattice provenance come from `EcologicalInputs`.
`extinction_cutoff`, `time_step`, and `maximum_iterations` are GLV execution
policy. The uniform streams are `signal`, `space`, and `checkpoint`; the
mean-field `space` payload is `null`. Completion metadata always contains Eco
Core's common `terminal_state` product.

The checked-in initial artifact was generated once with seed `777`. That seed
belongs to the artifact and GLV never requests it again. Deterministic execution
needs no runtime seed.

The interaction fixture contains `[[0, 1], [-1, 0]]`. For GLV 0.18.1 it was
regenerated through Eco Core 0.13.2's `persist_interaction_matrix` to use the
current PiP container encoding. All four examples reference the same verified
artifact and its content checksum.

Linux and Python 3.14+ are required. Activate `.venv` before every launch,
including in each new shell; Cargo does not install or activate Python. Keep
`wf_configs/study.json` and `wf_configs/parameters.json` in their standard locations.
