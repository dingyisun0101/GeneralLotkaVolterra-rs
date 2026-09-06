# General Lotka–Volterra Reader

Official NumPy analysis decoders for completed GLV recordings. Workflow owns
recording integrity and JSONL reconstruction; PiP tensors are serialized
directly into those records, and this package validates and converts GLV's
`abundance`, optional species-last `space`, and `total` payloads.

This package uses `scientific-workflow` 0.4.4 (import `scientific_workflow`),
released with Workflow 0.13.9, for recording formats 7 and 8. It validates the
GLV model kind from the recorded `GlvConstants` and decodes PiP 4.1.0-alpha's
schema-v2 dense-tensor payloads. Ordinary GLV sampling remains periodic in
format 7.

Linux and Python 3.14+ are required. From the GLV repository root:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install ./python
```

Activate the environment before every launch, including in each new shell.
Cargo does not install or activate Python. For examples using the `$npy` phase,
also install the conversion extra:

```sh
source .venv/bin/activate
python -m pip install \
  'scientific-workflow[npy] @ git+https://github.com/dingyisun0101/Scientific-Workflow.git@v0.13.9#subdirectory=python'
```

The Workflow companion is installed from the release Git tag or GitHub release
assets, not PyPI. Its former `scientific_workflow_reader` import has been removed.

```python
from general_lotka_volterra_reader import open_glv_recording

reader = open_glv_recording("path/to/task-recording")
signal = reader.read_stream("signal")
```

For large histories, use `reader.iter_verified_records(name)` and fill private
preallocated arrays or memmaps. Publish outputs only after iteration succeeds.
