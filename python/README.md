# General Lotka–Volterra Reader

Official NumPy analysis decoders for completed GLV recordings. Workflow owns
recording integrity and JSONL reconstruction; PiP tensors are serialized
directly into those records, and this package validates and converts GLV's
`abundance`, optional species-last `space`, and `total` payloads.

Reader 0.5.2 uses published `scientific-workflow` 0.6.x
(import `scientific_workflow`), coordinated with Workflow 0.16, for unchanged
raw recording formats 7 and 8. Its decoder API is unchanged. It validates the
GLV model kind from the recorded `GlvConstants` and decodes PiP 4.1.1-alpha's
schema-v2 dense-tensor payloads. Ordinary GLV sampling remains periodic in
format 7. This adapter reads raw recordings; Workflow owns current NPY v3
conversion, while historical NPY interpretation belongs to downstream analysis.

This Python distribution remains private and is installed from the repository.
It is not published to PyPI.

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
  'scientific-workflow[npy]==0.6.1'
```

The Workflow companion is installed from its published PyPI release. Its former `scientific_workflow_reader` import has been removed.

```python
from general_lotka_volterra_reader import open_glv_recording

reader = open_glv_recording("path/to/task-recording")
signal = reader.read_stream("signal")
```

For large histories, use `reader.iter_verified_records(name)` and fill private
preallocated arrays or memmaps. Publish outputs only after iteration succeeds.

Run the decoder and raw-format integration checks from the repository root:

```sh
PYTHONPATH=python/src python -m unittest discover -s python/tests -v
```
