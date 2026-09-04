# General Lotka–Volterra Reader

Official NumPy analysis decoders for completed GLV recordings. Workflow owns
recording integrity and JSONL reconstruction; PiP tensors are serialized
directly into those records, and this package validates and converts GLV's
`abundance`, optional species-last `space`, and `total` payloads.

Version 0.5 uses `scientific-workflow-reader` 0.4 for Workflow recording format
7 and validates the GLV model kind from the recorded `GlvConstants`. It decodes
the PiP 4 dense-tensor payload and has no legacy recording compatibility
layer.

```python
from general_lotka_volterra_reader import open_glv_recording

reader = open_glv_recording("path/to/task-recording")
signal = reader.read_stream("signal")
```

For large histories, use `reader.iter_verified_records(name)` and fill private
preallocated arrays or memmaps. Publish outputs only after iteration succeeds.
