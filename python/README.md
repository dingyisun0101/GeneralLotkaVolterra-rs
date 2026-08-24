# General Lotka–Volterra Reader

Official NumPy analysis decoders for completed GLV recordings. Workflow owns
recording integrity and JSONL reconstruction; PiP tensors are serialized
directly into those records, and this package validates and converts GLV's
`abundance`, optional species-last `space`, and `total` payloads.

Version 0.4 reads the PiP dense-tensor payload introduced by GLV 0.13. Older
GLV recordings that used the legacy ndarray-shaped JSON require the matching
older reader or an explicit conversion.

```python
from general_lotka_volterra_reader import open_glv_recording

reader = open_glv_recording("path/to/task-recording")
signal = reader.read_stream("signal")
```

For large histories, use `reader.iter_verified_records(name)` and fill private
preallocated arrays or memmaps. Publish outputs only after iteration succeeds.
