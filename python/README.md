# General Lotka–Volterra Reader

Official NumPy payload decoders for completed GLV recordings. Workflow owns
recording integrity and JSONL reconstruction; this package owns the meaning of
GLV's `abundance`, optional species-last `space`, and `total` payloads.

```python
from general_lotka_volterra_reader import open_glv_recording

reader = open_glv_recording("path/to/task-recording")
signal = reader.read_stream("signal")
```

For large histories, use `reader.iter_verified_records(name)` and fill private
preallocated arrays or memmaps. Publish outputs only after iteration succeeds.
