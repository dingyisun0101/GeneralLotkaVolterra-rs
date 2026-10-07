from __future__ import annotations

import unittest
import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from general_lotka_volterra_reader import GlvPayloadError, decode_abundance, decode_space, open_glv_recording


class DecoderTests(unittest.TestCase):
    def test_completed_raw_formats_preserve_glv_payloads_and_coordinates(self) -> None:
        fixture = json.loads((Path(__file__).parent / "fixtures/payloads.json").read_text())
        for version in (7, 8):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                stream = root / "stream_0000"
                stream.mkdir()
                data = b"".join(json.dumps({
                    "iteration": iteration,
                    "physical_time": physical_time,
                    "values": [fixture["abundance"], None, fixture["total"]],
                }, allow_nan=False).encode() + b"\n" for iteration, physical_time in ((0, 0.0), (2, 0.25)))
                (stream / "chunk-000000.jsonl").write_bytes(data)
                (root / "metadata.json").write_text(json.dumps({
                    "format": "scientific-workflow-jsonl", "version": version,
                    "status": {"state": "complete"},
                    "timing": {"created_at_utc": "2026-10-07T00:00:00Z", "finalized_at_utc": "2026-10-07T00:00:01Z",
                               "active_duration_ns": 1_000_000_000, "continuation_count": 0},
                    "records": {"encoding": "json", "framing": "json_lines"},
                    "time": {"iteration_name": "iteration", "physical_time_name": "physical_time"},
                    "user_metadata": {"constants": {"model": {"kind": "mean_field_replicator"}}},
                    "streams": [{"name": "signal", "directory": stream.name,
                                 "sampling_interval": {"iterations": 2} if version == 7 else {"initial_and_final": True},
                                 "fields": [{"name": name} for name in ("abundance", "space", "total")],
                                 "storage": {"layout": {"kind": "chunked", "target_bytes": 1024}, "storage_queue_bytes": 4096},
                                 "chunks": [{"ordinal": 0, "file": "chunk-000000.jsonl", "records": 2, "bytes": len(data),
                                             "checksum": "sha256:" + hashlib.sha256(data).hexdigest(),
                                             "first_iteration": 0, "last_iteration": 2}]}],
                }))
                reader = open_glv_recording(root)
                self.assertEqual(reader.format_version, version)
                series = reader.read_stream("signal")
                self.assertEqual(series.iterations, (0, 2))
                self.assertEqual(series[-1].physical_time, 0.25)
                abundance = series[-1].values["abundance"]
                self.assertEqual(abundance.dtype, np.dtype(np.float64))
                np.testing.assert_array_equal(abundance, [0.2, 0.3, 0.5])
                self.assertIsNone(series[-1].values["space"])
                self.assertEqual(series[-1].values["total"], 1.0)
                np.testing.assert_array_equal(reader.read_latest("signal").values["abundance"], abundance)
                self.assertEqual(tuple(record.iteration for record in reader.iter_verified_records("signal")), (0, 2))

    def test_abundance_is_contiguous_float64(self) -> None:
        value = json.loads((Path(__file__).parent / "fixtures/payloads.json").read_text())["abundance"]
        decoded = decode_abundance(value)
        self.assertEqual(decoded.dtype, np.dtype(np.float64))
        self.assertTrue(decoded.flags.c_contiguous)
        np.testing.assert_array_equal(decoded, [0.2, 0.3, 0.5])

    def test_species_last_space_is_reshaped(self) -> None:
        value = {
            "backend": "dense",
            "tensor": {
                "kind": "tensor",
                "version": 2,
                "scalar": "f64",
                "shape": [2, 2, 2],
                "data": list(range(8)),
            },
        }
        decoded = decode_space(value)
        self.assertEqual(decoded.shape, (2, 2, 2))
        self.assertEqual(decoded[1, 1, 1], 7.0)

    def test_invalid_shape_fails_closed(self) -> None:
        with self.assertRaises(GlvPayloadError):
            decode_abundance({
                "backend": "dense",
                "tensor": {
                    "kind": "tensor",
                    "version": 2,
                    "scalar": "f64",
                    "shape": [2],
                    "data": [1.0],
                },
            })


if __name__ == "__main__":
    unittest.main()
