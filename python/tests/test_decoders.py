from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np

from general_lotka_volterra_reader import GlvPayloadError, decode_abundance, decode_space


class DecoderTests(unittest.TestCase):
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
