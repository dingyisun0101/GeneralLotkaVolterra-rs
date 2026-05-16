from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from .render_from_output import load_json_samples
except ImportError:
    np = None
    load_json_samples = None


class RenderFromOutputTests(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy is required for plotting tests")
    def test_state_data_takes_precedence_and_space_can_be_null(self):
        payload = {
            "epoch": 1,
            "mode": {"Frequency": {"cutoff": None}},
            "samples": [
                {
                    "time": 0,
                    "state": {"data": [1.0, 3.0]},
                    "space": None,
                    "mass": 1.0,
                },
                {
                    "time": 1,
                    "state": {"data": [2.0, 2.0]},
                    "space": {"dim": [1, 2], "data": [100.0, 0.0]},
                    "mass": 1.0,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "1.json"
            path.write_text(json.dumps(payload))

            t, nu = load_json_samples(path)

        np.testing.assert_array_equal(t, np.array([0.0, 1.0]))
        np.testing.assert_allclose(nu, np.array([[0.25, 0.75], [0.5, 0.5]]))


if __name__ == "__main__":
    unittest.main()
