"""Runtime contracts tested with frozen models and small engineering fixtures.

Fixtures validate execution behavior; they are not accuracy benchmarks.
"""

import hashlib
import json
import unittest
from pathlib import Path

import joblib
import numpy as np

from autoshield.runtime import BehavioralRuntime, CompletedFlow

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def load_runtime(self, name):
        spec = json.loads((ROOT / "models/manifest.json").read_text())["models"][name]
        path = ROOT / "models" / spec["file"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), spec["sha256"])
        return BehavioralRuntime(joblib.load(path), spec["method"], spec["threshold"])

    def records(self):
        return [
            CompletedFlow(t, 1, 2, 80, (6, 10, 2, 1, 120, 60, 0, 0, 40, 80, 18))
            for t in [100, 100, 101, 200, 1000]
        ]

    def test_frozen_model_scores_do_not_depend_on_batch_partition(self):
        for name in ["flow", "hybrid"]:
            a, b = self.load_runtime(name), self.load_runtime(name)
            records = self.records()
            bulk = a.score_batch(records)
            single = [b.score_batch([r]) for r in records]
            np.testing.assert_allclose(
                bulk["scores"],
                np.concatenate([r["scores"] for r in single]),
                rtol=0,
                atol=1e-14,
            )
            np.testing.assert_equal(
                bulk["alerts"], np.concatenate([r["alerts"] for r in single])
            )

    def test_invalid_batch_is_rejected_before_mutating_history(self):
        runtime = self.load_runtime("hybrid")
        records = self.records()
        with self.assertRaises(ValueError):
            runtime.score_batch([records[-1], records[0]])
        self.assertEqual(runtime.observation.records, 0)
        self.assertIsNone(runtime.last_arrival)
        with self.assertRaises(ValueError):
            runtime.score_batch([records[0]] * 1001)
        self.assertEqual(runtime.observation.records, 0)


if __name__ == "__main__":
    unittest.main()
