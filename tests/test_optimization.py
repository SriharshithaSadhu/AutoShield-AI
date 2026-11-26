import unittest

from autoshield.optimization import Dimension, _dominates


class OptimizationTests(unittest.TestCase):
    def record(self, feasible, f1):
        return {
            "metrics": {
                "weighted_f1": f1,
                "macro_f1": f1,
                "top_label_ece": 0.02,
                "p95_latency_ms_per_row": 0.2,
                "serialized_size_mb": 0.2,
                "mean_fit_seconds": 1,
            },
            "feasibility": {
                "feasible": feasible,
                "constraints": {
                    "min_weighted_f1": 0.95,
                    "max_p95_latency_ms_per_row": 5,
                    "max_serialized_size_mb": 1.5,
                },
            },
        }

    def test_feasible_candidate_dominates_infeasible_candidate(self):
        a, b = self.record(True, 0.96), self.record(False, 0.94)
        self.assertTrue(_dominates(a, b))
        self.assertFalse(_dominates(b, a))

    def test_dimension_decoding_and_invalid_search_bounds(self):
        d = Dimension("trees", 10, 100, "int")
        self.assertEqual(d.decode(-1), 10)
        self.assertEqual(d.decode(2), 100)
        with self.assertRaises(ValueError):
            Dimension("invalid", 10, 10)


if __name__ == "__main__":
    unittest.main()
