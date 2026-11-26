import unittest
import numpy as np
from autoshield.features import transform
from autoshield.runtime import RAW_FIELDS


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array(
            [
                [6, 10, 2, 1, 100, 50, 1, 0, 40, 60, 18],
                [17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

    def test_directional_ratios_and_tcp_flags(self):
        values = transform(self.x, list(RAW_FIELDS), "physical14")
        self.assertEqual(values.shape, (2, 14))
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(float(values[0, 6]), 1 / 3)
        self.assertEqual(values[0, 12], 1)
        self.assertEqual(values[0, 13], 1)
        self.assertEqual(values[1, 6], 0)

    def test_missing_values_do_not_corrupt_other_rows(self):
        self.x[0, 4] = np.nan
        values = transform(self.x, list(RAW_FIELDS), "physical14")
        self.assertTrue(np.isnan(values[0, 3]))
        self.assertTrue(np.isnan(values[0, 4]))
        self.assertTrue(np.isfinite(values[1]).all())

    def test_duration_harmonization_changes_only_duration(self):
        a = transform(self.x, list(RAW_FIELDS), "physical14")
        b = transform(self.x, list(RAW_FIELDS), "harmonized14")
        self.assertAlmostEqual(float(b[1, 1]), np.log(2))
        np.testing.assert_allclose(b[:, np.arange(14) != 1], a[:, np.arange(14) != 1])


if __name__ == "__main__":
    unittest.main()
