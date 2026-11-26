import unittest, math
import numpy as np
from autoshield.observation import BehavioralObservation, FEATURE_NAMES


def event(engine, t, s=1, d=2, p=80, proto=6, flags=2, ipk=1, opk=0, dur=1):
    return np.array(engine.observe(t, s, d, p, proto, flags, ipk, opk, dur))


class BehavioralTests(unittest.TestCase):
    def test_same_timestamp_has_no_peer_information(self):
        b = BehavioralObservation()
        a = event(b, 100)
        c = event(b, 100)
        np.testing.assert_equal(a, c)
        later = event(b, 101)
        self.assertAlmostEqual(later[0], math.log1p(2 * math.exp(-0.001 / 10) / 10))
        self.assertEqual(len(later), len(FEATURE_NAMES))

    def test_matches_unbounded_causal_reference_with_contact_clipping(self):
        b = BehavioralObservation(max_hosts=128, contact_cap=3)
        rng = np.random.default_rng(13)
        past = []
        for i in range(1000):
            t = (i // 5) * 31 + (90000 if i >= 600 else 0)
            s = int(rng.integers(1, 4))
            d = int(rng.integers(10, 30))
            p = int(rng.integers(1, 20))
            out = event(b, t, s, d, p)
            previous = [e for e in past if e[0] < t and e[1] == s]
            for w, pi, di in [(10000, 16, 17), (60000, 18, 19)]:
                records = [e for e in previous if e[0] >= t - w]
                self.assertAlmostEqual(
                    out[pi], math.log1p(min(3, len({e[3] for e in records})))
                )
                self.assertAlmostEqual(
                    out[di], math.log1p(min(3, len({e[2] for e in records})))
                )
            expected = sum(math.exp(-(t - e[0]) / 10000) for e in previous) / 10
            self.assertAlmostEqual(out[0], math.log1p(expected), places=9)
            past.append((t, s, d, p))

    def test_time_shift_and_endpoint_renaming_below_capacity(self):
        a = BehavioralObservation()
        b = BehavioralObservation()
        records = [
            (0, 1, 2, 80),
            (0, 1, 3, 443),
            (10, 2, 1, 22),
            (100, 1, 2, 80),
            (70000, 1, 3, 53),
        ]
        for t, s, d, p in records:
            np.testing.assert_allclose(
                event(a, t, s, d, p),
                event(b, t + 1000000, s + 200, d + 200, p),
                rtol=0,
                atol=1e-10,
            )

    def test_timestamp_group_permutation(self):
        a = BehavioralObservation(contact_cap=3)
        b = BehavioralObservation(contact_cap=3)
        group = [(1, 10 + i, 100 + i) for i in range(10)]
        for s, d, p in group:
            event(a, 10, s, d, p)
        for s, d, p in reversed(group):
            event(b, 10, s, d, p)
        for t, p in [(20, 100), (30, 105), (10011, 106), (60011, 107)]:
            np.testing.assert_allclose(event(a, t, 1, 99, p), event(b, t, 1, 99, p))

    def test_memory_admission_caps_and_arrival_clock(self):
        b = BehavioralObservation(max_hosts=8, contact_cap=3)
        for i in range(1000):
            event(b, 0, i + 1, 2000 + i, i % 65536)
        self.assertLessEqual(len(b.pending), 8)
        event(b, 1)
        self.assertLessEqual(len(b.hosts), 8)
        self.assertGreater(b.pending_dropped_host_updates, 0)
        with self.assertRaises(ValueError):
            event(b, 0)

    def test_query_skipping_does_not_change_later_features(self):
        a = BehavioralObservation()
        b = BehavioralObservation()
        for i in range(100):
            args = (i // 3, 1 + i % 3, 10 + i % 7, 80 + i % 5, 6, 2, 1, 0, 1)
            value = np.array(a.observe(*args))
            other = b.observe(*args, query=i % 7 == 0)
            if i % 7 == 0:
                np.testing.assert_equal(value, np.array(other))
        np.testing.assert_allclose(a.global_features(), b.global_features())

    def test_memo_clearing_preserves_features_and_caches_stay_bounded(self):
        # Memoized values refer only to the bounded immutable pre-group host
        # table. Clearing memoization changes cost, not feature semantics.
        a = BehavioralObservation(max_hosts=8, contact_cap=3)
        for i in range(100):
            event(a, i, i % 8 + 1, i % 8 + 1)
        for t in [100, 101]:
            for i in range(100):
                value = event(a, t, i % 8 + 1, (i + 1) % 8 + 1)
                a.source_cache.clear()
                a.destination_cache.clear()
                other = event(a, t, i % 8 + 1, (i + 1) % 8 + 1)
                np.testing.assert_equal(value, other)
                self.assertLessEqual(len(a.source_cache), 8)
                self.assertLessEqual(len(a.destination_cache), 8)


if __name__ == "__main__":
    unittest.main()
