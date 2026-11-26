"""Conventional low-cost benign-profile anomaly score, with no update controller."""

import numpy as np


class QuantileTailAnomaly:
    # Fixed high-tail rate/cardinality dimensions. No outcome-driven selection.
    columns = np.array([0, 2, 7, 8, 10, 15, 16, 17, 18, 19])

    def fit(self, x):
        values = np.asarray(x)[:, self.columns]
        self.center = np.quantile(values, 0.5, axis=0)
        q99 = np.quantile(values, 0.99, axis=0)
        self.scale = np.maximum(q99 - self.center, 0.1)
        return self

    def score(self, x):
        values = np.asarray(x)[:, self.columns]
        return np.max((values - self.center) / self.scale, axis=1)


def scores(model, x):
    return (
        model.score(x)
        if isinstance(model, QuantileTailAnomaly)
        else model.booster_.predict(np.ascontiguousarray(x), num_threads=1)
    )
