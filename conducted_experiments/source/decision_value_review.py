"""Exact one-step expected decision value for two disagreement strata."""
from functools import lru_cache

import numpy as np
from scipy.stats import betabinom


@lru_cache(maxsize=128)
def posterior_grid(sample, population):
    attacks = np.arange(sample + 1)
    expected = attacks + (population - sample) * (attacks + 1) / (sample + 2)
    return expected, betabinom.pmf(attacks, sample, 1, 1)


@lru_cache(maxsize=4096)
def allocation(added, removed, budget, false_alarm_cost=1.0):
    """Maximize expected positive utility after observing the sampled labels."""
    budget = min(budget, added + removed)
    best = (-np.inf, 0)
    for plus in range(max(0, budget - removed), min(budget, added) + 1):
        minus = budget - plus
        a, pa = posterior_grid(plus, added)
        r, pr = posterior_grid(minus, removed)
        utility = (1 + false_alarm_cost) * (a[:, None] - r) - false_alarm_cost * (added - removed)
        value = float(pa @ np.maximum(utility, 0) @ pr)
        best = max(best, (value, plus))
    return best[1], budget - best[1]
