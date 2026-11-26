"""Matched-budget replay of simulated early verification with explicit label release."""
from collections import deque
from functools import lru_cache

import numpy as np
from scipy.stats import hypergeom


@lru_cache(maxsize=4096)
def lower_attack_share(population, sample_size, observed_attacks):
    """Invert the upper hypergeometric tail for one sampled finite population."""
    if population == sample_size:
        return observed_attacks / population
    low, high = 0, population
    while low < high:
        mid = (low + high) // 2
        if hypergeom.sf(observed_attacks - 1, population, mid, sample_size) <= .05:
            low = mid + 1
        else:
            high = mid
    return low / population


class VerificationPolicy:
    def __init__(self, strategy, budget=128, turnaround=1, seed=42):
        self.strategy = strategy
        self.budget = budget
        self.turnaround = turnaround
        self.rng = np.random.default_rng(seed)
        self.pending = deque()
        self.evidence = None
        self.model = None
        self.reviewed = 0
        self.switches = 0
        self.last_used = False
        self.insufficient_evidence = 0

    def route(self, chunk, model, active, candidate, frozen, labels):
        if model is not self.model:
            self.pending.clear()
            self.evidence = None
            self.model = model
        while self.pending and self.pending[0][0] <= chunk:
            _, outcomes, disagreements, population, removed_outcomes, removed_population = self.pending.popleft()
            self.evidence = outcomes[disagreements]
            self.evidence_population = population
            self.removed_evidence = removed_outcomes
            self.removed_population = removed_population
        supported = active and self.evidence is not None and len(self.evidence) >= 8
        evidence_share = 0.
        if supported:
            evidence_share = (lower_attack_share(self.evidence_population, len(self.evidence),
                                                 int(self.evidence.sum()))
                              if self.strategy == "disagreement_bound" else self.evidence.mean())
        use_candidate = bool(supported and evidence_share >= .90)
        if self.strategy in ("value", "balanced_value") and active and self.evidence is not None:
            def posterior_total(outcomes, population):
                return outcomes.sum() + (population-len(outcomes)) * (outcomes.sum()+1)/(len(outcomes)+2)
            delta_tp = (posterior_total(self.evidence, self.evidence_population)
                        - posterior_total(self.removed_evidence, self.removed_population))
            supported = True
            use_candidate = bool(2*delta_tp > self.evidence_population-self.removed_population)
        if self.strategy == "paired" and active and self.evidence is not None:
            # Require at least four reviews per nonempty stratum, or a census.
            supported = (len(self.evidence) >= min(4, self.evidence_population)
                         and len(self.removed_evidence) >= min(4, self.removed_population))
            if supported:
                added_tp = self.evidence.mean() * self.evidence_population if len(self.evidence) else 0.
                removed_tp = (self.removed_evidence.mean() * self.removed_population
                              if len(self.removed_evidence) else 0.)
                delta_tp = added_tp - removed_tp
                delta_fp = self.evidence_population - self.removed_population - delta_tp
                use_candidate = bool(delta_tp >= 0 and delta_fp <= 0)
        if active and not supported:
            self.insufficient_evidence += 1
        self.switches += bool(use_candidate != self.last_used)
        self.last_used = use_candidate
        if active:
            added = np.flatnonzero(candidate & ~frozen)
            removed = np.flatnonzero(frozen & ~candidate)
            removed_outcomes = np.empty(0, dtype=labels.dtype)
            if self.strategy in ("paired", "value", "balanced_value"):
                n_removed = min(len(removed), self.budget//2)
                n_added = min(len(added), self.budget-n_removed)
                n_removed = min(len(removed), self.budget-n_added)
                if self.strategy == "value":
                    from decision_value_review import allocation
                    n_added, n_removed = allocation(len(added), len(removed), self.budget)
                added_sample = self.rng.choice(added, n_added, replace=False)
                removed_sample = self.rng.choice(removed, n_removed, replace=False)
                removed_outcomes = labels[removed_sample].copy()
                remaining = min(len(candidate), self.budget)-n_removed-n_added
                agreements = np.flatnonzero(candidate==frozen)
                selected = np.r_[added_sample, removed_sample,
                                 self.rng.choice(agreements, remaining, replace=False)]
            elif self.strategy.startswith("disagreement"):
                selected = self.rng.choice(added, min(len(added), self.budget), replace=False)
                remaining = min(len(candidate), self.budget) - len(selected)
                if remaining:
                    others = np.flatnonzero(~(candidate & ~frozen))
                    selected = np.r_[selected, self.rng.choice(others, remaining, replace=False)]
            else:
                selected = self.rng.choice(len(candidate), min(len(candidate), self.budget), replace=False)
            self.reviewed += len(selected)
            # Simulated reviewer queue. Labels enter the decision only on release.
            self.pending.append((chunk + self.turnaround, labels[selected].copy(),
                                 (candidate[selected] & ~frozen[selected]).copy(), len(added),
                                 removed_outcomes, len(removed)))
        return candidate if use_candidate else frozen
