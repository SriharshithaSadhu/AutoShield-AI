"""Deadline routing with flow-weighted history separate from delivery batches."""
from collections import deque
from time import perf_counter_ns
import numpy as np


def batches(times, capacity=1000, deadline_ns=10**9):
    first = 0
    while first < len(times):
        deadline = int(times[first]) + deadline_ns
        stop = min(first + capacity, int(np.searchsorted(times, deadline, side="right")))
        release = int(times[stop-1]) if stop-first == capacity else min(deadline, int(times[-1]))
        yield first, stop, release
        first = stop


class DeadlineRouter:
    def __init__(self, evidence_flows=50, recovery_flows=0, minimum_dwell_seconds=0):
        self.evidence_flows = evidence_flows
        self.recovery_flows = recovery_flows
        self.minimum_dwell_ns = minimum_dwell_seconds * 10**9
        self.recoveries = []
        self.recovery_reference = 0.
        self.stable_flows = 0
        self.recent = deque(maxlen=evidence_flows)
        self.model = None
        self.history = deque(maxlen=3000)
        self.history_positives = 0
        self.latched = False
        self.events = []
        self.suppressed_attacks = 0
        self.waits = []
        self.batches = 0
        self.service_ns = []
        self.delivery_ns = []
        self.alert_latency = []
        self.finished_ns = None

    def route(self, model, times, candidate, frozen, infer=None, rate=1):
        if model is not self.model:
            self.history.clear()
            self.recent.clear()
            self.history_positives = 0
            self.latched = False
            self.stable_flows = 0
            self.model = model
        output = candidate.copy()
        for left, right, release in batches(times):
            started = perf_counter_ns()
            values = candidate[left:right]
            fallback = frozen[left:right]
            if infer is not None:
                batch_candidate, batch_frozen = infer(left, right)
                assert np.array_equal(batch_candidate, values)
                assert np.array_equal(batch_frozen, fallback)
                values, fallback = batch_candidate, batch_frozen
                output[left:right] = values
            self.batches += 1
            self.waits.append((release-times[left:right]).astype(np.int32))
            self.recent.extend(values.tolist())
            evidence = self.recent if self.evidence_flows > 50 else values
            evidence_rate = sum(evidence)/len(evidence)
            if len(self.history) == 3000 and len(evidence) >= self.evidence_flows and evidence_rate - self.history_positives/3000 > .10:
                if not self.latched:
                    self.events.append(release)
                    self.recovery_reference = self.history_positives/3000
                    self.stable_flows = 0
                self.latched = True
            elif self.latched and self.recovery_flows:
                if len(evidence) >= self.evidence_flows and evidence_rate <= self.recovery_reference + .03:
                    self.stable_flows += len(values)
                    if (self.stable_flows >= self.recovery_flows
                            and release - self.events[-1] >= self.minimum_dwell_ns):
                        self.latched = False
                        self.recoveries.append(release)
                else:
                    self.stable_flows = 0
            if self.latched:
                output[left:right] = fallback
            overflow = max(0, len(self.history) + len(values) - self.history.maxlen)
            self.history_positives -= sum(self.history.popleft() for _ in range(overflow))
            self.history.extend(values.tolist())
            self.history_positives += int(values.sum())
            if infer is not None:
                delivery_started = perf_counter_ns()
                payload = np.flatnonzero(output[left:right]).astype("<u4").tobytes()
                # In-memory receiver acknowledges the serialized alert indices.
                received = np.frombuffer(payload, dtype="<u4")
                self.delivery_ns.append(perf_counter_ns() - delivery_started)
                service = perf_counter_ns() - started
                self.service_ns.append(service)
                scaled_release = release // rate
                self.finished_ns = max(scaled_release, self.finished_ns or scaled_release) + service
                self.alert_latency.extend((self.finished_ns - times[left:right][received] // rate).tolist())
        return output
