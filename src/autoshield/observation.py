"""Bounded causal histories for completed exporter records, never labels.

`arrival_ms` is collector receipt time. Replays use completion-time order only
as an explicitly idealized arrival proxy; real export timestamps are absent.
Records sharing arrival_ms query an immutable pre-group state. At most a fixed
number of hosts and most-recent distinct contacts are retained.
"""

from collections import OrderedDict
import heapq, math
import numpy as np

TAUS = (10.0, 60.0)
FEATURE_NAMES = []
for tau in [10, 60]:
    FEATURE_NAMES.extend(
        [
            f"log_src_record_rate_{tau}s",
            f"log_src_incoming_rate_{tau}s",
            f"log_tcp_failure_proxy_rate_{tau}s",
            f"tcp_failure_proxy_fraction_{tau}s",
            f"log_unanswered_udp_rate_{tau}s",
            f"unanswered_udp_fraction_{tau}s",
            f"short_record_fraction_{tau}s",
            f"log_destination_incoming_rate_{tau}s",
        ]
    )
for window in [10000, 60000]:
    FEATURE_NAMES.extend(
        [
            f"log_capped_distinct_ports_{window}ms",
            f"log_capped_distinct_destinations_{window}ms",
        ]
    )
FEATURE_NAMES.extend(
    [
        "log_source_gap_ms_capped60s",
        "log_destination_gap_ms_capped60s",
        "source_has_history60s",
        "destination_has_history60s",
    ]
)


def priority(x):
    x = int(x) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


class Host:
    __slots__ = ("t", "values", "ports", "destinations", "last_out", "last_in")

    def __init__(self, t):
        self.t = t
        self.values = [0.0] * 14
        self.ports = OrderedDict()
        self.destinations = OrderedDict()
        self.last_out = None
        self.last_in = None


class Pending:
    __slots__ = ("counts", "ports", "destinations", "records")

    def __init__(self):
        self.counts = [0] * 7
        self.ports = OrderedDict()
        self.destinations = OrderedDict()
        self.records = 0


def touch(table, key, value, cap):
    table[key] = value
    table.move_to_end(key)
    if len(table) > cap:
        table.popitem(last=False)


class BehavioralObservation:
    def __init__(self, max_hosts=4096, contact_cap=64):
        if max_hosts < 1 or contact_cap < 1:
            raise ValueError("positive state caps required")
        self.max_hosts = max_hosts
        self.contact_cap = contact_cap
        self.hosts = OrderedDict()
        self.pending = {}
        self.pending_heap = []
        self.clock = None
        self.source_cache = {}
        self.destination_cache = {}
        self.evictions = 0
        self.pending_dropped_host_updates = 0
        self.max_hosts_seen = 0
        self.max_pending_seen = 0
        self.records = 0
        self.global_values = [0.0, 0.0]
        self.global_t = None
        self.pending_records = 0

    def _pending(self, host):
        if host in self.pending:
            return self.pending[host]
        rank = priority(host)
        if len(self.pending) >= self.max_hosts:
            if rank >= -self.pending_heap[0][0]:
                self.pending_dropped_host_updates += 1
                return None
            _, removed = heapq.heappop(self.pending_heap)
            self.pending_dropped_host_updates += self.pending[removed].records
            del self.pending[removed]
        p = Pending()
        self.pending[host] = p
        heapq.heappush(self.pending_heap, (-rank, host))
        self.max_pending_seen = max(self.max_pending_seen, len(self.pending))
        return p

    def _advance(self, t):
        if self.clock is not None and t < self.clock:
            raise ValueError(
                "arrival clock must be monotone; end times may arrive out of order"
            )
        if t == self.clock:
            return
        if self.clock is not None:
            previous = self.clock
            dt = 0.0 if self.global_t is None else (previous - self.global_t) / 1000.0
            self.global_values = [
                v * math.exp(-dt / tau) + self.pending_records
                for v, tau in zip(self.global_values, TAUS)
            ]
            self.global_t = previous
            for host in sorted(self.pending, key=priority):
                p = self.pending[host]
                if host in self.hosts:
                    h = self.hosts[host]
                    self.hosts.move_to_end(host)
                else:
                    if len(self.hosts) >= self.max_hosts:
                        self.hosts.popitem(last=False)
                        self.evictions += 1
                    h = Host(previous)
                    self.hosts[host] = h
                dt = (previous - h.t) / 1000.0
                for j, tau in enumerate(TAUS):
                    factor = math.exp(-dt / tau)
                    for k in range(7):
                        h.values[j * 7 + k] = h.values[j * 7 + k] * factor + p.counts[k]
                h.t = previous
                if p.counts[0]:
                    h.last_out = previous
                if p.counts[6]:
                    h.last_in = previous
                for key in p.ports:
                    touch(h.ports, key, previous, self.contact_cap)
                for key in p.destinations:
                    touch(h.destinations, key, previous, self.contact_cap)
            self.max_hosts_seen = max(self.max_hosts_seen, len(self.hosts))
        self.pending = {}
        self.pending_heap = []
        self.source_cache = {}
        self.destination_cache = {}
        self.clock = t
        self.pending_records = 0

    def _source(self, t, source):
        cached = self.source_cache.get(source)
        if cached is not None:
            return cached
        h = self.hosts.get(source)
        if h is None:
            return (0.0,) * 14 + (0.0, 0.0, 0.0, 0.0, math.log1p(60000), 0.0)
        dt = (t - h.t) / 1000.0
        values = []
        for j, tau in enumerate(TAUS):
            factor = math.exp(-dt / tau)
            v = [h.values[j * 7 + k] * factor for k in range(7)]
            values.extend(
                [
                    math.log1p(v[0] / tau),
                    math.log1p(v[6] / tau),
                    math.log1p(v[2] / tau),
                    v[2] / (v[1] + 1.0),
                    math.log1p(v[4] / tau),
                    v[4] / (v[3] + 1.0),
                    v[5] / (v[0] + 1.0),
                ]
            )
        for window in [10000, 60000]:
            boundary = t - window
            values.extend(
                [
                    math.log1p(sum(tt >= boundary for tt in h.ports.values())),
                    math.log1p(sum(tt >= boundary for tt in h.destinations.values())),
                ]
            )
        gap = 60000 if h.last_out is None else min(60000, t - h.last_out)
        values.extend(
            [math.log1p(gap), float(h.last_out is not None and t - h.last_out <= 60000)]
        )
        result = tuple(values)
        self.source_cache[source] = result
        return result

    def _destination(self, t, destination):
        cached = self.destination_cache.get(destination)
        if cached is not None:
            return cached
        h = self.hosts.get(destination)
        if h is None:
            return (0.0, 0.0, math.log1p(60000), 0.0)
        dt = (t - h.t) / 1000.0
        rates = tuple(
            math.log1p(h.values[j * 7 + 6] * math.exp(-dt / tau) / tau)
            for j, tau in enumerate(TAUS)
        )
        gap = 60000 if h.last_in is None else min(60000, t - h.last_in)
        result = (
            *rates,
            math.log1p(gap),
            float(h.last_in is not None and t - h.last_in <= 60000),
        )
        self.destination_cache[destination] = result
        return result

    def observe(
        self,
        arrival_ms,
        source,
        destination,
        destination_port,
        protocol,
        tcp_flags,
        in_packets,
        out_packets,
        duration_ms,
        query=True,
    ):
        t = int(arrival_ms)
        source = int(source)
        destination = int(destination)
        self._advance(t)
        if query:
            a = self._source(t, source)
            b = self._destination(t, destination)
            # 14 source EMA values, four cardinalities, gaps and history flags.
            result = (*a[:7], b[0], *a[7:14], b[1], *a[14:18], a[18], b[2], a[19], b[3])
        else:
            result = None
        p = self._pending(source)
        if p is not None:
            tcp = int(protocol) == 6
            udp = int(protocol) == 17
            flags = int(tcp_flags)
            failure = tcp and bool(
                (flags & 4) or ((flags & 2) and not (flags & 16) and out_packets == 0)
            )
            p.counts[0] += 1
            p.counts[1] += int(tcp)
            p.counts[2] += int(failure)
            p.counts[3] += int(udp)
            p.counts[4] += int(udp and out_packets == 0)
            p.counts[5] += int(in_packets + out_packets <= 2 and duration_ms <= 100)
            p.records += 1
            touch(p.ports, int(destination_port), t, self.contact_cap)
            touch(p.destinations, destination, t, self.contact_cap)
        p = self._pending(destination)
        if p is not None:
            p.counts[6] += 1
            p.records += 1
        self.records += 1
        self.pending_records += 1
        return result

    def global_features(self):
        if self.global_t is None:
            return (0.0, 0.0)
        dt = (self.clock - self.global_t) / 1000.0
        return tuple(
            math.log1p(v * math.exp(-dt / tau) / tau)
            for v, tau in zip(self.global_values, TAUS)
        )

    def diagnostics(self):
        return dict(
            records=self.records,
            state_hosts=len(self.hosts),
            pending_hosts=len(self.pending),
            max_hosts=self.max_hosts,
            contact_cap=self.contact_cap,
            evictions=self.evictions,
            pending_dropped_host_updates=self.pending_dropped_host_updates,
            maximum_state_hosts=self.max_hosts_seen,
            maximum_pending_hosts=self.max_pending_seen,
            retained_contact_entries=sum(
                len(h.ports) + len(h.destinations) for h in self.hosts.values()
            ),
        )
