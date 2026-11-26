"""Measure dual-model deadline routing on a desktop or Raspberry Pi."""
import argparse
import json
import platform
from pathlib import Path
from time import process_time

import joblib
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from deadline_alert_router import DeadlineRouter
from five_track_common import probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-flows", type=int, choices=(0, 1000, 3000), default=0)
    parser.add_argument("--dwell-seconds", type=int, choices=(0, 10, 60), default=0)
    args = parser.parse_args()
    model_a, model_b = joblib.load(args.first), joblib.load(args.second)
    with np.load(args.fixture) as fixture:
        features = fixture["x"]
    process = psutil.Process()
    rows = []
    with threadpool_limits(limits=1):
        for flows_per_second in (20, 140, 1000, 10000):
            x = features[np.arange(20000) % len(features)]
            t = np.arange(len(x), dtype=np.int64)*10**9//flows_per_second
            def infer(left, right):
                return probabilities(model_b, x[left:right]) > .5, probabilities(model_a, x[left:right]) > .5
            candidate, frozen = infer(0, len(x))
            router = DeadlineRouter(1000, args.recovery_flows, args.dwell_seconds)
            start = process_time()
            router.route(model_b, t, candidate, frozen, infer)
            rows.append(dict(flows_per_second=flows_per_second,
                             batches=router.batches, cpu_seconds=process_time()-start,
                             service_p95_ms=float(np.quantile(router.service_ns,.95)/1e6),
                             observed_rss_bytes=process.memory_info().rss,
                             delivery_p95_ms=float(np.quantile(router.delivery_ns,.95)/1e6)))
    result = dict(machine=platform.machine(), platform=platform.platform(), recovery_flows=args.recovery_flows,
                  dwell_seconds=args.dwell_seconds,
                  bundle_bytes=args.first.stat().st_size+args.second.stat().st_size,
                  scope="Repeated fixture at regular synthetic arrival rates; measured compute and local serialization. No detection claims.",
                  results=rows)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
