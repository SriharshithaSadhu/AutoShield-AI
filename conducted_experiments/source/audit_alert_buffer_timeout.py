"""Measure event-time buffering under prospective size and timeout limits."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def main():
    result = {}
    for name in ("cse_original_replay_core6", "cse_corrected/replay_core6"):
        with np.load(ROOT / "data" / name / "stream.npz") as stream:
            times, offsets = stream["completion_ns"], stream["chunk_offsets"]
        start = 2 * (len(offsets) - 1) // 3
        rows = {}
        for size, timeout in ((20000, None), (1000, None), (1000, 1), (1000, 5)):
            waits = []
            sizes = []
            for left, right in zip(offsets[start:-1], offsets[start+1:]):
                chunk = times[left:right]
                first = 0
                while first < len(chunk):
                    stop = min(first + size, len(chunk))
                    release = chunk[stop-1]
                    if timeout is not None:
                        deadline = chunk[first] + timeout * 10**9
                        time_stop = np.searchsorted(chunk, deadline, side="right")
                        if time_stop < stop or (stop == len(chunk) and stop-first < size):
                            stop = min(time_stop, stop)
                            release = min(deadline, chunk[-1])
                    waits.append((release - chunk[first:stop]) / 1e9)
                    sizes.append(stop-first)
                    first = stop
            sampled = np.concatenate(waits)
            rows[f"{size}/{timeout}"] = dict(p95_seconds=float(np.quantile(sampled,.95)),
                maximum_seconds=float(sampled.max()), batches=len(sizes), median_batch=float(np.median(sizes)))
        result[name] = rows
    path = ROOT / "results/corrected_cse_gate/buffer_timeout_diagnostic.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
