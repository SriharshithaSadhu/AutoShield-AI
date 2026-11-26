"""Measure when NF-ToN-IoT-v3 completed-flow records become scoreable.

The file is start-time ordered. This audit compares the timestamps at which
flows begin and end, and tracks campaign support by each timestamp. It does
not fit or score a detector and makes no causal performance claim.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "nf_ton_iot_v3" / "NF-ToN-IoT-v3.csv"
OUT_JSON = ROOT / "results" / "nf_ton_iot_v3_flow_availability_audit.json"
OUT_HOURLY = ROOT / "results" / "nf_ton_iot_v3_start_end_hourly_counts.csv"
CHUNK = 400_000


def utc_hour(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


def main() -> None:
    fields = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Label", "Attack"]
    lag_counts: Counter[str] = Counter()
    family_bounds: dict[str, dict[str, int]] = {}
    hourly: Counter[tuple[str, str, str]] = Counter()
    totals = Counter()
    lag_sum = lag_sum_sq = 0
    lag_min = None
    lag_max = 0
    rows = 0
    prev_start = None
    prev_end = None
    start_inversions = end_inversions = 0
    for frame in pd.read_csv(SOURCE, usecols=fields, chunksize=CHUNK, low_memory=False):
        start = pd.to_numeric(frame[fields[0]], errors="raise").to_numpy(dtype="int64")
        end = pd.to_numeric(frame[fields[1]], errors="raise").to_numpy(dtype="int64")
        labels = pd.to_numeric(frame["Label"], errors="raise").to_numpy(dtype="int8")
        families = frame["Attack"].fillna("<missing>").astype(str).to_numpy()
        lag = end - start
        if (lag < 0).any():
            raise ValueError("Found flow end before flow start")
        if prev_start is not None and start[0] < prev_start:
            start_inversions += 1
        if prev_end is not None and end[0] < prev_end:
            end_inversions += 1
        start_inversions += int((start[1:] < start[:-1]).sum())
        end_inversions += int((end[1:] < end[:-1]).sum())
        prev_start = int(start[-1])
        prev_end = int(end[-1])
        lag_min = int(lag.min()) if lag_min is None else min(lag_min, int(lag.min()))
        lag_max = max(lag_max, int(lag.max()))
        lag_sum += int(lag.sum(dtype="int64"))
        lag_sum_sq += int((lag.astype("float64") ** 2).sum())
        rows += len(frame)
        totals.update(f"label_{int(y)}" for y in labels)
        for bucket, mask in (("0-1s", lag <= 1000), ("1-10s", (lag > 1000) & (lag <= 10000)),
                             ("10-60s", (lag > 10000) & (lag <= 60000)),
                             ("1-10m", (lag > 60000) & (lag <= 600000)),
                             (">10m", lag > 600000)):
            lag_counts[bucket] += int(mask.sum())
        local = pd.DataFrame({"start": start, "end": end, "family": families,
                              "label": np.where(labels == 1, "attack", "benign")})
        local["start_hour"] = pd.to_datetime(local["start"], unit="ms", utc=True).dt.floor("h")
        local["end_hour"] = pd.to_datetime(local["end"], unit="ms", utc=True).dt.floor("h")
        for fam, group in local.groupby("family", sort=False):
            bound = family_bounds.setdefault(str(fam), {"start_min": int(group.start.min()),
                "start_max": int(group.start.max()), "end_min": int(group.end.min()),
                "end_max": int(group.end.max())})
            bound["start_min"] = min(bound["start_min"], int(group.start.min()))
            bound["start_max"] = max(bound["start_max"], int(group.start.max()))
            bound["end_min"] = min(bound["end_min"], int(group.end.min()))
            bound["end_max"] = max(bound["end_max"], int(group.end.max()))
        for time_col, basis in (("start_hour", "start"), ("end_hour", "end")):
            counts = local.groupby([time_col, "label", "family"], sort=False).size()
            for (hour, label, fam), count in counts.items():
                hourly[(basis + ":" + hour.strftime("%Y-%m-%dT%H:00:00Z"),
                        str(label), str(fam))] += int(count)
        if rows % 2_000_000 < CHUNK:
            print(f"rows={rows:,}", flush=True)

    mean = lag_sum / rows
    variance = max(0.0, lag_sum_sq / rows - mean * mean)
    by_hour: Counter[tuple[str, str, str, str]] = Counter()
    for (time_key, label, fam), count in hourly.items():
        if time_key.startswith("start:"):
            by_hour[("start", time_key[6:], label, fam)] += count
        elif time_key.startswith("end:"):
            by_hour[("end", time_key[4:], label, fam)] += count
    pd.DataFrame([{"basis": basis, "utc_hour": hour, "label": label,
                   "attack_family": family, "rows": count}
                  for (basis, hour, label, family), count in sorted(by_hour.items())]
                 ).to_csv(OUT_HOURLY, index=False)
    result = {
        "audit": "NF-ToN-IoT-v3 start versus flow-completion availability",
        "rows": rows,
        "source_order": "publisher CSV, sorted by flow start time",
        "adjacent_start_time_descents": start_inversions,
        "adjacent_end_time_descents_in_start_order": end_inversions,
        "flow_end_minus_start_ms": {
            "min": lag_min, "mean": mean, "sd": variance ** 0.5, "max": lag_max,
            "percentile_counts": dict(lag_counts),
        },
        "attack_family_start_end_availability_utc": {
            fam: {key: datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()
                  for key, ms in bounds.items()}
            for fam, bounds in sorted(family_bounds.items())
        },
        "interpretation": (
            "Start time describes when a flow begins; final aggregate flow features are only "
            "available after its end. The timestamp lag and start/end campaign boundaries "
            "quantify availability delay, but do not by themselves establish changed detector outcomes."
        ),
        "hourly_counts_csv": str(OUT_HOURLY.relative_to(ROOT)),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_HOURLY}")


if __name__ == "__main__":
    main()
