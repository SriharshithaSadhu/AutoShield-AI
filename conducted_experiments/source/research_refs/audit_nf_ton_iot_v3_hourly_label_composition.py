"""Summarize NF-ToN-IoT-v3 attack families by UTC hour of flow completion."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "nf_ton_iot_v3" / "NF-ToN-IoT-v3_endtime_ordered.csv"
OUT = ROOT / "results"
CHUNK_ROWS = 1_000_000

counts: dict[tuple[str, int, str], int] = defaultdict(int)
row_count = 0
min_end = max_end = None

for frame in pd.read_csv(
    SOURCE,
    usecols=["FLOW_END_MILLISECONDS", "Label", "Attack"],
    chunksize=CHUNK_ROWS,
    low_memory=False,
):
    ends = pd.to_numeric(frame["FLOW_END_MILLISECONDS"], errors="coerce")
    labels = pd.to_numeric(frame["Label"], errors="coerce")
    hours = pd.to_datetime(ends, unit="ms", utc=True).dt.floor("h")
    attacks = frame["Attack"].fillna("<MISSING>").astype(str).str.strip()
    grouped = pd.DataFrame({
        "hour_utc": hours.dt.strftime("%Y-%m-%dT%H:00:00Z"),
        "binary_label": labels.astype("int8"),
        "attack": attacks,
    }).groupby(["hour_utc", "binary_label", "attack"], sort=False).size()
    for (hour, label, attack), n in grouped.items():
        counts[(hour, int(label), attack)] += int(n)
    valid = ends.dropna()
    if len(valid):
        lo, hi = int(valid.min()), int(valid.max())
        min_end = lo if min_end is None else min(min_end, lo)
        max_end = hi if max_end is None else max(max_end, hi)
    row_count += len(frame)
    print(f"rows_scanned={row_count:,} hour_category_cells={len(counts)}", flush=True)

records = [
    {"hour_utc": hour, "binary_label": label, "attack": attack, "rows": n}
    for (hour, label, attack), n in sorted(counts.items())
]
table_path = OUT / "nf_ton_iot_v3_hourly_label_counts.csv"
pd.DataFrame(records).to_csv(table_path, index=False)

family_hours = defaultdict(list)
hour_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"benign": 0, "attack": 0})
for record in records:
    family_hours[record["attack"]].append(record["hour_utc"])
    key = "attack" if record["binary_label"] == 1 else "benign"
    hour_totals[record["hour_utc"]][key] += record["rows"]

summary = {
    "source": SOURCE.relative_to(ROOT).as_posix(),
    "rows": row_count,
    "hourly_cell_count": len(records),
    "utc_hour_count": len(hour_totals),
    "min_flow_end_utc": datetime.fromtimestamp(min_end / 1000, timezone.utc).isoformat() if min_end is not None else None,
    "max_flow_end_utc": datetime.fromtimestamp(max_end / 1000, timezone.utc).isoformat() if max_end is not None else None,
    "family_first_last_utc_hour": {
        family: {"first_hour": min(hours), "last_hour": max(hours)}
        for family, hours in sorted(family_hours.items())
    },
    "hourly_totals": dict(sorted(hour_totals.items())),
    "hourly_counts_csv": table_path.relative_to(ROOT).as_posix(),
    "interpretation": "Describes scheduled testbed campaign composition by flow completion hour; this is not evidence of natural production concept drift.",
}
json_path = OUT / "nf_ton_iot_v3_hourly_label_audit.json"
json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "hourly_totals"}, indent=2))
