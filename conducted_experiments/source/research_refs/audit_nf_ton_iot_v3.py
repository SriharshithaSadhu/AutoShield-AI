"""Audit official NF-ToN-IoT-v3 time, labels, and stream ordering in chunks."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "nf_ton_iot_v3"
SOURCE = ROOT / "NF-ToN-IoT-v3.csv"
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
CHUNKSIZE = 1_000_000
COLUMNS = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "FLOW_DURATION_MILLISECONDS", "Label", "Attack"]

daily: dict[tuple[str, int, str], int] = defaultdict(int)
attack_counts: Counter[str] = Counter()
label_counts: Counter[str] = Counter()
rows = missing_start = missing_end = missing_label = invalid_label = 0
inversions = negative_durations = duration_mismatch_over_1ms = 0
previous_start: int | None = None
min_start: int | None = None
max_start: int | None = None
max_end: int | None = None
min_start_utc = max_start_utc = None
first_utc = last_utc = None
prev_record_utc = None

for frame in pd.read_csv(SOURCE, usecols=COLUMNS, chunksize=CHUNKSIZE, low_memory=False):
    starts = pd.to_numeric(frame["FLOW_START_MILLISECONDS"], errors="coerce")
    ends = pd.to_numeric(frame["FLOW_END_MILLISECONDS"], errors="coerce")
    durations = pd.to_numeric(frame["FLOW_DURATION_MILLISECONDS"], errors="coerce")
    labels = pd.to_numeric(frame["Label"], errors="coerce")
    missing_start += int(starts.isna().sum())
    missing_end += int(ends.isna().sum())
    missing_label += int(labels.isna().sum())
    invalid_label += int((labels.notna() & ~labels.isin([0, 1])).sum())
    valid_starts = starts.dropna().astype("int64")
    valid_ends = ends.dropna().astype("int64")
    if not valid_starts.empty:
        chunk_min, chunk_max = int(valid_starts.min()), int(valid_starts.max())
        min_start = chunk_min if min_start is None else min(min_start, chunk_min)
        max_start = chunk_max if max_start is None else max(max_start, chunk_max)
        min_start_utc = min_start if min_start_utc is None else min(min_start_utc, chunk_min)
        max_start_utc = max_start if max_start_utc is None else max(max_start_utc, chunk_max)
        if previous_start is not None and int(valid_starts.iloc[0]) < previous_start:
            inversions += 1
        inversions += int((valid_starts.diff().dropna() < 0).sum())
        previous_start = int(valid_starts.iloc[-1])
        dates = pd.to_datetime(valid_starts, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        valid_frame = frame.loc[valid_starts.index, ["Attack"]].copy()
        valid_frame["date_utc"] = dates.to_numpy()
        valid_frame["binary_label"] = labels.loc[valid_starts.index].fillna(-1).astype("int8").to_numpy()
        valid_frame["attack"] = valid_frame["Attack"].fillna("<MISSING>").astype(str).str.strip()
        grouped = valid_frame.groupby(["date_utc", "binary_label", "attack"], sort=False).size()
        for key, count in grouped.items():
            daily[(key[0], int(key[1]), key[2])] += int(count)
    if not valid_ends.empty:
        value = int(valid_ends.max())
        max_end = value if max_end is None else max(max_end, value)
    both = starts.notna() & ends.notna()
    negative_durations += int(((ends[both] - starts[both]) < 0).sum())
    compare = both & durations.notna()
    duration_mismatch_over_1ms += int(((ends[compare] - starts[compare] - durations[compare]).abs() > 1).sum())
    attack_grouped = frame["Attack"].fillna("<MISSING>").astype(str).str.strip().value_counts()
    attack_counts.update(attack_grouped.to_dict())
    label_grouped = labels.value_counts(dropna=False)
    label_counts.update({str(int(key)) if pd.notna(key) and key in (0, 1) else "<INVALID>": int(value) for key, value in label_grouped.items()})
    rows += len(frame)

def utc_ms(value: int | None) -> str | None:
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat() if value is not None else None

daily_rows = [
    {"date_utc": date, "binary_label": label, "attack": attack, "rows": count}
    for (date, label, attack), count in sorted(daily.items())
]
pd.DataFrame(daily_rows).to_csv(OUT / "nf_ton_iot_v3_daily_label_counts.csv", index=False)
summary = {
    "source": str(SOURCE.relative_to(ROOT)),
    "rows": rows,
    "expected_rows_uq_page": 27_520_260,
    "row_count_matches_publisher": rows == 27_520_260,
    "missing_start": missing_start,
    "missing_end": missing_end,
    "missing_binary_label": missing_label,
    "invalid_binary_label": invalid_label,
    "binary_label_counts": dict(label_counts),
    "attack_label_counts": dict(attack_counts),
    "min_start_utc": utc_ms(min_start),
    "max_start_utc": utc_ms(max_start),
    "max_end_utc": utc_ms(max_end),
    "start_time_inversions_in_file_order": inversions,
    "negative_end_minus_start_rows": negative_durations,
    "duration_feature_differs_from_end_minus_start_by_over_1ms": duration_mismatch_over_1ms,
    "daily_counts_csv": "results/nf_ton_iot_v3_daily_label_counts.csv",
    "interpretation": "Publisher-grounded time and label audit only; this does not prove independent packet-to-flow attribution or temporal IDS performance.",
}
path = OUT / "nf_ton_iot_v3_temporal_data_audit.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
