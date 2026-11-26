"""Audit the official WUSTL-IIoT-2021 CSV before any temporal replay.

StartTime/LastTime are used only for the audit and event ordering. No model
features or predictions are produced by this script.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "wustl_iiot"
ZIP = DATA / "wustl_iiot_2021.zip"
CSV = DATA / "wustl_iiot_2021.csv"
OUT = DATA / "temporal_audit.json"
CHUNK = 200_000
ORDER = "LastTime"  # a flow is available once it has ended


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    with zipfile.ZipFile(ZIP) as zf:
        zip_bad_member = zf.testzip()
        members = [{"name": x.filename, "size": x.file_size,
                    "compressed_size": x.compress_size}
                   for x in zf.infolist()]

    rows = 0
    nulls = {"StartTime": 0, "LastTime": 0, "Target": 0, "Traffic": 0}
    target = {}
    traffic = {}
    confusion = {}
    first_order = []
    last_order = None
    order_violations = 0
    min_start = min_end = None
    max_start = max_end = None
    reverse_intervals = 0
    exact_duplicate_feature_rows = 0
    feature_cols = None
    hourly: dict[str, dict[str, int]] = {}

    for frame in pd.read_csv(CSV, chunksize=CHUNK, low_memory=False):
        rows += len(frame)
        for col in nulls:
            nulls[col] += int(frame[col].isna().sum())
        for k, v in frame["Target"].value_counts(dropna=False).items():
            target[str(k)] = target.get(str(k), 0) + int(v)
        for k, v in frame["Traffic"].value_counts(dropna=False).items():
            traffic[str(k)] = traffic.get(str(k), 0) + int(v)
        cross = pd.crosstab(frame["Traffic"].astype(str), frame["Target"].astype(str))
        for cls, row in cross.iterrows():
            confusion[str(cls)] = {str(k): int(v) for k, v in row.items()}

        start = pd.to_datetime(frame["StartTime"], errors="coerce")
        end = pd.to_datetime(frame["LastTime"], errors="coerce")
        for series, is_start in ((start, True), (end, False)):
            lo, hi = series.min(), series.max()
            if pd.notna(lo):
                key = "min_start" if is_start else "min_end"
                old = min_start if is_start else min_end
                if old is None or lo < old:
                    if is_start: min_start = lo
                    else: min_end = lo
            if pd.notna(hi):
                if is_start:
                    max_start = hi if max_start is None else max(max_start, hi)
                else:
                    max_end = hi if max_end is None else max(max_end, hi)
        reverse_intervals += int((end < start).sum())
        ordered = end
        vals = ordered.dropna()
        if len(vals):
            if last_order is not None:
                order_violations += int((vals.iloc[0] < last_order))
            order_violations += int((vals.diff().dropna() < pd.Timedelta(0)).sum())
            last_order = vals.iloc[-1]
        if len(first_order) < 10:
            first_order.extend(ordered.head(10 - len(first_order)).astype(str).tolist())

        hourly_frame = frame.assign(_hour=end.dt.floor("h").astype(str))
        tab = pd.crosstab(hourly_frame["_hour"], hourly_frame["Traffic"].astype(str))
        for hour, row in tab.iterrows():
            bucket = hourly.setdefault(str(hour), {})
            for k, v in row.items():
                bucket[str(k)] = bucket.get(str(k), 0) + int(v)

        # Exact duplicate count across non-temporal/non-identity model fields.
        excluded = {"StartTime", "LastTime", "SrcAddr", "DstAddr", "sIpId", "dIpId", "Traffic", "Target"}
        cols = [x for x in frame.columns if x not in excluded]
        feature_cols = cols
        exact_duplicate_feature_rows += int(frame.duplicated(cols).sum())

    report = {
        "source": "WUSTL-IIoT-2021 official download",
        "official_url": "https://research.engineering.wustl.edu/~jain/iiot2/ftp/wustl_iiot_2021.zip",
        "published_archive_size_bytes": 106192911,
        "downloaded_archive_size_bytes": ZIP.stat().st_size,
        "downloaded_archive_sha256": sha256(ZIP),
        "archive_members": members,
        "archive_integrity_bad_member": zip_bad_member,
        "csv_size_bytes": CSV.stat().st_size,
        "rows": rows,
        "null_counts": nulls,
        "start_time_min": str(min_start),
        "start_time_max": str(max_start),
        "last_time_min": str(min_end),
        "last_time_max": str(max_end),
        "start_to_last_reversed_intervals": reverse_intervals,
        "input_row_order_violations_by_chunk_boundary_or_within_chunk_for_lasttime": order_violations,
        "first_10_lasttime_values_in_file_order": first_order,
        "target_counts": target,
        "traffic_counts": traffic,
        "traffic_target_crosstab": confusion,
        "hourly_traffic_counts_by_lasttime": hourly,
        "model_feature_columns_after_official_identity_exclusions": feature_cols,
        "duplicate_model_feature_rows_within_csv_chunks": exact_duplicate_feature_rows,
        "duplicate_count_note": "Within-chunk only; no claim of global duplicate uniqueness.",
        "row_ordering_for_replay": "stable chronological sort by LastTime; file order is not chronological",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "downloaded_archive_size_bytes", "downloaded_archive_sha256", "rows",
        "start_time_min", "start_time_max", "last_time_min", "last_time_max",
        "start_to_last_reversed_intervals", "input_row_order_violations_by_chunk_boundary_or_within_chunk_for_lasttime",
        "target_counts", "traffic_counts", "hourly_traffic_counts_by_lasttime",
        "duplicate_model_feature_rows_within_csv_chunks")}, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
