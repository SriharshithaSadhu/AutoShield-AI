"""Streaming schema and chronology audit for the scoped CSE-CIC-IDS2018 files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "data" / "cse_cic_ids2018" / "raw"
OUTPUT = DATA_DIR.parent / "audit.json"


def audit_file(path: Path, chunksize: int = 200_000) -> dict[str, object]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing_required = [name for name in ("Timestamp", "Label") if name not in columns]
    if missing_required:
        raise ValueError(f"{path.name} missing required columns: {missing_required}")

    labels: dict[str, int] = {}
    row_count = 0
    bad_timestamps = 0
    out_of_capture_date = 0
    backwards = 0
    minimum = None
    maximum = None
    previous_timestamp = None
    date_token = path.name.split("_", 1)[0].split("-", 1)[1]
    expected_date = datetime.strptime(date_token, "%d-%m-%Y").date()
    for frame in pd.read_csv(path, usecols=["Timestamp", "Label"], chunksize=chunksize):
        timestamps = pd.to_datetime(frame["Timestamp"].astype(str).str.strip(), dayfirst=True, errors="coerce")
        bad_timestamps += int(timestamps.isna().sum())
        out_of_capture_date += int((timestamps.notna() & (timestamps.dt.date != expected_date)).sum())
        valid = timestamps.loc[timestamps.notna() & timestamps.dt.date.eq(expected_date)]
        if len(valid):
            minimum = min(minimum, valid.min()) if minimum is not None else valid.min()
            maximum = max(maximum, valid.max()) if maximum is not None else valid.max()
            backwards += int((valid.diff().dropna() < pd.Timedelta(0)).sum())
            if previous_timestamp is not None and valid.iloc[0] < previous_timestamp:
                backwards += 1
            previous_timestamp = valid.iloc[-1]
        for label, count in frame["Label"].astype(str).str.strip().value_counts(dropna=False).items():
            labels[label] = labels.get(label, 0) + int(count)
        row_count += len(frame)
    return {
        "file": path.name,
        "rows": row_count,
        "columns": columns,
        "timestamp_min": str(minimum),
        "timestamp_max": str(maximum),
        "invalid_timestamp_rows": bad_timestamps,
        "timestamp_rows_outside_capture_date": out_of_capture_date,
        "backward_timestamp_steps_in_source_order": backwards,
        "label_counts": labels,
    }


def main() -> None:
    files = sorted(DATA_DIR.glob("*_TrafficForML_CICFlowMeter.csv"))
    if not files:
        raise FileNotFoundError(f"No downloaded flow CSVs found under {DATA_DIR}")
    audits = [audit_file(path) for path in files]
    full_schemas = {tuple(item["columns"]) for item in audits}
    # The 20-February export includes four leading flow/IP/port identifiers
    # absent from the other daily exports. Compare the common traffic schema
    # after excluding precisely those identifier-only columns.
    optional_identifiers = {"Flow ID", "Src IP", "Dst IP", "Src Port"}
    model_schemas = {
        tuple(column for column in item["columns"] if column not in optional_identifiers)
        for item in audits
    }
    base_schema = audits[0]["columns"]
    schema_variations = [
        {"file": item["file"], "extra_columns": sorted(set(item["columns"]) - set(base_schema)),
         "missing_columns": sorted(set(base_schema) - set(item["columns"]))}
        for item in audits if tuple(item["columns"]) != tuple(base_schema)
    ]
    report = {
        "files": audits,
        "all_full_schemas_identical": len(full_schemas) == 1,
        "all_non_identifier_schemas_identical": len(model_schemas) == 1,
        "schema_variations": schema_variations,
        "optional_identifier_columns_excluded_for_comparison": sorted(optional_identifiers),
        "total_rows": sum(int(item["rows"]) for item in audits),
        "ordering_note": "Source row order is audited only; replay must sort by parsed Timestamp.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Audited {len(files)} files / {report['total_rows']:,} rows; "
        f"full schemas identical={report['all_full_schemas_identical']}; "
        f"non-identifier schemas identical={report['all_non_identifier_schemas_identical']}"
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
