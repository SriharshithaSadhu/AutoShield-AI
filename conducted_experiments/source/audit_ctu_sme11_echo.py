"""Audit CTU-SME-11 Echo flow data, chronology, label support, and leakage risks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ctu_sme11"
FLOW_DIR = DATA / "extracted"
OUT = DATA / "echo_temporal_audit.json"
EXPECTED_ARCHIVE_MD5 = "311cbd100eeddf76c8056919fe35b0cb"
DATES = [f"2023-02-{d:02d}" for d in range(20, 27)]
EXCLUDED_FROM_MODEL = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "label", "detailedlabel", "tunnel_parents",
]
CANDIDATE_NUMERIC = [
    "duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts",
    "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
]
CANDIDATE_CATEGORICAL = [
    "proto", "service", "conn_state", "local_orig", "local_resp", "history",
]


def file_hash(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_zeek(path: Path) -> pd.DataFrame:
    fields = None
    with path.open("r", encoding="utf-8", errors="strict") as f:
        for line in f:
            if line.startswith("#fields\t"):
                fields = line.rstrip("\r\n").split("\t")[1:]
                break
    if not fields:
        raise ValueError(f"No #fields schema found in {path}")
    return pd.read_csv(path, sep="\t", comment="#", names=fields,
                       na_values=["-", "(empty)"], keep_default_na=True,
                       low_memory=False)


def main() -> None:
    archive = DATA / "CTU-SME-11_Honeypot-Assistant-Amazon-Echo1stGen-1_v1.0.0.tar.bz2"
    archive_md5 = file_hash(archive, "md5")
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError(f"Archive MD5 mismatch: {archive_md5}")

    day_reports = []
    all_times = []
    label_duration = {}
    all_uid = set()
    uid_duplicates_across_days = 0
    aggregate_labels: dict[str, int] = {}
    aggregate_detailed: dict[str, int] = {}
    aggregate_features = {c: {"missing": 0, "rows": 0} for c in CANDIDATE_NUMERIC + CANDIDATE_CATEGORICAL}
    all_columns: list[str] | None = None
    global_min = global_max = None

    for day in DATES:
        path = FLOW_DIR / day / "zeek" / "conn.log.labeled"
        frame = read_zeek(path)
        if all_columns is None:
            all_columns = list(frame.columns)
        elif list(frame.columns) != all_columns:
            raise ValueError(f"Schema mismatch in {path.name}")
        if "ts" not in frame or "label" not in frame:
            raise ValueError(f"Required Zeek fields absent from {path}")
        ts = pd.to_numeric(frame["ts"], errors="coerce")
        dt = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
        labels = frame["label"].fillna("<missing>").astype(str).value_counts().to_dict()
        detailed = frame["detailedlabel"].fillna("<missing>").astype(str).value_counts().to_dict()
        for label, count in labels.items():
            aggregate_labels[label] = aggregate_labels.get(label, 0) + int(count)
        for label, group in frame.groupby("label", dropna=False):
            key = "<missing>" if pd.isna(label) else str(label)
            entry = label_duration.setdefault(key, {"rows": 0, "duration_missing": 0})
            entry["rows"] += int(len(group))
            entry["duration_missing"] += int(group["duration"].isna().sum())
        for label, count in detailed.items():
            aggregate_detailed[label] = aggregate_detailed.get(label, 0) + int(count)

        observed = dt.dropna()
        day_min = observed.min() if len(observed) else None
        day_max = observed.max() if len(observed) else None
        if day_min is not None:
            global_min = day_min if global_min is None else min(global_min, day_min)
            global_max = day_max if global_max is None else max(global_max, day_max)
        monotonic_violations = int((observed.diff().dropna() < pd.Timedelta(0)).sum())
        all_times.append(observed)

        dup_uid = int(frame["uid"].duplicated().sum()) if "uid" in frame else None
        if "uid" in frame:
            for uid in frame["uid"].dropna().astype(str):
                if uid in all_uid:
                    uid_duplicates_across_days += 1
                all_uid.add(uid)

        feature_missing = {}
        for column in CANDIDATE_NUMERIC + CANDIDATE_CATEGORICAL:
            if column not in frame:
                raise ValueError(f"Missing candidate feature {column} in {path}")
            aggregate_features[column]["missing"] += int(frame[column].isna().sum())
            aggregate_features[column]["rows"] += len(frame)
            feature_missing[column] = int(frame[column].isna().sum())

        day_reports.append({
            "capture_day": day,
            "rows": len(frame),
            "timestamp_missing": int(dt.isna().sum()),
            "timestamp_min_utc": None if day_min is None else day_min.isoformat(),
            "timestamp_max_utc": None if day_max is None else day_max.isoformat(),
            "timestamp_monotonic_violations_in_file_order": monotonic_violations,
            "labels": {str(k): int(v) for k, v in labels.items()},
            "detailed_label_cardinality": len(detailed),
            "detailed_label_counts": {str(k): int(v) for k, v in detailed.items()},
            "duplicate_uid_within_day": dup_uid,
            "candidate_feature_missing": feature_missing,
        })

    combined = pd.concat(all_times, ignore_index=True)
    # Files are individually sorted, but chronology must still be established
    # globally from ts; this measures whether concatenating by capture-day
    # filename alone yields a monotonic stream.
    cross_file_order_violations = int((combined.diff().dropna() < pd.Timedelta(0)).sum())
    # Also record how many adjacent rows share the same timestamp once the full
    # stream is stably ordered; chunking must not split equal event times.
    equal_timestamp_adjacencies = int((combined.sort_values(kind="stable").diff()
                                       == pd.Timedelta(0)).sum())
    reports_rows = sum(x["rows"] for x in day_reports)
    feature_missing_totals = {
        c: {"missing": v["missing"], "rows": v["rows"],
            "missing_fraction": v["missing"] / max(1, v["rows"])}
        for c, v in aggregate_features.items()
    }
    for entry in label_duration.values():
        entry["duration_missing_fraction"] = entry["duration_missing"] / max(1, entry["rows"])
    report = {
        "source_doi": "10.5281/zenodo.7958259",
        "archive_md5_verified": archive_md5,
        "archive_size_bytes": archive.stat().st_size,
        "capture_days": DATES,
        "rows": reports_rows,
        "timestamp_min_utc": None if global_min is None else global_min.isoformat(),
        "timestamp_max_utc": None if global_max is None else global_max.isoformat(),
        "labels_total": aggregate_labels,
        "detailed_label_cardinality": len(aggregate_detailed),
        "detailed_label_counts": aggregate_detailed,
        "fields": all_columns,
        "model_input_candidates": {
            "numeric": CANDIDATE_NUMERIC,
            "categorical": CANDIDATE_CATEGORICAL,
            "excluded": EXCLUDED_FROM_MODEL,
        },
        "candidate_feature_missing_totals": feature_missing_totals,
        "duration_missing_by_label": label_duration,
        "flow_start_ts_plus_duration_rows": int(sum(v["rows"] - v["duration_missing"] for v in label_duration.values())),
        "duplicate_uid_across_capture_days": uid_duplicates_across_days,
        "capture_day_concatenation_timestamp_order_violations": cross_file_order_violations,
        "equal_timestamp_adjacencies_after_global_sort": equal_timestamp_adjacencies,
        "global_timestamp_sort_required": cross_file_order_violations > 0,
        "per_day": day_reports,
        "label_policy": "Benign=0, Malicious=1; Background retained as a separate audit/unknown category, never silently mapped.",
        "notes": [
            "Zeek #open metadata are not flow timestamps; use the per-row ts field.",
            "Zeek Conn::Info defines ts as first-packet time and duration as an optional connection interval; final flow counters are not available at ts.",
            "The per-day labels.config files include IP-based labeling rules; IP and port identifiers are excluded from candidate model inputs to reduce label/identity leakage.",
            "No per-flow availability time is imputed where duration is missing. A ts-sorted replay of end-of-flow counters would be temporally leaky.",
            "The feature list is provisional until the data audit and leakage review are complete; this script does not train a model.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "rows", "timestamp_min_utc", "timestamp_max_utc", "labels_total",
        "detailed_label_cardinality", "candidate_feature_missing_totals",
        "duration_missing_by_label", "flow_start_ts_plus_duration_rows",
        "duplicate_uid_across_capture_days", "capture_day_concatenation_timestamp_order_violations",
        "equal_timestamp_adjacencies_after_global_sort")}, indent=2))
    print("Per day:")
    for row in day_reports:
        print(row["capture_day"], row["rows"], row["timestamp_min_utc"], row["timestamp_max_utc"], row["labels"])
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
