"""Audit chronology, label support, and flow-end observability for CTU RPi."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ctu_sme11"
FLOW_DIR = DATA / "raspberrypi_extracted"
OUT = DATA / "raspberrypi_temporal_audit.json"
ARCHIVE = DATA / "CTU-SME-11_Honeypot-Minicomputer-RaspberryPi-Gen3-20_v1.0.0.tar.bz2"
EXPECTED_MD5 = "94285412e422f9983b7431c32acaf5b9"
SUMMARY_EXPECTED = {"rows": 199266, "Benign": 45888, "Malicious": 153378, "Background": 0}


def file_md5(path: Path) -> str:
    h = hashlib.md5()
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
        raise ValueError(f"Missing #fields in {path}")
    return pd.read_csv(path, sep="\t", comment="#", names=fields,
                       na_values=["-", "(empty)"], keep_default_na=True,
                       low_memory=False)


def main() -> None:
    archive_md5 = file_md5(ARCHIVE)
    if archive_md5 != EXPECTED_MD5:
        raise ValueError(f"Archive MD5 mismatch: {archive_md5}")
    day_reports = []
    frames = []
    for path in sorted(FLOW_DIR.glob("*/zeek/conn.log.labeled")):
        frame = read_zeek(path)
        frame["capture_day"] = path.parts[-3]
        frames.append(frame)
        counts = frame["label"].fillna("<missing>").astype(str).value_counts().to_dict()
        ts = pd.to_numeric(frame["ts"], errors="coerce")
        timestamp = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
        duration = pd.to_numeric(frame["duration"], errors="coerce")
        end_ts = ts + duration
        end_time = pd.to_datetime(end_ts, unit="s", utc=True, errors="coerce")
        day_reports.append({
            "capture_day": path.parts[-3],
            "rows": len(frame),
            "labels": {str(k): int(v) for k, v in counts.items()},
            "ts_min_utc": None if timestamp.isna().all() else timestamp.min().isoformat(),
            "ts_max_utc": None if timestamp.isna().all() else timestamp.max().isoformat(),
            "ts_order_violations_in_file_order": int((timestamp.diff().dropna() < pd.Timedelta(0)).sum()),
            "known_end_time_order_violations_in_file_order": int((end_time.diff().dropna() < pd.Timedelta(0)).sum()),
            "duration_missing": int(duration.isna().sum()),
            "duration_negative": int((duration.dropna() < 0).sum()),
            "end_before_start": int((duration.dropna() < 0).sum()),
        })
    if len(day_reports) != 7:
        raise ValueError(f"Expected 7 daily flow files, found {len(day_reports)}")
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["ts_num"] = pd.to_numeric(all_rows["ts"], errors="coerce")
    all_rows["duration_num"] = pd.to_numeric(all_rows["duration"], errors="coerce")
    all_rows["end_ts"] = all_rows["ts_num"] + all_rows["duration_num"]
    labels = all_rows["label"].fillna("<missing>").astype(str).value_counts().to_dict()
    counts = {str(k): int(v) for k, v in labels.items()}
    duration_missing = all_rows["duration_num"].isna()
    missing_by_label = {}
    state_by_label = {}
    for label, group in all_rows.groupby("label", dropna=False):
        key = "<missing>" if pd.isna(label) else str(label)
        missing = int(group["duration_num"].isna().sum())
        missing_by_label[key] = {"missing": missing, "rows": int(len(group)),
                                 "fraction": missing / max(1, len(group))}
        state_by_label[key] = {
            str(state): {"rows": int(len(state_group)),
                         "duration_missing": int(state_group["duration_num"].isna().sum())}
            for state, state_group in group.groupby("conn_state", dropna=False)
        }
    complete = all_rows.loc[~duration_missing].copy()
    complete["end_time"] = pd.to_datetime(complete["end_ts"], unit="s", utc=True, errors="coerce")
    end_order_file = complete["end_time"]
    end_order = end_order_file.sort_values(kind="stable")
    model_columns = [c for c in all_rows.columns if c not in {
        "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
        "label", "detailedlabel", "tunnel_parents", "capture_day", "ts_num",
        "duration_num", "end_ts",
    }]
    report = {
        "source_doi": "10.5281/zenodo.7958259",
        "archive_md5_verified": archive_md5,
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "official_summary_expected": SUMMARY_EXPECTED,
        "capture_days": [r["capture_day"] for r in day_reports],
        "rows": len(all_rows),
        "labels_total": counts,
        "summary_counts_match": len(all_rows) == SUMMARY_EXPECTED["rows"] and all(
            counts.get(k, 0) == v for k, v in SUMMARY_EXPECTED.items() if k != "rows"),
        "duration_missing_total": int(duration_missing.sum()),
        "duration_missing_by_label": missing_by_label,
        "conn_state_by_label_and_duration_observability": state_by_label,
        "complete_duration_rows": len(complete),
        "complete_duration_rows_by_label": {
            str(k): int((~g["duration_num"].isna()).sum())
            for k, g in all_rows.groupby("label", dropna=False)
        },
        "complete_duration_end_time_range_utc": [
            None if end_order.isna().all() else end_order.min().isoformat(),
            None if end_order.isna().all() else end_order.max().isoformat(),
        ],
        "complete_duration_end_order_violations_in_archive_concatenation": int((end_order_file.diff().dropna() < pd.Timedelta(0)).sum()),
        "complete_duration_end_equal_timestamp_adjacencies": int((end_order.sort_values(kind="stable").diff() == pd.Timedelta(0)).sum()),
        "candidate_features_excluding_identifiers_and_labels": model_columns,
        "per_day": day_reports,
        "timing_policy": "ts is first packet. ts+duration is only a candidate completed-flow availability key when duration is present; no availability time is imputed for missing duration.",
        "protocol_gate": "Do not use final flow counters as per-flow predictions at ts. Evaluate either day-end batches with explicit one-day label delay, or a biased complete-duration sensitivity. Packet-level online replay requires separate PCAP reprocessing and frozen Zeek configuration.",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["rows", "labels_total", "summary_counts_match",
          "duration_missing_total", "duration_missing_by_label", "complete_duration_rows",
          "complete_duration_rows_by_label", "per_day"]}, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
