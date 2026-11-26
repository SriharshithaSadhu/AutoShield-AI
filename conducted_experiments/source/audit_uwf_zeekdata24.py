"""Integrity, chronology, and label-support gate for UWF-ZeekData24 Parquet."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "uwf_zeekdata24"
MANIFEST = DATA / "download_manifest.json"
OUTPUT = DATA / "audit.json"
FETCH_ROWS = 250_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    con = duckdb.connect()
    con.execute("SET threads=4")
    files = []
    shared_schema = None
    overall_labels = {}
    daily_labels = {}
    label_tactic_counts = {}
    total_rows = total_start_regressions = total_end_regressions = 0
    total_invalid = total_negative_duration = total_duration_null = 0
    datetime_ts_absdiff_max = 0.0
    for item in manifest["files"]:
        path = DATA / "parquet" / item["filename"]
        if not path.exists() or path.stat().st_size != item["actual_bytes"]:
            raise RuntimeError(f"Missing/size mismatch: {path}")
        digest = sha256(path)
        if digest != item["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {path}")
        schema_rows = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
        schema = [{"name": r[0], "type": r[1], "nullable": r[2]} for r in schema_rows]
        names = [r["name"] for r in schema]
        if shared_schema is None:
            shared_schema = schema
        elif names != [c["name"] for c in shared_schema]:
            raise RuntimeError(f"Column schema mismatch in {path.name}")

        summary = con.execute("""
            SELECT count(*) AS rows,
                   min(datetime) AS datetime_min, max(datetime) AS datetime_max,
                   min(ts) AS ts_min, max(ts) AS ts_max,
                   min(duration) AS duration_min, max(duration) AS duration_max,
                   count(*) FILTER (WHERE datetime IS NULL OR ts IS NULL) AS invalid_time,
                   count(*) FILTER (WHERE duration IS NULL) AS null_duration,
                   count(*) FILTER (WHERE duration < 0) AS negative_duration,
                   max(abs(epoch(datetime)-ts)) FILTER (WHERE datetime IS NOT NULL AND ts IS NOT NULL) AS datetime_ts_absdiff_max
            FROM read_parquet(?)
        """, [str(path)]).fetchone()
        rows = int(summary[0])
        labels = con.execute("""
            SELECT coalesce(cast(label_binary AS varchar), '<NULL>') AS label, count(*)
            FROM read_parquet(?) GROUP BY 1 ORDER BY 1
        """, [str(path)]).fetchall()
        tactics = con.execute("""
            SELECT coalesce(label_tactic, '<NULL>') AS label, count(*)
            FROM read_parquet(?) GROUP BY 1 ORDER BY 1
        """, [str(path)]).fetchall()
        daily = con.execute("""
            SELECT cast(datetime AS date)::varchar AS day,
                   coalesce(cast(label_binary AS varchar), '<NULL>') AS label,
                   count(*) AS n
            FROM read_parquet(?) GROUP BY 1,2 ORDER BY 1,2
        """, [str(path)]).fetchall()
        inconsistencies = con.execute("""
            SELECT count(*) FILTER (WHERE lower(cast(label_binary AS varchar))='true'
                                     AND lower(coalesce(label_tactic,''))='none') AS positive_none,
                   count(*) FILTER (WHERE lower(cast(label_binary AS varchar))='false'
                                     AND lower(coalesce(label_tactic,''))<>'none') AS negative_non_none,
                   count(*) FILTER (WHERE lower(cast(label_binary AS varchar)) NOT IN ('true','false')) AS unexpected_binary
            FROM read_parquet(?)
        """, [str(path)]).fetchone()

        # Preserve source ordering and count start/completion-order descents.
        start_regressions = end_regressions = 0
        prev_start = prev_end = None
        cursor = con.execute("SELECT ts, duration FROM read_parquet(?)", [str(path)])
        while batch := cursor.fetchmany(FETCH_ROWS):
            values = np.asarray(batch, dtype=np.float64)
            starts = values[:, 0]
            durations = values[:, 1]
            valid = np.isfinite(starts)
            starts = starts[valid]
            durations = durations[valid]
            ends = starts + np.where(np.isfinite(durations) & (durations >= 0), durations, 0.0)
            if len(starts):
                start_regressions += int(np.count_nonzero(np.diff(starts) < 0))
                end_regressions += int(np.count_nonzero(np.diff(ends) < 0))
                if prev_start is not None and starts[0] < prev_start:
                    start_regressions += 1
                if prev_end is not None and ends[0] < prev_end:
                    end_regressions += 1
                prev_start, prev_end = float(starts[-1]), float(ends[-1])

        for label, n in labels:
            overall_labels[label] = overall_labels.get(label, 0) + int(n)
        for day, label, n in daily:
            daily_labels.setdefault(day, {})[label] = int(n)
        for label, n in tactics:
            label_tactic_counts[label] = label_tactic_counts.get(label, 0) + int(n)
        dt_diff = float(summary[10] or 0.0)
        datetime_ts_absdiff_max = max(datetime_ts_absdiff_max, dt_diff)
        record = {
            "interval": item["interval"], "filename": item["filename"],
            "rows": rows, "sha256_verified": digest,
            "datetime_min": str(summary[1]), "datetime_max": str(summary[2]),
            "ts_min_epoch": summary[3], "ts_max_epoch": summary[4],
            "duration_min_seconds": summary[5], "duration_max_seconds": summary[6],
            "invalid_time_rows": int(summary[7]), "null_duration_rows": int(summary[8]),
            "negative_duration_rows": int(summary[9]),
            "max_abs_datetime_minus_ts_seconds": dt_diff,
            "binary_label_counts": {str(k): int(v) for k, v in labels},
            "tactic_label_counts": {str(k): int(v) for k, v in tactics},
            "binary_tactic_inconsistencies": {
                "positive_but_tactic_none": int(inconsistencies[0]),
                "negative_but_tactic_non_none": int(inconsistencies[1]),
                "unexpected_binary_value": int(inconsistencies[2]),
            },
            "source_order_start_time_descents": start_regressions,
            "source_order_completion_time_descents": end_regressions,
            "schema": schema,
        }
        files.append(record)
        total_rows += rows
        total_start_regressions += start_regressions
        total_end_regressions += end_regressions
        total_invalid += record["invalid_time_rows"]
        total_duration_null += record["null_duration_rows"]
        total_negative_duration += record["negative_duration_rows"]
        print(f"{item['interval']}: rows={rows:,}; start_desc={start_regressions:,}; "
              f"end_desc={end_regressions:,}; labels={record['binary_label_counts']}", flush=True)

    result = {
        "dataset": "UWF-ZeekData24",
        "source": manifest["source"],
        "license": manifest["license"],
        "download_manifest_sha256": manifest["manifest_sha256"],
        "total_rows": total_rows,
        "total_start_order_descents": total_start_regressions,
        "total_completion_order_descents": total_end_regressions,
        "invalid_time_rows": total_invalid,
        "null_duration_rows": total_duration_null,
        "negative_duration_rows": total_negative_duration,
        "max_abs_datetime_minus_unix_ts_seconds": datetime_ts_absdiff_max,
        "binary_label_counts": overall_labels,
        "tactic_label_counts": label_tactic_counts,
        "calendar_day_label_counts": daily_labels,
        "candidate_predictor_columns_after_exclusions": [
            "conn_state", "duration", "history", "local_orig", "local_resp",
            "missed_bytes", "orig_bytes", "orig_ip_bytes", "orig_pkts", "proto",
            "resp_bytes", "resp_ip_bytes", "resp_pkts", "service",
        ],
        "excluded_columns_and_reasons": {
            "src_ip_zeek": "endpoint identity leakage / memorization",
            "dest_ip_zeek": "endpoint identity leakage / memorization",
            "src_port_zeek": "potential campaign/machine identity proxy; retain only under explicit ablation",
            "dest_port_zeek": "attack destination-service proxy; evaluate separately, do not include silently",
            "community_id": "flow tuple-derived identifier",
            "uid": "per-connection identifier",
            "ts": "availability ordering only",
            "datetime": "availability ordering only",
            "label_binary": "target",
            "label_tactic": "target-derived label",
            "label_technique": "target-derived label",
            "label_cve": "target-derived label/proxy",
        },
        "causal_completion_time_definition": "datetime + duration seconds; source order is audited but full replay must sort by completion time before scoring final conn features",
        "files": files,
        "limitations": [
            "Dataset authors describe a controlled cyber range with automated attacks, although the captured packet/flow telemetry is real, not synthetically generated rows.",
            "Labels are joined from mission logs using time, addresses, ports, and a one-minute slop factor; per-flow label errors and temporal leakage need further audit.",
            "Zeek conn records contain final aggregate features; score at flow completion, not the first-packet timestamp.",
            "Zeek duration may not include trailing teardown packets, so datetime+duration can understate final-feature availability for some TCP flows.",
            "CSE-CIC-IDS2018 and NF-ToN-IoT-v3 have already been development-explored; UWF-ZeekData24 is a new external source but has close published UWF-specific intrusion studies.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"total rows={total_rows:,}, labels={overall_labels}, start descents={total_start_regressions:,}, "
          f"completion descents={total_end_regressions:,}", flush=True)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
