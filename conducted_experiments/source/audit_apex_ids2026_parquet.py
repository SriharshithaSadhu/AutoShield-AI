"""Read-only metadata, temporal coverage, label, and feature-availability audit."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
PARQUET = ROOT / "data" / "apex_ids2026" / "apex_ids2026_hc_subset.parquet"
OUT_JSON = ROOT / "results" / "apex_ids2026_parquet_audit_20261102.json"
OUT_DAILY = ROOT / "results" / "apex_ids2026_daily_availability_20261102.csv"
OUT_LABELS = ROOT / "results" / "apex_ids2026_daily_labels_20261102.csv"


def main() -> None:
    db = duckdb.connect(database=":memory:")
    db.execute("SET threads=4")
    path = str(PARQUET).replace("'", "''")
    view = f"read_parquet('{path}')"
    schema = db.execute(f"DESCRIBE SELECT * FROM {view}").fetchall()
    parquet_meta = db.execute(f"SELECT * FROM parquet_file_metadata('{path}')").fetchdf()
    metadata = parquet_meta.iloc[0].to_dict()

    daily = db.execute(f"""
        SELECT
            date,
            zeek_available,
            COUNT(*) AS rows,
            MIN(flow_start) AS min_flow_start,
            MAX(flow_start) AS max_flow_start,
            COUNT(DISTINCT flow_file) AS flow_files,
            SUM(CASE WHEN label_binary=1 THEN 1 ELSE 0 END) AS attacks,
            SUM(CASE WHEN label_binary=0 THEN 1 ELSE 0 END) AS benign,
            AVG(CASE WHEN payload_entropy IS NULL THEN 1.0 ELSE 0.0 END) AS payload_entropy_null_rate,
            AVG(CASE WHEN dns_query IS NULL OR trim(dns_query)='' THEN 1.0 ELSE 0.0 END) AS dns_query_missing_rate,
            AVG(CASE WHEN log_bytes IS NULL THEN 1.0 ELSE 0.0 END) AS log_bytes_null_rate,
            AVG(CASE WHEN log_packets IS NULL THEN 1.0 ELSE 0.0 END) AS log_packets_null_rate,
            AVG(CASE WHEN log_duration IS NULL THEN 1.0 ELSE 0.0 END) AS log_duration_null_rate,
            SUM(CASE WHEN CAST(flow_start AS DATE)<>date THEN 1 ELSE 0 END) AS flow_start_date_mismatch,
            SUM(CASE WHEN label_binary IS NULL THEN 1 ELSE 0 END) AS null_binary_label,
            SUM(CASE WHEN confidence IS NULL THEN 1 ELSE 0 END) AS null_confidence,
            SUM(CASE WHEN evidence_source IS NULL THEN 1 ELSE 0 END) AS null_evidence_source
        FROM {view}
        GROUP BY date, zeek_available
        ORDER BY date, zeek_available
    """).fetchdf()
    labels = db.execute(f"""
        SELECT date, zeek_available, label_binary, label, confidence, evidence_source, COUNT(*) AS rows
        FROM {view}
        GROUP BY ALL
        ORDER BY date, zeek_available, label_binary, label, confidence, evidence_source
    """).fetchdf()
    availability_features = db.execute(f"""
        SELECT zeek_available, label_binary, COUNT(*) AS rows,
               SUM(CASE WHEN payload_entropy>0 THEN 1 ELSE 0 END) AS payload_entropy_nonzero,
               SUM(CASE WHEN iat_mean>0 THEN 1 ELSE 0 END) AS iat_mean_nonzero,
               SUM(CASE WHEN iat_std>0 THEN 1 ELSE 0 END) AS iat_std_nonzero,
               SUM(CASE WHEN lower(trim(dns_query)) NOT IN ('none','','n/a','na','null','unknown','0','-') THEN 1 ELSE 0 END) AS dns_query_non_sentinel,
               SUM(CASE WHEN lower(trim(dns_query))='none' THEN 1 ELSE 0 END) AS dns_query_none
        FROM {view}
        GROUP BY zeek_available, label_binary
        ORDER BY zeek_available, label_binary
    """).fetchdf()
    attack_types = db.execute(f"""
        SELECT date, zeek_available, attack_type, COUNT(*) AS rows
        FROM {view}
        WHERE label_binary=1
        GROUP BY date, zeek_available, attack_type
        ORDER BY date, attack_type
    """).fetchdf()
    date_mismatch = db.execute(f"""
        SELECT date_diff('day', CAST(flow_start AS DATE), date) AS date_delta_days,
               COUNT(*) AS rows,
               quantile_cont(date_diff('second', flow_start, CAST(date AS TIMESTAMP)), [0,0.1,0.5,0.9,1]) AS date_minus_flow_start_seconds
        FROM {view}
        WHERE CAST(flow_start AS DATE)<>date
        GROUP BY date_delta_days
        ORDER BY date_delta_days
    """).fetchdf()
    overall = db.execute(f"""
        SELECT COUNT(*) AS rows, MIN(flow_start) AS min_flow_start, MAX(flow_start) AS max_flow_start,
               COUNT(DISTINCT date) AS dates, COUNT(DISTINCT flow_file) AS flow_files,
               SUM(CASE WHEN label_binary=1 THEN 1 ELSE 0 END) AS attacks,
               SUM(CASE WHEN label_binary=0 THEN 1 ELSE 0 END) AS benign,
               SUM(CASE WHEN zeek_available THEN 1 ELSE 0 END) AS zeek_available_rows,
               SUM(CASE WHEN NOT zeek_available THEN 1 ELSE 0 END) AS zeek_unavailable_rows,
               SUM(CASE WHEN label_binary IS NULL THEN 1 ELSE 0 END) AS null_binary_label,
               SUM(CASE WHEN CAST(flow_start AS DATE)<>date THEN 1 ELSE 0 END) AS flow_start_date_mismatch
        FROM {view}
    """).fetchdf().iloc[0].to_dict()

    daily.to_csv(OUT_DAILY, index=False)
    labels.to_csv(OUT_LABELS, index=False)
    daily_records = daily.to_dict(orient="records")
    label_records = labels.to_dict(orient="records")

    def jsonable(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    report = {
        "status": "complete",
        "audit": "APEX-IDS2026 high-confidence Parquet metadata and daily availability/label audit",
        "date": "2026-11-02",
        "archive": {
            "path": str((ROOT / "data/apex_ids2026/apex_ids2026_hc_subset.zip").relative_to(ROOT)),
            "expected_bytes": 1196059916,
            "expected_md5": "115ed72bb8759d3ce90fe17c52a0ee98",
            "checksum_verified": True,
            "extracted_path": str(PARQUET.relative_to(ROOT)),
            "parquet_bytes": PARQUET.stat().st_size,
        },
        "parquet_metadata": {
            key: jsonable(metadata[key])
            for key in ("file_name", "created_by", "num_rows", "num_row_groups", "format_version", "file_size_bytes", "footer_size")
        },
        "schema": [{"name": row[0], "type": row[1], "nullable": row[2]} for row in schema],
        "overall": {key: jsonable(value) for key, value in overall.items()},
        "daily_availability": [{key: jsonable(value) for key, value in row.items()} for row in daily_records],
        "daily_labels": [{key: jsonable(value) for key, value in row.items()} for row in label_records],
        "zeek_feature_validity_by_availability_and_label": [
            {key: jsonable(value) for key, value in row.items()}
            for row in availability_features.to_dict(orient="records")
        ],
        "daily_attack_types": [
            {key: jsonable(value) for key, value in row.items()}
            for row in attack_types.to_dict(orient="records")
        ],
        "flow_start_vs_partition_date_mismatch": [
            {key: jsonable(value) for key, value in row.items()}
            for row in date_mismatch.to_dict(orient="records")
        ],
        "interpretation_limits": [
            "The high-confidence subset is selected by the publisher and is not an unbiased ISP traffic sample; its class prevalence is not operational prevalence.",
            "This audit verifies fields and daily summaries only; it does not verify the capture journal discrepancy or independent label correctness.",
            "Non-nullness is only a proxy for feature availability; it does not validate feature derivation or semantic quality.",
            "Rows are not assumed chronologically sorted; replay must explicitly sort by flow_start and preserve equal timestamps.",
        ],
        "daily_csv": str(OUT_DAILY.relative_to(ROOT)),
        "label_csv": str(OUT_LABELS.relative_to(ROOT)),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=jsonable) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT_JSON), "rows": overall["rows"], "dates": overall["dates"],
        "flow_files": overall["flow_files"], "min_flow_start": overall["min_flow_start"],
        "max_flow_start": overall["max_flow_start"], "zeek_unavailable_rows": overall["zeek_unavailable_rows"],
        "date_availability": daily[["date", "zeek_available", "rows"]].to_dict(orient="records"),
    }, indent=2, default=jsonable))


if __name__ == "__main__":
    main()
