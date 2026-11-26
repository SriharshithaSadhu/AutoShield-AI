"""Audit possible overlap between the project sample and temporal CICIDS2017.

This script does not modify the pinned source files. It builds a conservative
feature-signature match set, reports the rows that would be excluded, and
records the remaining timestamped/labeled rows by source file and label.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_GLOB = "data/cicids2017_temporal/raw/*.parquet"
SAMPLE_PATH = ROOT / "data" / "CICIDS2017_sample_km.csv"
OUTPUT_PATH = ROOT / "data" / "cicids2017_temporal" / "overlap_audit.json"
CONTINUOUS_TOLERANCE = 1e-11

# Features that distinguish integer-valued/rounded raw CICFlow records. These
# are inverted from the sample's shared min-max representation and rounded.
INTEGER_SIGNATURE_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
]

# These are zero in the saved sample and are unusable for source matching;
# CICFlowMeter can emit non-finite values for zero-duration flows.
NONFINITE_RATE_FEATURES = {
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
}

RAW_TO_PROJECT_CLASS = """CASE
    WHEN r.Label = 'BENIGN' THEN 0
    WHEN r.Label = 'Bot' THEN 1
    WHEN r.Label IN ('FTP-Patator', 'SSH-Patator') THEN 2
    WHEN starts_with(r.Label, 'DoS') OR r.Label = 'DDoS' THEN 3
    WHEN r.Label = 'Infiltration' THEN 4
    WHEN r.Label = 'PortScan' THEN 5
    WHEN starts_with(r.Label, 'Web Attack') THEN 6
    ELSE -1
END"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sql_literal(value: float) -> str:
    return repr(float(value))


def main() -> None:
    sample = pd.read_csv(SAMPLE_PATH)
    source_glob = str(ROOT / SOURCE_GLOB)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    source_schema = {
        row[0]: row[1]
        for row in con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [source_glob]
        ).fetchall()
    }
    features = [column for column in sample.columns if column != "Label"]
    ranges: dict[str, tuple[float, float]] = {}
    for feature in features:
        lower, upper = con.execute(
            f'SELECT min("{feature}"), max("{feature}") '
            "FROM read_parquet(?) WHERE isfinite(\"{}\")".format(feature),
            [source_glob],
        ).fetchone()
        if lower is None or upper is None or upper == lower:
            continue
        if feature in NONFINITE_RATE_FEATURES:
            continue
        ranges[feature] = (float(lower), float(upper))

    non_key_features = [
        feature
        for feature in ranges
        if feature not in INTEGER_SIGNATURE_FEATURES
    ]
    expected: dict[str, np.ndarray] = {
        "sid": np.arange(len(sample), dtype=np.int32),
        "project_class": sample["Label"].astype(int).to_numpy(),
    }
    for feature in INTEGER_SIGNATURE_FEATURES:
        lower, upper = ranges[feature]
        expected[feature] = np.rint(
            lower + sample[feature].to_numpy(dtype=float) * (upper - lower)
        ).astype(np.int64)
    for index, feature in enumerate(non_key_features):
        expected[f"coordinate_{index}"] = sample[feature].to_numpy(dtype=float)
    con.register("sample_signatures", pd.DataFrame(expected))

    equality = [
        f'r."{feature}" = s."{feature}"'
        for feature in INTEGER_SIGNATURE_FEATURES
    ]
    equality.append(f"({RAW_TO_PROJECT_CLASS}) = s.project_class")
    join_predicate = " AND ".join(equality)

    continuous_predicates = []
    for index, feature in enumerate(non_key_features):
        lower, upper = ranges[feature]
        denominator = upper - lower
        coordinate = f"s.coordinate_{index}"
        normalized_raw = (
            f'(r."{feature}" - ({sql_literal(lower)}))'
            f' / ({sql_literal(denominator)})'
        )
        continuous_predicates.append(
            f'(r."{feature}" IS NOT NULL AND isfinite(r."{feature}") '
            f"AND abs({normalized_raw} - {coordinate}) "
            f"< {CONTINUOUS_TOLERANCE})"
        )
    full_signature_predicate = " AND ".join(continuous_predicates)

    raw_cte = f"""WITH raw AS (
        SELECT *
        FROM read_parquet(?, filename=true, file_row_number=true)
    ), matched AS (
        SELECT DISTINCT
            r.filename,
            r.file_row_number,
            r."Timestamp" AS timestamp,
            r."Flow ID" AS flow_id,
            r.Label AS raw_label,
            s.sid
        FROM raw r
        JOIN sample_signatures s ON {join_predicate}
        WHERE r."Timestamp" IS NOT NULL
          AND r.Label IS NOT NULL
          AND {full_signature_predicate}
    )"""

    matched_sample_rows = con.execute(
        raw_cte
        + " SELECT sid, count(DISTINCT file_row_number) AS candidates "
        "FROM matched GROUP BY sid",
        [source_glob],
    ).df()
    candidate_counts = matched_sample_rows["candidates"]

    baseline = con.execute(
        "SELECT filename, Label AS raw_label, count(*) AS valid_rows "
        "FROM read_parquet(?, filename=true) "
        "WHERE \"Timestamp\" IS NOT NULL AND Label IS NOT NULL "
        "GROUP BY 1, 2",
        [source_glob],
    ).df()
    exclusions = con.execute(
        raw_cte
        + " SELECT filename, raw_label, count(DISTINCT file_row_number) "
        "AS conservative_exclusion_rows FROM matched GROUP BY 1, 2",
        [source_glob],
    ).df()

    summary = baseline.merge(exclusions, how="left", on=["filename", "raw_label"])
    summary["conservative_exclusion_rows"] = (
        summary["conservative_exclusion_rows"].fillna(0).astype(np.int64)
    )
    summary["remaining_rows"] = (
        summary["valid_rows"] - summary["conservative_exclusion_rows"]
    )

    output = {
        "status": "CONSERVATIVE FEATURE-SIGNATURE OVERLAP AUDIT; NOT EXACT ROW-ID DEDUPLICATION",
        "sample_path": "data/CICIDS2017_sample_km.csv",
        "sample_sha256": sha256(SAMPLE_PATH),
        "source_glob": SOURCE_GLOB,
        "source_manifest": "data/cicids2017_temporal/manifest.json",
        "normalization_assumption": (
            "Each saved sample feature is compared with the same feature "
            "normalized by the timestamped source's finite global min/max. "
            "Integer-like fields are inverted and rounded; other nonconstant "
            "finite feature coordinates use absolute normalized tolerance "
            f"{CONTINUOUS_TOLERANCE:g}. This transform yielded a candidate "
            "raw match for every sample row, but exact generation-time scaler "
            "parameters have not been recovered independently."
        ),
        "signature_features": {
            "integer_key_features": INTEGER_SIGNATURE_FEATURES,
            "continuous_coordinate_features": non_key_features,
            "excluded_from_signature": sorted(NONFINITE_RATE_FEATURES),
            "class_map": {
                "0": "BENIGN",
                "1": "Bot",
                "2": "FTP-Patator or SSH-Patator",
                "3": "DoS variants or DDoS",
                "4": "Infiltration",
                "5": "PortScan",
                "6": "Web Attack variants",
            },
        },
        "sample_rows": int(len(sample)),
        "sample_unique_exact_rows": int(sample.drop_duplicates().shape[0]),
        "sample_rows_with_candidate_source_rows": int(len(matched_sample_rows)),
        "sample_rows_with_one_candidate_source_row": int(
            (candidate_counts == 1).sum()
        ),
        "sample_rows_with_multiple_candidate_source_rows": int(
            (candidate_counts > 1).sum()
        ),
        "candidate_rows_per_sample_row": {
            "median": float(candidate_counts.median()),
            "p90": float(candidate_counts.quantile(0.90)),
            "p99": float(candidate_counts.quantile(0.99)),
            "maximum": int(candidate_counts.max()),
        },
        "total_valid_timestamped_labeled_rows": int(summary["valid_rows"].sum()),
        "total_conservative_exclusion_rows": int(
            summary["conservative_exclusion_rows"].sum()
        ),
        "total_remaining_rows_if_signature_filtered": int(
            summary["remaining_rows"].sum()
        ),
        "by_file_and_label": [
            {
                "file": Path(row.filename).name,
                "raw_label": row.raw_label,
                "valid_rows": int(row.valid_rows),
                "conservative_exclusion_rows": int(
                    row.conservative_exclusion_rows
                ),
                "remaining_rows": int(row.remaining_rows),
            }
            for row in summary.sort_values(["filename", "raw_label"]).itertuples()
        ],
        "limitations": [
            "The project sample has no timestamps or persistent flow identifiers.",
            "A feature-signature match can identify many distinct timestamped flows with identical model inputs; all such rows are conservatively counted as exclusions.",
            "Rate features with non-finite source values are omitted from the match signature.",
            "The derived candidate filter can cause severe class-specific data loss and must not be treated as exact original-row identity.",
            "Do not use this source as confirmatory evidence until the transformation assumption and resulting retained class support are independently reviewed.",
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "sample_rows_with_candidates": output["sample_rows_with_candidate_source_rows"],
        "sample_rows_with_single_candidate": output["sample_rows_with_one_candidate_source_row"],
        "potentially_excluded_rows": output["total_conservative_exclusion_rows"],
        "remaining_rows": output["total_remaining_rows_if_signature_filtered"],
    }, indent=2))
    con.close()


if __name__ == "__main__":
    main()
