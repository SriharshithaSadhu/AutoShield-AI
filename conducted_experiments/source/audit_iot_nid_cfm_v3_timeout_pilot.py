"""Audit the bounded CICFlowMeter-V3 IoT-NID extraction/timeout pilot.

The analysis uses two creator-provided PCAPs and their timestamp-aligned rows
in the official IoTID20 CSV. A four-hour timestamp offset was established by
the prior raw-capture inventory audit and is treated as observed metadata,
not a verified timezone configuration.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "iot_nid_cfm_v3_pilot"
OFFICIAL = ROOT / "data" / "provenance" / "IoTID20_official_original.csv"
SOURCE_MANIFEST = ROOT / "data" / "iot_nid_raw" / "manifest.json"
CFM_REPO = ROOT / "research_refs" / "CICFlowMeter-V3"
RESULT_JSON = OUTPUT_DIR / "audit.json"
REPORT = OUTPUT_DIR / "README.md"
TIME_OFFSET_HOURS = 4
ACTIVITY_TIMEOUT_US = 5_000_000
IDENTITY_COLUMNS = {"Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Label"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(name: str) -> str:
    return name.replace("_", " ")


def counts_and_feature_comparison(generated_path: Path, official: pd.DataFrame, raw_manifest: dict) -> dict:
    generated = pd.read_csv(generated_path, low_memory=False)
    generated["_aligned_time"] = pd.to_datetime(
        generated["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    low, high = generated["_aligned_time"].min(), generated["_aligned_time"].max()
    if generated_path.name.startswith("benign-"):
        expected_cat, expected_subcat = "Normal", "Normal"
    else:
        expected_cat, expected_subcat = "DoS", "DoS-Synflooding"
    reference = official[
        official["Cat"].eq(expected_cat)
        & official["Sub_Cat"].eq(expected_subcat)
        & official["_aligned_time"].between(low, high)
    ].copy()
    generated["_key"] = generated["Flow ID"].astype(str) + "|" + generated["_aligned_time"].astype(str)
    reference["_key"] = reference["Flow_ID"].astype(str) + "|" + reference["_aligned_time"].astype(str)

    generated_counts = generated["_key"].value_counts()
    reference_counts = reference["_key"].value_counts()
    generated_unique = set(generated_counts.index)
    reference_unique = set(reference_counts.index)
    common_keys = generated_unique & reference_unique
    one_to_one_keys = {
        key for key in common_keys
        if generated_counts[key] == 1 and reference_counts[key] == 1
    }

    generated_name_map = {normalize_name(c): c for c in generated.columns}
    official_name_map = {normalize_name(c): c for c in reference.columns}
    feature_names = sorted(set(generated_name_map) & set(official_name_map) - {
        "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Label",
        "Cat", "Sub Cat", "aligned time", "key",
        "_aligned time", "_key", " aligned time", " key",
    })
    if len(feature_names) != 77:
        raise ValueError(f"Expected 77 shared predictors; found {len(feature_names)}")

    # Limit direct feature comparison to unambiguous one-to-one flow/time keys.
    left = generated[generated["_key"].isin(one_to_one_keys)][["_key"] + [generated_name_map[n] for n in feature_names]].copy()
    right = reference[reference["_key"].isin(one_to_one_keys)][["_key"] + [official_name_map[n] for n in feature_names]].copy()
    left = left.rename(columns={generated_name_map[n]: n for n in feature_names})
    right = right.rename(columns={official_name_map[n]: n for n in feature_names})
    pairs = left.merge(right, on="_key", suffixes=("_generated", "_official"), validate="one_to_one")

    feature_agreement = []
    full_vector_equal = np.ones(len(pairs), dtype=bool)
    for name in feature_names:
        gen_values = pd.to_numeric(pairs[f"{name}_generated"], errors="coerce").to_numpy(dtype=np.float64)
        off_values = pd.to_numeric(pairs[f"{name}_official"], errors="coerce").to_numpy(dtype=np.float64)
        equal = (gen_values == off_values) | (np.isnan(gen_values) & np.isnan(off_values))
        full_vector_equal &= equal
        finite = np.isfinite(gen_values) & np.isfinite(off_values)
        feature_agreement.append({
            "feature": name,
            "exact_agreement_fraction": float(equal.mean()) if len(equal) else None,
            "finite_pair_count": int(finite.sum()),
            "median_absolute_difference": float(np.median(np.abs(gen_values[finite] - off_values[finite])))
            if finite.any() else None,
        })

    input_name = "benign-dec.pcap" if expected_cat == "Normal" else "dos-synflooding-5-dec.pcap"
    input_record = next(item for item in raw_manifest["files"] if item["path"].endswith(input_name))
    timeout_us = int(generated_path.stem.rsplit("_", 1)[1])
    return {
        "capture": input_name,
        "capture_sha256": input_record["sha256"],
        "capture_bytes": input_record["bytes"],
        "flow_timeout_microseconds": timeout_us,
        "activity_timeout_microseconds": ACTIVITY_TIMEOUT_US,
        "generated_csv": str(generated_path.relative_to(ROOT).as_posix()),
        "generated_csv_sha256": sha256(generated_path),
        "generated_flow_rows": int(len(generated)),
        "generated_unique_flow_timestamp_keys": int(generated["_key"].nunique()),
        "official_category": expected_cat,
        "official_subcategory": expected_subcat,
        "official_rows_in_generated_time_window": int(len(reference)),
        "generated_minus_official_rows": int(len(generated) - len(reference)),
        "generated_to_official_row_count_ratio": float(len(generated) / len(reference)) if len(reference) else None,
        "generated_timestamp_min": low.isoformat(),
        "generated_timestamp_max": high.isoformat(),
        "official_timestamp_offset_hours_applied": TIME_OFFSET_HOURS,
        "official_unique_flow_timestamp_keys": int(reference["_key"].nunique()),
        "common_flow_timestamp_keys": int(len(common_keys)),
        "one_to_one_flow_timestamp_pairs": int(len(pairs)),
        "exact_77_feature_vectors_among_one_to_one_pairs": int(full_vector_equal.sum()),
        "median_per_feature_exact_agreement": float(np.median([
            item["exact_agreement_fraction"] for item in feature_agreement
            if item["exact_agreement_fraction"] is not None
        ])) if feature_agreement else None,
        "features_exact_on_every_one_to_one_pair": int(sum(
            item["exact_agreement_fraction"] == 1.0 for item in feature_agreement
        )),
        "feature_agreement": feature_agreement,
    }


def main() -> None:
    raw_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    official = pd.read_csv(OFFICIAL, low_memory=False)
    official["_aligned_time"] = pd.to_datetime(
        official["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce"
    ) + pd.Timedelta(hours=TIME_OFFSET_HOURS)
    generated_files = sorted(OUTPUT_DIR.glob("*_flowTimeout_*.csv"))
    if not generated_files:
        raise FileNotFoundError(f"No extracted CFM-V3 CSVs found under {OUTPUT_DIR}")
    comparisons = [counts_and_feature_comparison(path, official, raw_manifest) for path in generated_files]
    try:
        cfm_commit = subprocess.check_output(
            ["git", "-C", str(CFM_REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        cfm_commit = None

    result = {
        "status": "complete",
        "date": "2026-11-03",
        "experiment": "bounded CICFlowMeter-V3 timeout sensitivity and IoTID20 published-flow comparison",
        "development_only": True,
        "implementation": {
            "version": "CICFlowMeter-V3 3.0",
            "git_commit": cfm_commit,
            "license": "CICFlowMeter repository LICENSE.txt (permissive MIT-style grant; see source file)",
            "runner": "research_refs/cfm_timeout_pilot/CICFlowMeterTimeoutPilot.java",
            "runner_change": "wrapper only; delegates packet parsing, flow grouping, features, and CSV serialization to the checked-out CICFlowMeter-V3 classes",
            "runtime": "Ubuntu 24.04 WSL, OpenJDK 8u504, bundled Linux jnetpcap 1.4.r1425, libpcap 1.10.4",
            "baseline_flow_timeout_microseconds": 120_000_000,
            "activity_timeout_microseconds": ACTIVITY_TIMEOUT_US,
            "source_time_unit": "PacketReader uses timestampInMicros; generated Flow Duration is in microseconds",
        },
        "comparison": {
            "source_dataset": "official IoTID20 processed CSV",
            "source_csv_sha256": sha256(OFFICIAL),
            "raw_capture_archive_sha256": raw_manifest["archive_sha256"],
            "timestamp_alignment": "official CSV timestamps shifted +4 hours to align to raw-PCAP/CICFlowMeter output; observed in the prior raw-capture alignment audit, not asserted as verified timezone configuration",
            "row_identity": "Flow ID plus second-resolution timestamp",
            "feature_check": "77 shared numeric predictors, exact equality, restricted to unambiguous one-to-one identity/time pairs",
            "captures": ["benign-dec.pcap", "dos-synflooding-5-dec.pcap"],
            "timeout_values_microseconds": sorted({item["flow_timeout_microseconds"] for item in comparisons}),
        },
        "captures": comparisons,
        "limits": [
            "Two captures do not establish behavior for all 42 IoT-NID captures or all CSE flows.",
            "Timeout changes are experimental sensitivities, not evidence of the timeout used by the IoTID20 publishers.",
            "Flow count agreement is not equivalent to feature-row agreement; the full 77-feature vector matched none of the unambiguous pairs in this pilot.",
            "The comparison does not attribute discrepancies uniquely to software version; timeout, flow key, direction, duplicate-first-packet behavior, and packet-label aggregation may contribute.",
            "The normal capture has many repeated Flow ID/second keys, so only 139 of 40,073 normal reference rows can be compared under an unambiguous one-to-one key at the setting whose generated count is nearest; these feature agreement rates are not population estimates.",
        ],
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CICFlowMeter-V3 bounded extraction and timeout pilot",
        "",
        "**Status:** reproducible two-capture provenance pilot; development evidence only.",
        "",
        f"Pinned CICFlowMeter-V3 version 3.0 commit `{cfm_commit}` was compiled under Ubuntu 24.04 WSL / OpenJDK 8u504 using its bundled Linux jnetpcap 1.4.r1425 library and system libpcap. The wrapper passes the PCAPs to the repository's existing `FlowGenerator`; it changes no feature code. The default tested flow timeout is 120 seconds (120,000,000 microseconds), with activity timeout held at 5 seconds.",
        "",
        f"The two creator-provided captures are compared with official IoTID20 rows in their observed timestamp-aligned windows using a +{TIME_OFFSET_HOURS}-hour offset. Flow counts are summarized below. Count agreement is not treated as feature equivalence.",
        "",
        "| Capture | Timeout (µs) | CFM-V3 rows | Official rows in window | Count difference | 1:1 flow/time pairs | Exact 77-feature vectors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['capture']} | {item['flow_timeout_microseconds']:,} | {item['generated_flow_rows']:,} | "
            f"{item['official_rows_in_generated_time_window']:,} | {item['generated_minus_official_rows']:+,} | "
            f"{item['one_to_one_flow_timestamp_pairs']:,} | {item['exact_77_feature_vectors_among_one_to_one_pairs']:,} |"
        )
    lines += [
        "",
        "## What the measurements show",
        "",
        "- The 40,073-row official benign window is approached at a 500-µs flow timeout (40,270 generated rows, +0.49%), but the paired attack PCAP then yields only 223 rows versus 5,214 official rows in its time window.",
        "- Attack row counts are close at 100 ms (5,215 generated / 5,215 official) and at the default 120 seconds (5,179 / 5,212), while the benign output at 100 ms is only 3,415 rows and at 120 seconds is 496 rows.",
        "- Across the tested settings, count parity is capture-dependent. None of the unambiguous one-to-one flow/time pairs has an exact match across all 77 predictors, including at count-matched settings.",
        "- At the 100-ms attack setting, 5,187 flow/time pairs are one-to-one. The median per-feature exact agreement is high (99.44%), but packet-count, subflow-count, packet-rate, header-length, and down/up fields differ systematically. The benign setting with near-matched row counts has 139 one-to-one pairs and a median per-feature exact agreement of 67.63%.",
        "- These results support a specific, testable concern: a single global timeout can affect benign and attack flow generation very differently. They do not prove which timeout or exporter generated the official CSV, or that exporter mismatch alone caused IDS transfer failure.",
        "",
        "## Limits and next step",
        "",
        "The source CSV has 40,073 benign rows from one benign capture; the attack sample is one of several DoS-Synflooding captures. Row identity is Flow ID plus second-resolution timestamp, which is ambiguous for repeated tuples, especially in the benign capture. Feature agreement is calculated only for the 1:1 subset, not as a population estimate. Packet labels remain the official capture/category labels; this pilot does not independently reconstruct attack packet filters or flow-level labels.",
        "",
        "Next work should inspect all 42 capture-specific timeout and feature-generation relationships only after documenting the IoTID20 label lifting rule and selecting a predeclared subset. Do not tune a timeout to force agreement, attribute all drift to the exporter, or scale this pilot into a novelty claim. HERA's existing exporter-sensitivity work remains the closest adjacent baseline; a distinct contribution would need matched PCAPs, labels, resource costs, transfer evaluation, and a locked independent stream.",
        "",
        f"Machine-readable measurements and feature-level diagnostics: `{RESULT_JSON.relative_to(ROOT).as_posix()}`.",
        "Reproduction command from the project root: `research_refs/cfm_timeout_pilot/run_cfm_v3_timeout_pilot.ps1`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "captures": [
        {"capture": x["capture"], "timeout_us": x["flow_timeout_microseconds"],
         "generated": x["generated_flow_rows"], "official": x["official_rows_in_generated_time_window"],
         "pairs": x["one_to_one_flow_timestamp_pairs"],
         "exact77": x["exact_77_feature_vectors_among_one_to_one_pairs"]}
        for x in comparisons
    ], "json": str(RESULT_JSON), "report": str(REPORT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
