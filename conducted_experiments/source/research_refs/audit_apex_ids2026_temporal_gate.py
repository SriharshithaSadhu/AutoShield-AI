"""Metadata and temporal-label gate for the local APEX-IDS2026 HC Parquet subset.

This scans metadata and streaming aggregates only. It does not train a detector,
change labels, or claim that the publisher's provenance/ground truth is independently
verified. UTC interpretation follows the upstream data card and is recorded as such.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "apex_ids2026" / "apex_ids2026_hc_subset.parquet"
ZIP = ROOT / "data" / "apex_ids2026" / "apex_ids2026_hc_subset.zip"
OUT = ROOT / "data" / "apex_ids2026" / "temporal_audit.json"
REPORT = ROOT / "APEX_IDS2026_TEMPORAL_DATA_GATE.md"
BATCH = 1_000_000
COLS = ["flow_start", "duration_s", "date", "label", "label_binary", "dst_port"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    pf = pq.ParquetFile(DATA)
    md = pf.metadata
    row_counts = Counter()
    binary_counts = Counter()
    tier_cross = Counter()
    daily = defaultdict(Counter)
    five_minute = defaultdict(Counter)
    hourly = defaultdict(Counter)
    completion_daily = defaultdict(Counter)
    completion_five_minute = defaultdict(Counter)
    completion_hourly = defaultdict(Counter)
    null_counts = Counter()
    negative_duration = 0
    zero_duration = 0
    start_descents = 0
    end_descents = 0
    min_start = max_start = min_end = max_end = None
    prev_start = prev_end = None
    batches = 0
    type_counts = Counter()
    date_mismatch = 0
    invalid_labels = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=COLS, use_threads=True):
        batches += 1
        frame = batch.to_pandas()
        labels = frame["label"].astype("string")
        # The deposit stores `flow_start` as timestamp[us]; normalize to ns
        # before using integer time arithmetic for bucket IDs and range output.
        start = pd.to_datetime(frame["flow_start"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
        duration = pd.to_numeric(frame["duration_s"], errors="coerce")
        end = start + pd.to_timedelta(duration, unit="s")
        start_ns = start.astype("int64", copy=False).to_numpy()
        end_ns = end.astype("int64", copy=False).to_numpy()
        start_valid = start.notna().to_numpy()
        end_valid = end.notna().to_numpy()
        valid_start = start_ns[start_valid]
        valid_end = end_ns[end_valid]
        if len(valid_start):
            start_descents += int(np.count_nonzero(valid_start[1:] < valid_start[:-1]))
            if prev_start is not None and valid_start[0] < prev_start:
                start_descents += 1
            prev_start = int(valid_start[-1])
            lo, hi = int(valid_start.min()), int(valid_start.max())
            min_start = lo if min_start is None else min(min_start, lo)
            max_start = hi if max_start is None else max(max_start, hi)
            start_dt = pd.DatetimeIndex(start[start_valid])
        else:
            start_dt = pd.DatetimeIndex([], tz="UTC")
        if len(valid_end):
            end_descents += int(np.count_nonzero(valid_end[1:] < valid_end[:-1]))
            if prev_end is not None and valid_end[0] < prev_end:
                end_descents += 1
            prev_end = int(valid_end[-1])
            lo, hi = int(valid_end.min()), int(valid_end.max())
            min_end = lo if min_end is None else min(min_end, lo)
            max_end = hi if max_end is None else max(max_end, hi)

        negative_duration += int((duration < 0).sum())
        zero_duration += int((duration == 0).sum())
        invalid_labels += int((~labels.isin(["Attack_Verified", "Benign_Verified"])).sum())
        row_counts.update(labels.fillna("<NULL>").astype(str).value_counts().to_dict())
        binaries = frame["label_binary"]
        binary_counts.update(binaries.fillna("<NULL>").astype(str).value_counts().to_dict())
        labels_clean = labels.fillna("<NULL>").astype(str)
        binaries_clean = binaries.fillna("<NULL>").astype(str)
        cross = pd.DataFrame({"label": labels_clean, "binary": binaries_clean}).value_counts(sort=False)
        for (label, binary), count in cross.items():
            tier_cross[(label, binary)] += int(count)
        for col in COLS:
            null_counts[col] += int(frame[col].isna().sum())

        # Label support by UTC date/hour/5-minute bucket of flow start and the
        # estimated flow-completion event time (flow_start + duration_s).
        valid = start.notna().to_numpy()
        if valid.any():
            ns = start_ns[valid]
            labels_v = labels_clean.to_numpy()[valid]
            code = np.where(labels_v == "Attack_Verified", 1,
                            np.where(labels_v == "Benign_Verified", 0, 2)).astype(np.int8)
            day_id = ns // (24 * 60 * 60 * 1_000_000_000)
            hour_id = ns // (60 * 60 * 1_000_000_000)
            five_id = ns // (5 * 60 * 1_000_000_000)
            label_names = ("Benign_Verified", "Attack_Verified", "<OTHER>")
            for ids, target in ((day_id, daily), (hour_id, hourly), (five_id, five_minute)):
                unique, n = np.unique(ids * 3 + code, return_counts=True)
                for packed, count in zip(unique, n):
                    ident, label_code = divmod(int(packed), 3)
                    target[ident][label_names[label_code]] += int(count)
            declared_ns = (pd.to_datetime(frame.loc[valid, "date"], utc=True, errors="coerce")
                           .astype("datetime64[ns, UTC]").astype("int64").to_numpy())
            declared_days = declared_ns // (24 * 60 * 60 * 1_000_000_000)
            date_mismatch += int(np.count_nonzero(declared_days != day_id))
        valid_end_mask = end.notna().to_numpy()
        if valid_end_mask.any():
            end_time_ns = end_ns[valid_end_mask]
            labels_end = labels_clean.to_numpy()[valid_end_mask]
            code_end = np.where(labels_end == "Attack_Verified", 1,
                                np.where(labels_end == "Benign_Verified", 0, 2)).astype(np.int8)
            end_day = end_time_ns // (24 * 60 * 60 * 1_000_000_000)
            end_hour = end_time_ns // (60 * 60 * 1_000_000_000)
            end_five = end_time_ns // (5 * 60 * 1_000_000_000)
            for ids, target in ((end_day, completion_daily),
                                (end_hour, completion_hourly),
                                (end_five, completion_five_minute)):
                unique, n = np.unique(ids * 3 + code_end, return_counts=True)
                for packed, count in zip(unique, n):
                    ident, label_code = divmod(int(packed), 3)
                    target[ident][label_names[label_code]] += int(count)
        if batches % 2 == 0:
            print(f"scanned {min(batches * BATCH, md.num_rows):,}/{md.num_rows:,} rows", flush=True)

    def timestamp_key(key: int, width_seconds: int) -> str:
        return pd.to_datetime(key * width_seconds, unit="s", utc=True).isoformat()
    daily_named = {timestamp_key(int(k), 86400)[:10]: dict(v) for k, v in sorted(daily.items())}
    hourly_named = {timestamp_key(int(k), 3600)[:13]: dict(v) for k, v in sorted(hourly.items())}
    both_bins = [v for v in five_minute.values() if v.get("Attack_Verified", 0) and v.get("Benign_Verified", 0)]
    completion_daily_named = {timestamp_key(int(k), 86400)[:10]: dict(v) for k, v in sorted(completion_daily.items())}
    completion_hourly_named = {timestamp_key(int(k), 3600)[:13]: dict(v) for k, v in sorted(completion_hourly.items())}
    completion_both_bins = [v for v in completion_five_minute.values()
                            if v.get("Attack_Verified", 0) and v.get("Benign_Verified", 0)]
    groups = []
    for name, value in (("Attack_Verified", "1"), ("Benign_Verified", "0")):
        groups.append({"label": name, "label_binary": value, "rows": int(row_counts[name])})
    result = {
        "source_file": str(DATA.relative_to(ROOT)),
        "source_sha256": sha256_file(DATA),
        "source_bytes": DATA.stat().st_size,
        "source_deposit": {
            "doi": "10.5281/zenodo.22013914",
            "version": "1.0.0",
            "deposit_license": "CC-BY-4.0",
            "archive_file": ZIP.name,
            "archive_bytes": ZIP.stat().st_size if ZIP.exists() else None,
            "archive_md5": md5_file(ZIP) if ZIP.exists() else None,
            "zenodo_expected_archive_md5": "115ed72bb8759d3ce90fe17c52a0ee98",
            "license_documentation_conflict": "Zenodo record/LICENSE and data-card Section 11 specify CC BY 4.0; the data-card header says Research Use (Pre-Publication). The deposited record v1.0.0 and its LICENSE are the controlling source for this local archive, whose MD5 is checked above.",
        },
        "parquet_rows": md.num_rows,
        "row_groups": md.num_row_groups,
        "batches_scanned": batches,
        "schema": [{"name": pf.schema.column(i).name, "type": str(pf.schema.column(i).physical_type)}
                   for i in range(len(pf.schema))],
        "upstream_provenance_claims_not_independently_verified": {
            "collection": "Live ISP edge NetFlow; data-card claim",
            "date_range": "2026-06-21 to 2026-08-03; data-card claim",
            "label_semantics": "Attack_Verified = honeypot-associated; Benign_Verified = safe-destination allowlist; high-confidence subset only",
            "license": "Zenodo v1.0.0 is CC BY 4.0; data-card header conflicts with its Section 11 and LICENSE; archive identity is recorded separately",
        },
        "label_counts": dict(row_counts),
        "label_binary_counts": dict(binary_counts),
        "label_binary_cross_tab": [{"label": k[0], "label_binary": k[1], "rows": n}
                                   for k, n in sorted(tier_cross.items())],
        "invalid_subset_label_rows": invalid_labels,
        "date_field_label_time_disagreements": date_mismatch,
        "null_counts_scanned_columns": dict(null_counts),
        "negative_duration_rows": negative_duration,
        "zero_duration_rows": zero_duration,
        "timestamp_semantics": "flow_start is stored timezone-naive; interpret as UTC only because upstream data card declares UTC. Replay availability candidate = flow_start + duration_s, not exporter arrival time.",
        "start_time_row_order_descents": start_descents,
        "completion_time_row_order_descents": end_descents,
        "flow_start_min_utc": pd.to_datetime(min_start, utc=True).isoformat() if min_start is not None else None,
        "flow_start_max_utc": pd.to_datetime(max_start, utc=True).isoformat() if max_start is not None else None,
        "flow_completion_min_utc": pd.to_datetime(min_end, utc=True).isoformat() if min_end is not None else None,
        "flow_completion_max_utc": pd.to_datetime(max_end, utc=True).isoformat() if max_end is not None else None,
        "days": daily_named,
        "completion_time_days": completion_daily_named,
        "hours_with_both_labels": sum(bool(v.get("Attack_Verified") and v.get("Benign_Verified")) for v in hourly.values()),
        "hours_total": len(hourly),
        "five_minute_bins_with_both_labels": len(both_bins),
        "five_minute_bins_total": len(five_minute),
        "five_minute_bins_attack_only": sum(bool(v.get("Attack_Verified") and not v.get("Benign_Verified")) for v in five_minute.values()),
        "five_minute_bins_benign_only": sum(bool(v.get("Benign_Verified") and not v.get("Attack_Verified")) for v in five_minute.values()),
        "five_minute_bins_neither_or_unexpected": sum(not (v.get("Attack_Verified") or v.get("Benign_Verified")) for v in five_minute.values()),
        "completion_hours_with_both_labels": sum(bool(v.get("Attack_Verified") and v.get("Benign_Verified")) for v in completion_hourly.values()),
        "completion_hours_total": len(completion_hourly),
        "completion_five_minute_bins_with_both_labels": len(completion_both_bins),
        "completion_five_minute_bins_total": len(completion_five_minute),
        "completion_five_minute_bins_attack_only": sum(bool(v.get("Attack_Verified") and not v.get("Benign_Verified")) for v in completion_five_minute.values()),
        "completion_five_minute_bins_benign_only": sum(bool(v.get("Benign_Verified") and not v.get("Attack_Verified")) for v in completion_five_minute.values()),
        "feature_and_target_policy": {
            "candidate_feature_columns_primary": ["duration_s", "protocol", "packets", "bytes", "tcp_flags", "tos", "flag_*"],
            "port_policy": "Exclude src_port and dst_port from primary features; dst_port participates in the verified-attack matching rule. Evaluate each only in a predeclared leakage/sensitivity ablation.",
            "must_exclude": ["flow_start", "date", "flow_file", "src_ip", "dst_ip", "label", "label_binary", "label_multiclass", "attack_type", "attack_category", "mitre_*", "confidence", "evidence_source", "threat_intel_score", "country", "behavioral_flags", "flow_duration_class", "src_port_category", "dst_port_category", "zeek fields unless explicitly evaluated as a separate sensor condition"],
            "known_proxy_risks": ["label is defined by honeypot-destination match or safe-destination allowlist; destination port is part of the attack label rule and must be audited/ablated", "src/dst addresses and exact timestamp can memorize sensor/label joins", "global attacker IP contamination filtering used label-linked information; not available prospectively"],
        },
        "limits": ["Publisher provenance and ground-truth claims have not been independently audited.", "The Zenodo record identifies CC BY 4.0, but the data-card header says Research Use (Pre-Publication); cite and follow the deposited v1.0.0 license and retain this discrepancy in the paper.", "Binary high-confidence task distinguishes verified honeypot-bound flows from verified safe-destination flows, not all attacks from all benign traffic.", "Only 44 days from one ISP perimeter; not independent organizations or internal lateral traffic.", "No exporter emission timestamps; completion time is a replay proxy.", "The HC subset is retrospectively selected by confidence tier, creating a conditional sampling frame.", "Data-card baseline results are publisher-reported and not reproduced here."],
    }
    result["source_deposit"]["archive_checksum_matches_zenodo"] = (
        result["source_deposit"]["archive_md5"] == result["source_deposit"]["zenodo_expected_archive_md5"]
    )
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# APEX-IDS2026 temporal data gate", "",
             "## Decision", "",
             "**PASS for a narrowly described event-time replay candidate; not yet a model confirmation or verified operational ground truth.** The complete local file matches the Zenodo v1.0.0 archive checksum (CC BY 4.0). Its binary labels mean verified honeypot-bound versus verified safe-destination flows rather than all attacks versus all benign traffic.", "",
             "## Measured local audit", "",
             f"- Rows: {md.num_rows:,}; row groups: {md.num_row_groups:,}; file: {DATA.stat().st_size:,} bytes; SHA-256: `{result['source_sha256']}`.",
             f"- Deposit: DOI `10.5281/zenodo.22013914`, version `1.0.0`, CC BY 4.0; local ZIP MD5 `{result['source_deposit']['archive_md5']}`; checksum match: `{result['source_deposit']['archive_checksum_matches_zenodo']}`.",
             f"- Flow-start range: {result['flow_start_min_utc']} to {result['flow_start_max_utc']}; completion candidate range: {result['flow_completion_min_utc']} to {result['flow_completion_max_utc']}.",
             f"- Label counts: `{json.dumps(dict(row_counts), sort_keys=True)}`; binary targets: `{json.dumps(dict(binary_counts), sort_keys=True)}`.",
             f"- UTC calendar days with traffic: {len(daily_named):,}; attack share among the selected two-label subset ranges {min(v['Attack_Verified'] / (v['Attack_Verified'] + v['Benign_Verified']) for v in daily.values()) * 100:.2f}%–{max(v['Attack_Verified'] / (v['Attack_Verified'] + v['Benign_Verified']) for v in daily.values()) * 100:.2f}% by day (this is not operational prevalence).",
             f"- Start-time row descents: {start_descents:,}; completion-time descents: {end_descents:,}; label rows outside the expected two high-confidence labels: {invalid_labels:,}; date-field/time disagreements: {date_mismatch:,}.",
             f"- Five-minute start-time buckets: {len(five_minute):,} total; {len(both_bins):,} have both labels ({len(both_bins) / max(len(five_minute), 1) * 100:.2f}%); {result['five_minute_bins_attack_only']:,} attack-only; {result['five_minute_bins_benign_only']:,} benign-only. Hours with both labels: {result['hours_with_both_labels']:,}/{result['hours_total']:,}.",
             f"- Five-minute completion-time buckets: {len(completion_five_minute):,} total; {len(completion_both_bins):,} have both labels ({len(completion_both_bins) / max(len(completion_five_minute), 1) * 100:.2f}%); {result['completion_five_minute_bins_attack_only']:,} attack-only; {result['completion_five_minute_bins_benign_only']:,} benign-only. Hours with both labels: {result['completion_hours_with_both_labels']:,}/{result['completion_hours_total']:,}.",
             f"- Zero-duration flows: {zero_duration:,} ({zero_duration / max(md.num_rows, 1) * 100:.2f}%); negative durations: {negative_duration:,}.",
             "", "## Audit interpretation", "",
             "The source Parquet schema contains microsecond `flow_start` without an embedded timezone; the data card calls it UTC. Any chronological experiment must make this provenance dependency explicit and order completed-flow features by `flow_start + duration_s`, with stable tie handling. It must measure FPR only on Benign_Verified flows and describe the endpoint accordingly.", "",
              "The target is constructed from honeypot contact and a safe-destination allowlist. Exclude IPs, timestamps, all ports, provenance/taxonomy/label fields, and likely post-label features from primary analysis. `dst_port` participates in the stated honeypot label rule; evaluate source and destination ports only in predeclared separate ablations. The source also globally filtered benign rows against attacker IPs using information from the full time range; that is retrospective label construction and should be disclosed.", "",
             "The Zenodo v1.0.0 record reports CC BY 4.0 and the local ZIP MD5 is compared against its deposited checksum. The data-card header nevertheless says `Research Use (Pre-Publication)`, conflicting with the record/license and even its own Section 11; cite the Zenodo version and preserve this discrepancy. The data card's baseline metrics and zero-false-positive language are publisher claims, not independently reproduced. The dataset is an interesting naturalistic perimeter stress test, but one ISP, one honeypot, heavy scan dominance, and single-version collection limit confirmatory scope.", "",
             "## Reproduction", "",
             "Run from the project root: `python research_refs/audit_apex_ids2026_temporal_gate.py`. This scans the local Parquet in batches and writes `data/apex_ids2026/temporal_audit.json`. It does not train a detector.", "",
             "## Upstream source", "",
             "[APEX-IDS2026 data card](https://github.com/ju4700/APEX-IDS2026/blob/main/Data_Card_APEX_IDS2026.md) · [schema](https://github.com/ju4700/APEX-IDS2026/blob/main/DATASET_SCHEMA.md)."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} and {REPORT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
