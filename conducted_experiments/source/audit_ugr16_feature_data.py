"""Audit pinned UGR'16 feature release before model experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ugr16_feature_data" / "raw"
OUT = ROOT / "data" / "ugr16_feature_data" / "audit.json"
ATTACK_COLUMNS = ["dos", "scan11", "scan44", "botnet", "blacklist", "udpscan", "sshscan", "spam"]


def read_pair(variant: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_csv(DATA / f"UGR16{variant}.X{split}.csv", low_memory=False)
    y = pd.read_csv(DATA / f"UGR16{variant}.Y{split}.csv", low_memory=False)
    if "Row" not in x or "Row" not in y:
        raise ValueError(f"Missing timestamp key in {variant} {split}")
    if x["Row"].astype(str).tolist() != y["Row"].astype(str).tolist():
        raise ValueError(f"Timestamp/order mismatch in {variant} {split} X/Y")
    if y.columns.tolist()[1:] != ATTACK_COLUMNS:
        raise ValueError(f"Unexpected label schema in {variant} {split}: {y.columns.tolist()}")
    return x, y


def summarize_split(x: pd.DataFrame, y: pd.DataFrame) -> dict:
    timestamps = pd.to_datetime(x["Row"].astype(str), format="%Y%m%d%H%M", errors="coerce")
    intervals = timestamps.diff().dropna().dt.total_seconds().div(60)
    feature_cols = [c for c in x.columns if c != "Row"]
    numeric = x[feature_cols].apply(pd.to_numeric, errors="coerce")
    attack_counts = y[ATTACK_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    active = attack_counts.gt(0)
    observed = timestamps.notna()
    valid_times = timestamps.loc[observed]
    expected_slots = 0 if valid_times.empty else int((valid_times.max() - valid_times.min()).total_seconds() // 60) + 1
    known_injected = active[["dos", "scan11", "scan44", "botnet"]].any(axis=1)
    return {
        "rows": int(len(x)),
        "feature_count": int(len(feature_cols)),
        "timestamp_first": None if not observed.any() else timestamps[observed].min().isoformat(),
        "timestamp_last": None if not observed.any() else timestamps[observed].max().isoformat(),
        "invalid_timestamps": int((~observed).sum()),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "non_increasing_timestamp_steps": int((intervals <= 0).sum()),
        "non_one_minute_steps": int((intervals != 1).sum()),
        "expected_minute_slots_between_first_and_last": expected_slots,
        "missing_minute_slots_between_first_and_last": max(0, expected_slots - int(valid_times.nunique())),
        "max_observed_gap_minutes": None if intervals.empty else float(intervals.max()),
        "missing_feature_values": int(numeric.isna().sum().sum()),
        "nonfinite_feature_values": int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()),
        "constant_features": numeric.columns[numeric.nunique(dropna=False) <= 1].tolist(),
        "label_count_totals": {str(c): int(pd.to_numeric(y[c], errors="coerce").fillna(0).sum()) for c in ATTACK_COLUMNS},
        "minutes_with_label_count": {str(c): int(active[c].sum()) for c in ATTACK_COLUMNS},
        "minutes_with_any_nonzero_label_count": int(active.any(axis=1).sum()),
        "minutes_with_no_nonzero_label_count": int((~active.any(axis=1)).sum()),
        "minutes_with_known_injected_attack_label": int(known_injected.sum()),
        "minutes_with_blacklist_only": int((active["blacklist"] & ~active[["dos", "scan11", "scan44", "botnet", "udpscan", "sshscan", "spam"]].any(axis=1)).sum()),
    }


def main() -> None:
    variants = {}
    for variant in ("v3", "v4"):
        splits = {}
        for split in ("train", "test"):
            x, y = read_pair(variant, split)
            splits[split] = summarize_split(x, y)
        variants[variant] = splits
    if variants["v3"]["test"]["rows"] != variants["v4"]["test"]["rows"]:
        raise ValueError("Expected v3/v4 to share test row count")
    report = {
        "dataset": "UGR'16 minute-level Feature-as-a-Counter data",
        "source": "Author-maintained josecamachop/UGR16_FeatureData, pinned in download_manifest.json",
        "scope": "Anonymized variants v3 (bidirectional source flows) and v4 (unidirectional source flows); training excludes June per published variant definition",
        "label_semantics": "Y matrices contain per-minute counts for eight attack/anomaly categories; row-level binary outcomes are not supplied directly",
        "variants": variants,
        "known_limitation": "Background is not verified benign; blacklist labels and some anomalies are not equivalent to confirmed injected attacks. Report all-label and injected-attack-only analyses separately.",
        "timing_limitation": "Features are traffic counters aggregated by minute; this supports window-level chronology, not per-flow latency or packet-level detection.",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
