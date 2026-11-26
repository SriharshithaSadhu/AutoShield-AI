"""Audit exact duplicate overlap in the reproduced IoTID20 random split."""
from __future__ import annotations

import json
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "provenance" / "IoTID20_official_original.csv"
SAMPLE = ROOT / "data" / "IoTID20_sample_0.05.csv"
CAPTURE_AUDIT = ROOT / "results" / "iot_nid_raw_capture_inventory_20261102.json"
CHUNK_MANIFEST = ROOT / "data" / "cse_replay_chunk_cache_v2" / "manifest.json"
OUT = ROOT / "results" / "iotid20_split_integrity_audit_20261103.json"


def canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_overlap(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> dict[str, int | float]:
    train_hash = pd.util.hash_pandas_object(train[columns], index=False).to_numpy(dtype=np.uint64)
    test_hash = pd.util.hash_pandas_object(test[columns], index=False).to_numpy(dtype=np.uint64)
    labels: dict[int, set[int]] = defaultdict(set)
    for signature, label in zip(train_hash, train["Label"].astype(int)):
        labels[int(signature)].add(int(label))
    overlaps = [int(signature) in labels for signature in test_hash]
    conflicts = [
        bool(overlap and any(label != int(test_label) for label in labels[int(signature)]))
        for overlap, signature, test_label in zip(overlaps, test_hash, test["Label"])
    ]
    return {
        "columns": len(columns),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_rows_with_exact_feature_signature_in_train": int(sum(overlaps)),
        "fraction_of_test_rows_with_signature_in_train": float(np.mean(overlaps)),
        "overlapping_test_rows_with_conflicting_five_class_label": int(sum(conflicts)),
        "overlap_label_conflict_fraction": float(sum(conflicts) / max(1, sum(overlaps))),
    }


def main() -> None:
    source_header = pd.read_csv(SOURCE, nrows=0).columns.tolist()
    full = pd.read_csv(SOURCE, low_memory=False)
    input_columns = [column for column in source_header if column not in {"Label", "Cat", "Sub_Cat"}]
    exact_duplicate_extras_all86 = int(full.duplicated(keep="first").sum())
    exact_duplicate_extras_input83 = int(full[input_columns].duplicated(keep="first").sum())
    unique_all86 = full.drop_duplicates(keep="first")

    sample = pd.read_csv(SAMPLE, low_memory=False)
    if len(sample.columns) - 1 != 83:
        raise ValueError("Expected the reproduced IoTID20 notebook to use 83 numeric inputs")
    train_idx, test_idx = train_test_split(
        np.arange(len(sample)), train_size=0.5, test_size=0.5, random_state=0,
    )
    train = sample.iloc[train_idx].reset_index(drop=True)
    test = sample.iloc[test_idx].reset_index(drop=True)
    notebook_inputs = [column for column in sample.columns if column != "Label"]

    manifest = json.loads(CHUNK_MANIFEST.read_text(encoding="utf-8"))
    common77 = manifest["features"]
    sample_map: dict[str, list[str]] = defaultdict(list)
    for column in sample.columns:
        sample_map[canonical(column)].append(column)
    common77_sample = [sample_map[canonical(column)][0] for column in common77
                       if len(sample_map[canonical(column)]) == 1]
    if len(common77_sample) != 77:
        raise ValueError("Cannot map the notebook sample to the frozen common-77 feature list")

    all83_overlap = hash_overlap(train, test, notebook_inputs)
    common77_overlap = hash_overlap(train, test, common77_sample)

    # Original IoTID20 codes: 0=DoS, 1=MITM, 2=Mirai, 3=Normal, 4=Scan.
    # Check whether common-77 overlap is also ambiguous for the binary task.
    train_hash = pd.util.hash_pandas_object(train[common77_sample], index=False).to_numpy(dtype=np.uint64)
    test_hash = pd.util.hash_pandas_object(test[common77_sample], index=False).to_numpy(dtype=np.uint64)
    binary_labels = lambda values: (values.astype(int) != 3).astype(np.int8)
    train_binary = binary_labels(train["Label"])
    test_binary = binary_labels(test["Label"])
    binary_by_signature: dict[int, set[int]] = defaultdict(set)
    for signature, label in zip(train_hash, train_binary):
        binary_by_signature[int(signature)].add(int(label))
    binary_overlap_rows = [int(signature) in binary_by_signature for signature in test_hash]
    binary_conflicts = [
        bool(overlap and any(label != int(test_label) for label in binary_by_signature[int(signature)]))
        for overlap, signature, test_label in zip(binary_overlap_rows, test_hash, test_binary)
    ]
    common77_overlap["overlapping_test_rows_with_conflicting_binary_label"] = int(sum(binary_conflicts))
    common77_overlap["overlap_binary_conflict_fraction"] = float(sum(binary_conflicts) / max(1, sum(binary_overlap_rows)))

    capture = json.loads(CAPTURE_AUDIT.read_text(encoding="utf-8"))
    normal = full["Cat"].astype(str).str.strip().str.casefold().eq("normal")
    timestamps = pd.to_datetime(full["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    normal_capture = next(item for item in capture["captures"] if item["file"] == "benign-dec.pcap")
    aligned_epoch = timestamps.astype("int64").to_numpy(dtype=np.float64) / 1e9 + 240 * 60
    valid = timestamps.notna().to_numpy()
    normal_interval = (
        (aligned_epoch >= normal_capture["first_packet_epoch"] - 1.0)
        & (aligned_epoch <= normal_capture["last_packet_epoch"] + 1.0)
        & valid
    )

    result = {
        "status": "complete",
        "date": "2026-11-03",
        "study": "IoTID20 exact-duplicate and random-split overlap integrity audit",
        "source_file": str(SOURCE.relative_to(ROOT)),
        "source_sha256": file_sha256(SOURCE),
        "source_rows": len(full),
        "published_exact_duplicate_reproduction": {
            "key": "all 86 columns, including Label, Cat and Sub_Cat",
            "duplicate_rows_removed_by_drop_duplicates": exact_duplicate_extras_all86,
            "rows_after_exact_deduplication": int(len(unique_all86)),
            "post_dedup_binary_counts": {
                "benign": int(unique_all86["Label"].astype(str).str.strip().str.casefold().eq("normal").sum()),
                "attack": int(unique_all86["Label"].astype(str).str.strip().str.casefold().eq("anomaly").sum()),
            },
            "input-only_exact_duplicate_rows_removed": exact_duplicate_extras_input83,
        },
        "notebook_split": {
            "sample_file": str(SAMPLE.relative_to(ROOT)),
            "rows": len(sample),
            "input_columns": len(notebook_inputs),
            "procedure": "train_test_split(train_size=0.5, test_size=0.5, random_state=0)",
            "five_class_codes": {"0": "DoS", "1": "MITM", "2": "Mirai", "3": "Normal", "4": "Scan"},
            "exact_full_input_overlap": all83_overlap,
            "common77_traffic_signature_overlap": common77_overlap,
            "interpretation": [
                "The 83-input overlap is an exact duplicate model-input signature crossing the original row-random train/test boundary.",
                "Common-77 overlap ignores identifiers and ports; matching common-77 vectors can represent different flows and labels, so it is a feature-support overlap statistic, not an exact-row duplicate count.",
            ],
        },
        "capture_lineage": {
            "timestamp_alignment_offset_minutes": capture["timestamp_alignment"]["best_csv_to_pcap_offset_minutes"],
            "timestamp_interval_match_fraction": capture["timestamp_alignment"]["fraction_inside_any_capture_interval"],
            "capture_category_match_fraction": capture["timestamp_alignment"]["category_match_fraction_of_valid_rows"],
            "normal_rows": int(normal.sum()),
            "normal_rows_inside_benign_capture_interval": int((normal & normal_interval).sum()),
            "normal_capture_file": "benign-dec.pcap",
            "normal_capture_count": 1,
            "caveat": "The only normal capture is one source session, so IoTID20 cannot provide an independent capture-level benign holdout while retaining benign training data.",
        },
        "limits": [
            "Exact duplicate removal is an established IoTID20 preprocessing result, not a novel contribution by this project.",
            "The original random split evaluates interpolation among related published rows; it is not independent-capture or temporal confirmation.",
            "The sample's common-77 overlap includes nonidentical flows and must not be described as exact duplicates.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "exact_duplicate_rows_removed": exact_duplicate_extras_all86,
        "input_only_duplicate_rows_removed": exact_duplicate_extras_input83,
        "notebook_exact_test_overlap": all83_overlap["test_rows_with_exact_feature_signature_in_train"],
        "notebook_common77_test_overlap": common77_overlap["test_rows_with_exact_feature_signature_in_train"],
        "common77_five_class_conflicts": common77_overlap["overlapping_test_rows_with_conflicting_five_class_label"],
        "common77_binary_conflicts": common77_overlap["overlapping_test_rows_with_conflicting_binary_label"],
        "normal_rows_inside_benign_capture": int((normal & normal_interval).sum()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
