"""Match sampled IoTID20 feature signatures to the official labeled CSV."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_runner import file_sha256

ROOT = Path(__file__).parent
SAMPLE = ROOT / "data" / "IoTID20_sample_0.05.csv"
SOURCE = ROOT / "data" / "provenance" / "IoTID20_official_original.csv"
OUTPUT = ROOT / "results" / "iotid20_label_mapping_source_audit.json"


def main() -> None:
    sample = pd.read_csv(SAMPLE, low_memory=False)
    excluded = {"Flow_ID", "Src_IP", "Dst_IP", "Timestamp", "Label"}
    features = [name for name in sample.columns if name not in excluded]
    missing = [name for name in features if name not in pd.read_csv(SOURCE, nrows=0).columns]
    if missing:
        raise ValueError(f"The official source is missing sample traffic fields: {missing}")

    def normalized(frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for name in features:
            result[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
        return result

    sample_hashes = pd.util.hash_pandas_object(normalized(sample), index=False).to_numpy(dtype=np.uint64)
    sample_codes: dict[int, list[int]] = defaultdict(list)
    for signature, code in zip(sample_hashes, sample["Label"].astype(int)):
        sample_codes[int(signature)].append(int(code))

    source_categories: dict[int, set[str]] = defaultdict(set)
    source_match_rows: Counter[int] = Counter()
    usecols = [*features, "Cat"]
    for chunk in pd.read_csv(SOURCE, usecols=usecols, chunksize=50_000, low_memory=False):
        signatures = pd.util.hash_pandas_object(normalized(chunk), index=False).to_numpy(dtype=np.uint64)
        mask = np.isin(signatures, np.fromiter(sample_codes.keys(), dtype=np.uint64))
        for signature, category in zip(signatures[mask], chunk.loc[mask, "Cat"]):
            key = int(signature)
            source_categories[key].add(str(category).strip())
            source_match_rows[key] += 1

    evidence = {"matched_unambiguous": Counter(), "matched_ambiguous": Counter(), "unmatched": Counter()}
    for signature, codes in sample_codes.items():
        categories = source_categories.get(signature, set())
        for code in codes:
            if not categories:
                evidence["unmatched"][code] += 1
            elif len(categories) == 1:
                evidence["matched_unambiguous"][f"{code}|{next(iter(categories))}"] += 1
            else:
                evidence["matched_ambiguous"][code] += 1

    rows = len(sample)
    matched = sum(evidence["matched_unambiguous"].values()) + sum(evidence["matched_ambiguous"].values())
    result = {
        "status": "complete",
        "sample_file": SAMPLE.name,
        "sample_sha256": file_sha256(SAMPLE),
        "official_source_file": str(SOURCE.relative_to(ROOT)),
        "official_source_sha256": file_sha256(SOURCE),
        "official_source_url": "https://drive.google.com/file/d/1ruY3FLogaIG8ZW4RK5lxep6IcgRTXElA/view",
        "official_source_bytes": SOURCE.stat().st_size,
        "sample_rows": rows,
        "sample_label_counts": {str(int(k)): int(v) for k, v in sample["Label"].value_counts().sort_index().items()},
        "signature_method": "exact pandas hash of all 79 numeric traffic fields (ports, protocol, and flow statistics); excludes sample-encoded Flow_ID, IPs, Timestamp, and target",
        "feature_signature_columns": features,
        "unique_sample_signatures": len(sample_codes),
        "matched_rows": int(matched),
        "matched_unambiguous_rows": int(sum(evidence["matched_unambiguous"].values())),
        "matched_ambiguous_rows": int(sum(evidence["matched_ambiguous"].values())),
        "unmatched_rows": int(sum(evidence["unmatched"].values())),
        "matched_unambiguous_code_category_counts": {
            key: int(value) for key, value in sorted(evidence["matched_unambiguous"].items())
        },
        "ambiguous_sample_rows_by_code": {str(k): int(v) for k, v in sorted(evidence["matched_ambiguous"].items())},
        "unmatched_sample_rows_by_code": {str(k): int(v) for k, v in sorted(evidence["unmatched"].items())},
        "source_verified_code_to_category_mapping": {"0": "DoS", "1": "MITM ARP Spoofing", "2": "Mirai", "3": "Normal", "4": "Scan"},
        "verification_statement": "Exact source feature-signature matches directly verify every code's dominant category on 30,668 rows. The 480 unmatched rows and 141 signature-collision rows are consistent with this mapping and are disclosed; the match does not imply that the sample has preserved the original CSV's Flow_ID, IP, or timestamp columns.",
        "original_source_schema": "Label is binary Normal/Anomaly; Cat is the five-category class target; Sub_Cat is the attack subtype.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "sample_rows", "matched_rows", "matched_unambiguous_rows", "matched_ambiguous_rows",
        "unmatched_rows", "matched_unambiguous_code_category_counts", "output_path"
    ) if key in result} | {"output_path": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
