"""Independent chunked verification of the end-time-ordered NF-ToN stream."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nf_ton_iot_v3"
OUTPUT = DATA / "NF-ToN-IoT-v3_endtime_ordered.csv"
MANIFEST = ROOT / "results" / "nf_ton_iot_v3_endtime_ordered_manifest.json"
VERIFY = ROOT / "results" / "nf_ton_iot_v3_endtime_ordered_verification.json"

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
with OUTPUT.open("rb") as output_handle:
    digest = hashlib.sha256()
    while block := output_handle.read(8 * 1024 * 1024):
        digest.update(block)
    sha = digest.hexdigest()
rows = bad_times = negative_intervals = end_inversions = 0
label_counts: dict[str, int] = {}
previous_end = None
equal_end_group = 0
maximum_equal_end_group = 0

for chunk in pd.read_csv(
    OUTPUT,
    usecols=["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Label"],
    chunksize=1_000_000,
    low_memory=False,
):
    starts = pd.to_numeric(chunk["FLOW_START_MILLISECONDS"], errors="coerce")
    ends = pd.to_numeric(chunk["FLOW_END_MILLISECONDS"], errors="coerce")
    labels = pd.to_numeric(chunk["Label"], errors="coerce")
    bad_times += int((starts.isna() | ends.isna() | ~labels.isin([0, 1])).sum())
    negative_intervals += int((ends < starts).sum())
    end_values = ends.astype("int64").to_numpy()
    if len(end_values):
        if previous_end is not None and int(end_values[0]) < previous_end:
            end_inversions += 1
        end_inversions += int((end_values[1:] < end_values[:-1]).sum())
        for end in end_values:
            value = int(end)
            if value == previous_end:
                equal_end_group += 1
            else:
                maximum_equal_end_group = max(maximum_equal_end_group, equal_end_group)
                equal_end_group = 1
            previous_end = value
    rows += len(chunk)
    for label, count in labels.value_counts(dropna=False).items():
        key = str(int(label)) if pd.notna(label) else "<MISSING>"
        label_counts[key] = label_counts.get(key, 0) + int(count)
    print(f"verified_rows={rows:,} end_inversions={end_inversions:,}", flush=True)
maximum_equal_end_group = max(maximum_equal_end_group, equal_end_group)

verification = {
    "output": OUTPUT.relative_to(ROOT).as_posix(),
    "rows": rows,
    "label_counts": label_counts,
    "invalid_time_or_label_rows": bad_times,
    "negative_end_minus_start_rows": negative_intervals,
    "end_time_inversions": end_inversions,
    "max_equal_end_timestamp_group_rows": maximum_equal_end_group,
    "sha256": sha,
    "hash_matches_generation_manifest": sha == manifest["output_sha256"],
    "row_count_matches_generation_manifest": rows == manifest["rows"],
    "label_counts_match_generation_manifest": label_counts == manifest["output_label_counts"],
    "ordering_and_time_invariants_pass": bad_times == 0 and negative_intervals == 0 and end_inversions == 0,
}
VERIFY.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
print(json.dumps(verification, indent=2))
if not all((verification["hash_matches_generation_manifest"],
            verification["row_count_matches_generation_manifest"],
            verification["label_counts_match_generation_manifest"],
            verification["ordering_and_time_invariants_pass"])):
    raise SystemExit("Independent end-time stream verification failed")
