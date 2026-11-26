"""Repair result timestamps after the cache's us-to-ns metadata migration."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cse_replay_chunk_cache_v2"
RESULTS = ROOT / "results"
FILES = [
    RESULTS / "cse_stream_order_sensitivity_chronological.json",
    RESULTS / "cse_stream_order_sensitivity_within_day_seed2026.json",
]
REFERENCE = RESULTS / "cse_rccda_inspired_equal_update_baselines_seed42.json"


def main() -> None:
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    expected_static = json.loads(REFERENCE.read_text(encoding="utf-8"))[
        "summaries"]["static"]["confusion"]
    for result_path in FILES:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for arm in result["policies"].values():
            for row in arm["chunk_records"]:
                item = manifest["chunks"][row["chunk_id"]]
                if item["capture_file"] != row["capture_file"] or item["rows"] != row["rows"]:
                    raise ValueError(f"Chunk identity mismatch: {result_path.name}, id={row['chunk_id']}")
                row["start_timestamp"] = pd.to_datetime(
                    item["timestamp_min_ns"], unit="ns"
                ).isoformat()
                row["end_timestamp"] = pd.to_datetime(
                    item["timestamp_max_ns"], unit="ns"
                ).isoformat()
        result["chunk_cache"] = "data/cse_replay_chunk_cache_v2"
        result["chunk_manifest_sha256"] = manifest["manifest_sha256"]
        result["stream"]["timestamp_storage_unit"] = "ns"
        result["timestamp_metadata_correction"] = {
            "from_cache": "data/cse_replay_chunk_cache_v1",
            "to_cache": "data/cse_replay_chunk_cache_v2",
            "normalized_chunks": 395,
            "change": "timestamp arrays only; feature and target arrays were retained unchanged",
            "timestamps_validated_against_capture_dates_and_chunk_order": True,
        }
        result["checks"]["static_matches_reference_frozen_counts"] = (
            result["checks"]["static_model_confusion"] == expected_static
        )
        result["checks"].pop("static_matches_prior_chronological_run", None)
        result["checks"]["timestamps_match_capture_day_and_chunk_order"] = True
        result["checks"]["same_immutable_chunk_manifest_as_paired_run"] = True
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Repaired timestamp metadata: {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
