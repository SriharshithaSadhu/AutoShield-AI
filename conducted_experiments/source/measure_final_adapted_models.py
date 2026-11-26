"""Reconstruct the final chunk-805 periodic checkpoints and measure their size.

This uses the same source rows, final three released CSE chunks, feature
pipeline, and fixed candidate configurations recorded by the replay.
"""
from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from drift_adaptation_experiment import RAW_DIR, capture_chunks, capture_date, feature_columns
from experiment_protocol import make_leakage_safe_pipeline
from research_runner import make_estimator, file_sha256

ROOT = Path(__file__).parent
REPLAY_PATH = ROOT / "results" / "iotid20_cse_transfer_replay_seed42.json"
SOURCE_PATH = ROOT / "data" / "iotid20_binary_common77_dev.csv"
MODEL_DIR = ROOT / "results" / "end_to_end_models"


def main() -> None:
    result = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    source = pd.read_csv(SOURCE_PATH)
    captures = sorted(RAW_DIR.glob("*_TrafficForML_CICFlowMeter.csv"), key=lambda p: capture_date(p.name))
    final_capture = captures[-1]
    features = feature_columns(final_capture)
    expected_last_three = result["records"]["reference_default_no_adaptation"][-4:-1]
    expected_final = result["records"]["reference_default_no_adaptation"][-1]
    assert [r["stream_chunk"] for r in expected_last_three] == [803, 804, 805]
    assert expected_final["stream_chunk"] == 806
    chunks, exclusions = capture_chunks(final_capture, features, result["chunk_rows_target"])
    tail = []
    for chunk in chunks:
        tail.append(chunk)
        if len(tail) > 4:
            tail.pop(0)
    if len(tail) < 4:
        raise ValueError("Expected at least four chronological chunks in the final capture")
    update_chunks = tail[-4:-1]
    measured = [
        (int(len(chunk)), chunk["Timestamp"].iloc[0].isoformat(), chunk["Timestamp"].iloc[-1].isoformat())
        for chunk in update_chunks
    ]
    expected = [(int(r["rows"]), r["start_timestamp"], r["end_timestamp"]) for r in expected_last_three]
    if measured != expected:
        raise ValueError(f"Reconstructed labeled windows differ from replay: {measured} != {expected}")
    recent_x = pd.concat([chunk[features] for chunk in update_chunks], ignore_index=True)
    recent_y = np.concatenate([chunk["target"].to_numpy(dtype=np.int8) for chunk in update_chunks])
    X = pd.concat([source[features], recent_x], ignore_index=True)
    y = np.concatenate([source["is_attack"].to_numpy(dtype=np.int8), recent_y])
    for name in ("reference_default", "mopso_best_feasible"):
        config = result["models"][name]["config"]
        model = make_leakage_safe_pipeline(make_estimator("lightgbm", config, 42, 1), random_state=42)
        model.fit(X, y)
        output_path = MODEL_DIR / f"{name}_adapted_after_chunk805.joblib"
        joblib.dump(model, output_path, compress=0)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        result["models"][name]["final_adapted_model"] = {
            "after_chunk": 805,
            "training_rows": int(len(y)),
            "sha256": file_sha256(output_path),
            "serialized_size_mb": size_mb,
            "size_cap_mb": 1.5,
            "within_size_cap": bool(size_mb <= 1.5),
            "path": str(output_path.relative_to(ROOT)),
        }
    result["final_checkpoint_reconstruction"] = {
        "verified_labeled_chunk_indices": [803, 804, 805],
        "next_chunk_index": 806,
        "training_rows": int(len(y)),
        "final_capture": final_capture.name,
        "capture_timestamp_exclusions": exclusions,
        "note": "Reconstructed the model state after the logged chunk-805 refit to measure the actual adapted serialization size; chunk 806 remained future to this update.",
    }
    REPLAY_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: info["final_adapted_model"] for name, info in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
