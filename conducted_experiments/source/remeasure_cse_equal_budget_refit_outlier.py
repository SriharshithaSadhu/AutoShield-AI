"""Isolated repeat timing for the 266-second periodic-refit outlier.

Reconstructs the exact initial core and three-chunk buffer used immediately
after stream chunk 99, then repeats the same full-refit operation on fresh
copies of the initial model. This diagnoses whether the single recorded time
is repeatable on this host; it cannot explain historical background load.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import drift_adaptation_experiment as exp

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cse_replay_chunk_cache_v2"
SOURCE_RESULT = ROOT / "results" / "cse_equal_update_budget_seed42.json"
OUT = ROOT / "results" / "cse_equal_budget_refit_outlier_recheck.json"
REPEAT_COUNT = 5
SEED = 42
TARGET_STREAM_CHUNK = 99


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    features = manifest["features"]
    paths = sorted(exp.RAW_DIR.glob("*_TrafficForML_CICFlowMeter.csv"), key=lambda p: exp.capture_date(p.name))
    initial_path = next(p for p in paths if exp.capture_date(p.name) == exp.INITIAL_DATE)
    initial, _ = exp.read_sorted_capture(initial_path, features)
    split = int(len(initial) * exp.INITIAL_TRAIN_FRACTION)
    core_x, core_y = exp.stratified_cap(
        initial.iloc[:split], features, exp.MAX_INITIAL_TRAIN_ROWS, SEED,
    )
    initial_model = exp.fit_reference(core_x, core_y, SEED)

    # The target update after stream chunk 99 uses zero-based cache chunks 96–98.
    target_meta = manifest["chunks"][TARGET_STREAM_CHUNK - 1]
    buffer_indices = list(range(TARGET_STREAM_CHUNK - 3, TARGET_STREAM_CHUNK))
    recent = deque(maxlen=3)
    hashes = []
    for index in buffer_indices:
        meta = manifest["chunks"][index]
        path = CACHE / meta["file"]
        hashes.append({"file": meta["file"], "expected_sha256": meta["sha256"], "actual_sha256": sha256(path)})
        if hashes[-1]["expected_sha256"] != hashes[-1]["actual_sha256"]:
            raise ValueError(f"Cache chunk checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as row:
            recent.append((pd.DataFrame(row["x"], columns=features), row["y"].astype(np.int8)))

    durations = []
    for repeat in range(REPEAT_COUNT):
        model = exp.clone_model(initial_model)
        durations.append(exp.refit(model, core_x, core_y, recent))
        print(f"repeat {repeat + 1}/{REPEAT_COUNT}: {durations[-1]:.3f} s", flush=True)

    source = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    outlier = next(
        r for r in source["summaries"]["periodic_24"]["chunk_records"]
        if r["stream_chunk"] == TARGET_STREAM_CHUNK and r["model_updated_after_chunk"]
    )
    result = {
        "status": "complete", "date": "2026-11-02",
        "question": "Is the periodic schedule's 266-second refit after chunk 99 repeatable under an isolated reconstruction of its update inputs?",
        "replayed_update": {
            "policy": "periodic_24", "stream_chunk": TARGET_STREAM_CHUNK,
            "capture_file": target_meta["capture_file"],
            "recorded_fit_seconds": outlier["refit_seconds"],
            "buffer_global_chunks_zero_based": buffer_indices,
            "buffer_rows": [manifest["chunks"][i]["rows"] for i in buffer_indices],
            "buffer_hashes": hashes,
        },
        "repeat_count": REPEAT_COUNT,
        "isolated_refit_seconds": durations,
        "median_seconds": float(np.median(durations)),
        "max_seconds": float(max(durations)),
        "interpretation_limits": [
            "The isolated repeats use reconstructed update inputs and a fresh initial model each time; they do not reproduce any transient system load from the original replay.",
            "A non-repeated outlier suggests timing contamination or transient host effects but does not identify the cause.",
            "CPU timing results remain host-specific and are not deployment guarantees.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "recorded_seconds": outlier["refit_seconds"],
                      "median_repeat_seconds": result["median_seconds"], "max_repeat_seconds": result["max_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
