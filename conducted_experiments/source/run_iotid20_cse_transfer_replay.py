"""Chronological CSE replay of fixed models trained on IoTID20.

This carries the source baseline and feasible MOPSO-selected model across the
same ordered CSE stream. Periodic refits happen only after chunk labels have
been revealed and affect the next chunk. There is no search on CSE.
"""
from __future__ import annotations

from collections import deque
import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from drift_adaptation_experiment import (
    CHUNK_ROWS,
    RAW_DIR,
    capture_date,
    capture_chunks,
    evaluate_chunk,
    feature_columns,
)
from research_runner import file_sha256, make_estimator
from experiment_protocol import make_leakage_safe_pipeline

ROOT = Path(__file__).parent
PREPARED = ROOT / "data" / "iotid20_binary_common77_dev.csv"
SEARCH = ROOT / "results" / "iotid20_binary_common77_mopso_seed42.json"
MODELS = ROOT / "results" / "end_to_end_models"
OUTPUT = ROOT / "results" / "iotid20_cse_transfer_replay_seed42.json"
UPDATE_EVERY = 5
RECENT_CHUNKS = 3


def summary(rows: list[dict]) -> dict:
    counts = {key: sum(int(row[key]) for row in rows) for key in ("tn", "fp", "fn", "tp")}
    positive_f1 = 2 * counts["tp"] / max(1, 2 * counts["tp"] + counts["fp"] + counts["fn"])
    negative_f1 = 2 * counts["tn"] / max(1, 2 * counts["tn"] + counts["fp"] + counts["fn"])
    total = max(1, counts["tn"] + counts["fp"] + counts["fn"] + counts["tp"])
    return {
        "chunks": len(rows),
        "rows": sum(row["rows"] for row in rows),
        "confusion_counts": counts,
        "weighted_f1_from_pooled_confusion": float(
            ((counts["tp"] + counts["fn"]) * positive_f1
             + (counts["tn"] + counts["fp"]) * negative_f1) / total
        ),
        "macro_f1_from_pooled_confusion": float((positive_f1 + negative_f1) / 2),
        "false_positive_rate": float(counts["fp"] / max(1, counts["fp"] + counts["tn"])),
        "mean_chunk_weighted_f1": float(np.mean([row["weighted_f1"] for row in rows])),
        "mean_chunk_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
        "mean_chunk_fpr": float(np.mean([row["false_positive_rate"] for row in rows])),
        "p95_prediction_latency_ms_per_row": float(np.percentile(
            [row["prediction_latency_ms_per_row"] for row in rows], 95)),
    }


def main() -> None:
    source = pd.read_csv(PREPARED)
    mapping_audit = json.loads((ROOT / "results" / "iotid20_label_mapping_source_audit.json").read_text(encoding="utf-8"))
    search = json.loads(SEARCH.read_text(encoding="utf-8"))
    archive_feasible = [r for r in search["pareto_archive"] if r["feasibility"]["feasible"]]
    selected = max(archive_feasible, key=lambda row: row["metrics"]["weighted_f1"])
    configs = {
        "reference_default": {
            "n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31,
            "class_weight": "balanced", "verbosity": -1,
        },
        "mopso_best_feasible": selected["config"],
    }
    paths = sorted(RAW_DIR.glob("*_TrafficForML_CICFlowMeter.csv"),
                   key=lambda path: capture_date(path.name))
    if len(paths) != 10:
        raise ValueError(f"Expected all ten CSE captures; found {len(paths)}")
    features = feature_columns(paths[0])
    models = {}
    for name in configs:
        path = MODELS / f"{name}.joblib"
        if not path.exists():
            model = make_leakage_safe_pipeline(make_estimator("lightgbm", configs[name], 42, 1), random_state=42)
            model.fit(source[features], source["is_attack"].to_numpy(dtype=np.int8))
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path, compress=0)
        models[f"{name}_no_adaptation"] = joblib.load(path)
        models[f"{name}_periodic_refit"] = joblib.load(path)

    records = {name: [] for name in models}
    recent: deque[tuple[pd.DataFrame, np.ndarray]] = deque(maxlen=RECENT_CHUNKS)
    stream_index = 0
    exclusions = []
    updates = {name: [] for name in configs}
    started = time.perf_counter()
    for path in paths:
        chunks, day_exclusions = capture_chunks(path, features, CHUNK_ROWS)
        exclusions.append({"file": path.name, **day_exclusions})
        for chunk in chunks:
            stream_index += 1
            y = chunk["target"].to_numpy(dtype=np.int8)
            for name, model in models.items():
                metric = evaluate_chunk(model, chunk, features)
                metric.pop("_positive_probabilities", None)
                metric.update({"stream_chunk": stream_index, "capture_file": path.name})
                records[name].append(metric)
            # The current labels enter the replay buffer only after all four
            # arms have produced predictions for this chunk.
            recent.append((chunk[features].copy(), y.copy()))
            if stream_index % UPDATE_EVERY == 0:
                recent_x = pd.concat([part[0] for part in recent], ignore_index=True)
                recent_y = np.concatenate([part[1] for part in recent])
                update_x = pd.concat([source[features], recent_x], ignore_index=True)
                update_y = np.concatenate([source["is_attack"].to_numpy(dtype=np.int8), recent_y])
                for name in configs:
                    arm = f"{name}_periodic_refit"
                    fit_start = time.perf_counter()
                    models[arm].fit(update_x, update_y)
                    updates[name].append({
                        "after_chunk": stream_index,
                        "labels_released_through_chunk": stream_index,
                        "fit_seconds": time.perf_counter() - fit_start,
                        "training_rows": int(len(update_y)),
                    })
                del recent_x, recent_y, update_x, update_y
            if stream_index == 1 or stream_index % 25 == 0:
                print(f"Replay progress: {stream_index} chunks; {path.name}", flush=True)
            if stream_index % 50 == 0:
                interim = {
                    "status": "running",
                    "completed_chunks": stream_index,
                    "elapsed_seconds": time.perf_counter() - started,
                    "records": records,
                    "updates": updates,
                    "timestamp_exclusions": exclusions,
                }
                OUTPUT.with_suffix(".partial.json").write_text(json.dumps(interim, indent=2), encoding="utf-8")
        gc.collect()

    result = {
        "status": "complete",
        "experiment": "IoTID20-trained binary model chronological CSE-CIC-IDS2018 replay",
        "development_only": True,
        "seed": 42,
        "cse_raw_files": [{"file": path.name, "sha256": file_sha256(path)} for path in paths],
        "rows": sum(row["rows"] for row in records["reference_default_no_adaptation"]),
        "feature_count": len(features),
        "features": features,
        "source_dataset_sha256": file_sha256(ROOT / "data" / "IoTID20_sample_0.05.csv"),
        "source_prepared_sha256": file_sha256(PREPARED),
        "target_mapping": "IoTID20 code 3 source-verified Normal/benign; all other codes attacks; CSE Label Benign vs non-Benign",
        "target_mapping_evidence": {
            "official_source_sha256": mapping_audit["official_source_sha256"],
            "unambiguous_matched_rows": mapping_audit["matched_unambiguous_rows"],
            "ambiguous_signature_rows": mapping_audit["matched_ambiguous_rows"],
            "unmatched_rows": mapping_audit["unmatched_rows"],
            "source_match_audit": "results/iotid20_label_mapping_source_audit.json",
        },
        "target_mapping_caveat": "621 sample rows were ambiguous or unmatched by exact numeric signatures; all unambiguous matches verify the same five-code category mapping.",
        "chunk_rows_target": CHUNK_ROWS,
        "equal_timestamp_rows_kept_together": True,
        "adaptation": {
            "type": "periodic supervised refit",
            "every_chunks": UPDATE_EVERY,
            "recent_labeled_chunks": RECENT_CHUNKS,
            "training_data": "all IoTID20 source training rows plus the most recent released CSE chunk labels",
            "labels_used_for_current_chunk_predictions": False,
        },
        "models": {
            name: {"config": configs[name], "source_holdout": json.loads(
                (ROOT / "results" / "iotid20_binary_common77_end_to_end_seed42.json").read_text(encoding="utf-8")
            )["same_grouped_holdout_comparison"][name]["holdout_metrics"],
              "replay_no_adaptation": summary(records[f"{name}_no_adaptation"]),
              "replay_periodic_refit": summary(records[f"{name}_periodic_refit"]),
              "periodic_updates": updates[name]}
            for name in configs
        },
        "timestamp_exclusions": exclusions,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
        "interpretation": "Compare measured performance as observed; the optimizer is not presumed superior. This is development evidence because the CSE schedule has been used in prior experiments.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    partial_path = OUTPUT.with_suffix(".partial.json")
    partial_path.unlink(missing_ok=True)
    print(json.dumps({
        "status": result["status"], "rows": result["rows"],
        "models": {name: {
            "no_adaptation": info["replay_no_adaptation"],
            "periodic_refit": info["replay_periodic_refit"],
        } for name, info in result["models"].items()},
        "elapsed_seconds": result["elapsed_seconds"], "output": str(OUTPUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

