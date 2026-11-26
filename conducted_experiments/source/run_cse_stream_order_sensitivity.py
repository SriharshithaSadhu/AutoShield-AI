"""Paired stream-order sensitivity study on fixed CSE replay chunks.

The target records and chunk membership are materialized once. Treatments
change only the order of intact chunks within each capture day. This is a
counterfactual stress replay, not a claim of chronological deployment.
"""
from __future__ import annotations

from collections import deque
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import joblib
import numpy as np
import pandas as pd

import drift_adaptation_experiment as exp
from experiment_protocol import make_leakage_safe_pipeline
from research_runner import file_sha256, make_estimator
from run_rccda_inspired_cse import binary_log_loss, rccda_decision


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cse_replay_chunk_cache_v2"
RESULTS = ROOT / "results"
CHUNK_ROWS = 20_000
MAX_INITIAL_ROWS = 100_000
RECENT_CHUNKS = 3
UPDATE_BUDGET_RATE = 0.10
RCCDA_V = 10.0
RCCDA_KP = 1.0
RCCDA_KD = 0.1
RANDOM_ACTION_SEED = 20261101


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def json_sha256(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(blob)


def replay_paths() -> tuple[list[Path], Path]:
    paths = sorted(exp.RAW_DIR.glob("*_TrafficForML_CICFlowMeter.csv"),
                   key=lambda p: exp.capture_date(p.name))
    initial = [p for p in paths if exp.capture_date(p.name) == exp.INITIAL_DATE]
    later = [p for p in paths if p not in initial]
    if len(paths) != 10 or len(initial) != 1 or len(later) != 9:
        raise ValueError(f"Expected 10 captures (1 source + 9 replay); got {len(paths)}")
    return later, initial[0]


def source_hashes(paths: list[Path], initial_path: Path) -> dict[str, str]:
    return {str(p.relative_to(ROOT)): file_sha256(p)
            for p in [initial_path, *paths]}


def audit_manifest_timestamps(manifest: dict) -> bool:
    previous_max: dict[str, pd.Timestamp] = {}
    for item in manifest["chunks"]:
        lo = pd.to_datetime(item["timestamp_min_ns"], unit="ns")
        hi = pd.to_datetime(item["timestamp_max_ns"], unit="ns")
        if lo.date().isoformat() != item["capture_date"] or hi.date().isoformat() != item["capture_date"]:
            raise ValueError(f"Chunk timestamps do not match capture day: {item['file']}")
        prior = previous_max.get(item["capture_file"])
        if prior is not None and lo < prior:
            raise ValueError(f"Chunk order is not chronological within capture: {item['file']}")
        previous_max[item["capture_file"]] = hi
    return True


def make_cache(paths: list[Path], initial_path: Path) -> dict:
    features = exp.feature_columns(initial_path)
    src_hashes = source_hashes(paths, initial_path)
    if CACHE.exists():
        manifest_path = CACHE / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Chunk cache exists without manifest: {CACHE}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_version") != 2 or manifest.get("timestamp_storage_unit") != "ns":
            raise RuntimeError("Cached timestamps are not normalized nanoseconds; build cache v2 first")
        if json_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
            raise RuntimeError("Cached manifest checksum mismatch")
        if manifest["source_sha256"] != src_hashes or manifest["features"] != features:
            raise RuntimeError("Cached chunks do not match current source files/features")
        for chunk in manifest["chunks"]:
            chunk_path = CACHE / chunk["file"]
            if not chunk_path.is_file() or file_sha256(chunk_path) != chunk["sha256"]:
                raise RuntimeError(f"Chunk cache checksum mismatch: {chunk_path}")
        audit_manifest_timestamps(manifest)
        return manifest

    temp = CACHE.with_name(CACHE.name + ".building")
    if temp.exists():
        raise RuntimeError(f"Incomplete cache directory exists; inspect before removal: {temp}")
    temp.mkdir(parents=True)
    records = []
    capture_exclusions = []
    index = 0
    try:
        for path in paths:
            print(f"Materializing fixed chunks: {path.name}", flush=True)
            chunks, exclusions = exp.capture_chunks(path, features, CHUNK_ROWS)
            capture_exclusions.append({"file": path.name, **exclusions})
            for chunk in chunks:
                x = chunk[features].to_numpy(dtype=np.float64, copy=True)
                y = chunk["target"].to_numpy(dtype=np.int8, copy=True)
                # DuckDB returns datetime64[us] for the oversized 20-Feb file;
                # normalize before integer conversion so all cache chunks use ns.
                timestamps = chunk["Timestamp"].astype("datetime64[ns]").astype("int64").to_numpy()
                out_name = f"chunk_{index:06d}.npz"
                out_path = temp / out_name
                # Uncompressed NPZ avoids lossy type conversion and allows
                # bounded, one-chunk-at-a-time materialization and replay.
                np.savez(out_path, x=x, y=y, timestamps_ns=timestamps)
                records.append({
                    "global_chunk": index,
                    "capture_file": path.name,
                    "capture_date": exp.capture_date(path.name).date().isoformat(),
                    "file": out_name,
                    "sha256": file_sha256(out_path),
                    "rows": int(len(y)),
                    "benign": int((y == 0).sum()),
                    "attack": int((y == 1).sum()),
                    "timestamp_min_ns": int(timestamps.min()),
                    "timestamp_max_ns": int(timestamps.max()),
                })
                index += 1
        manifest = {
            "cache_version": 2,
            "purpose": "immutable same-row, same-chunk stream-order counterfactual replay",
            "source_sha256": src_hashes,
            "feature_count": len(features),
            "features": features,
            "chunk_rows_target": CHUNK_ROWS,
            "equal_timestamps_kept_together": True,
            "timestamp_storage_unit": "ns",
            "capture_exclusions": capture_exclusions,
            "chunks": records,
        }
        manifest["manifest_sha256"] = json_sha256(manifest)
        audit_manifest_timestamps(manifest)
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                             encoding="utf-8")
        temp.rename(CACHE)
        return manifest
    except Exception:
        # Keep the building directory for diagnosis; never silently delete data.
        raise


def schedule_chunks(manifest: dict, order: str, seed: int | None) -> list[int]:
    by_day: dict[str, list[int]] = {}
    for row in manifest["chunks"]:
        by_day.setdefault(row["capture_date"], []).append(row["global_chunk"])
    schedule = []
    rng = random.Random(seed) if seed is not None else None
    for day in sorted(by_day):
        day_ids = by_day[day].copy()
        if order == "within_day_permutation":
            assert rng is not None
            rng.shuffle(day_ids)
        schedule.extend(day_ids)
    return schedule


def load_chunk(manifest: dict, chunk_id: int) -> pd.DataFrame:
    item = manifest["chunks"][chunk_id]
    with np.load(CACHE / item["file"], allow_pickle=False) as arrays:
        data = {name: arrays["x"][:, i]
                for i, name in enumerate(manifest["features"])}
        data["target"] = arrays["y"]
        data["Timestamp"] = pd.to_datetime(arrays["timestamps_ns"], unit="ns")
    frame = pd.DataFrame(data)
    if len(frame) != item["rows"]:
        raise AssertionError(f"Chunk row-count mismatch for id={chunk_id}")
    return frame


def model_summary(records: list[dict]) -> dict:
    counts = {k: sum(int(r[k]) for r in records) for k in ("tn", "fp", "fn", "tp")}
    tn, fp, fn, tp = (counts[k] for k in ("tn", "fp", "fn", "tp"))
    attack_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    benign_f1 = 2 * tn / max(1, 2 * tn + fp + fn)
    n = tn + fp + fn + tp
    return {
        "chunks": len(records), "rows": n, "confusion": counts,
        "pooled_macro_f1": (attack_f1 + benign_f1) / 2,
        "pooled_weighted_f1": ((tp + fn) * attack_f1 + (tn + fp) * benign_f1) / max(1, n),
        "pooled_attack_recall": tp / max(1, tp + fn),
        "pooled_benign_fpr": fp / max(1, fp + tn),
        "mean_chunk_macro_f1": float(np.mean([r["macro_f1"] for r in records])),
        "mean_chunk_fpr": float(np.mean([r["false_positive_rate"] for r in records])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("chronological", "within_day_permutation"), required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="required for within_day_permutation; omitted for chronological")
    args = parser.parse_args()
    if args.order == "chronological" and args.seed is not None:
        parser.error("--seed is only valid with within_day_permutation")
    if args.order == "within_day_permutation" and args.seed is None:
        parser.error("--seed is required for within_day_permutation")

    later_paths, initial_path = replay_paths()
    manifest = make_cache(later_paths, initial_path)
    timestamps_valid = audit_manifest_timestamps(manifest)
    features = manifest["features"]
    initial, exclusions = exp.read_sorted_capture(initial_path, features)
    split = int(len(initial) * exp.INITIAL_TRAIN_FRACTION)
    core_x, core_y = exp.stratified_cap(
        initial.iloc[:split], features, MAX_INITIAL_ROWS, exp.SEED
    )
    if len(np.unique(core_y)) != 2:
        raise ValueError("Initial training partition must contain both classes")
    initial_model = exp.fit_reference(core_x, core_y, exp.SEED)
    del initial

    states = {
        "static": {"model": exp.clone_model(initial_model), "records": [], "updates": [],
                   "queue": 0.0, "loss_history": [], "previous_loss": None},
        "rccda_paper_estimator_port": {"model": exp.clone_model(initial_model), "records": [], "updates": [],
                   "queue": 0.0, "loss_history": [], "previous_loss": None},
        "periodic_10pct": {"model": exp.clone_model(initial_model), "records": [], "updates": [],
                   "queue": 0.0, "loss_history": [], "previous_loss": None},
    }
    order_seed = args.seed
    schedule = schedule_chunks(manifest, args.order, order_seed)
    schedule_payload = {"order": args.order, "seed": order_seed, "chunk_ids": schedule}
    schedule_hash = json_sha256(schedule_payload)
    recent: deque[tuple[pd.DataFrame, np.ndarray]] = deque(maxlen=RECENT_CHUNKS)
    rng_update = random.Random(RANDOM_ACTION_SEED)
    budget = UPDATE_BUDGET_RATE
    trace = []
    start = time.perf_counter()

    for stream_pos, chunk_id in enumerate(schedule, start=1):
        item = manifest["chunks"][chunk_id]
        chunk = load_chunk(manifest, chunk_id)
        y = chunk["target"].to_numpy(dtype=np.int8)
        x = chunk[features].copy()
        actions = {}
        for name, state in states.items():
            metric = exp.evaluate_chunk(state["model"], chunk, features, decision_threshold=0.5)
            p = metric.pop("_positive_probabilities")
            loss = binary_log_loss(y, p)
            update = False
            if name == "rccda_paper_estimator_port":
                best = min(state["loss_history"], default=loss)
                update, gain, q_next = rccda_decision(
                    loss, state["previous_loss"], best, state["queue"],
                    cost=1.0, mean_cost_budget=budget,
                    v=RCCDA_V, kp=RCCDA_KP, kd=RCCDA_KD,
                )
                state["queue"] = q_next
                state["loss_history"].append(loss)
                state["previous_loss"] = loss
                metric.update({"estimated_gain": gain, "queue_after": q_next})
            elif name == "periodic_10pct":
                update = stream_pos % round(1 / budget) == 0
            metric.update({
                "stream_position": stream_pos,
                "chunk_id": chunk_id,
                "capture_file": item["capture_file"],
                "update_requested": bool(update),
                "log_loss": loss,
            })
            state["records"].append(metric)
            actions[name] = bool(update)

        # Label-release event: only after all arms predicted the whole chunk.
        recent.append((x, y.copy()))
        for name, state in states.items():
            if not actions[name] or name == "static":
                continue
            fit_start = time.perf_counter()
            elapsed = exp.refit(state["model"], core_x, core_y, recent)
            state["updates"].append({
                "after_stream_position": stream_pos,
                "after_chunk_id": chunk_id,
                "labels_released_for_chunk_id": chunk_id,
                "fit_seconds_reported_by_refit": float(elapsed),
                "training_rows": int(len(core_y) + sum(len(bx) for bx, _ in recent)),
            })
        trace.append({"stream_position": stream_pos, "chunk_id": chunk_id,
                      "actions": actions, "labels_released_after_scoring": True})
        if stream_pos == 1 or stream_pos % 100 == 0:
            print(f"{args.order} seed={args.seed} progress {stream_pos}/{len(schedule)}",
                  flush=True)
        del chunk, x, y

    summaries = {name: model_summary(state["records"]) for name, state in states.items()}
    static_counts = summaries["static"]["confusion"]
    expected = None
    old = RESULTS / "cse_rccda_inspired_equal_update_baselines_seed42.json"
    if old.exists():
        reference = json.loads(old.read_text(encoding="utf-8"))
        expected = reference.get("summaries", {}).get("static", {}).get("confusion")
    # In either treatment, verify the static model is invariant on the exact
    # same row multiset. A supplied chronological result is an additional check.
    result = {
        "study": "CSE fixed-row stream-order sensitivity for adaptation policies",
        "status": "development pilot; counterfactual within-capture block order is not natural chronology",
        "order": args.order,
        "order_seed": args.seed,
        "random_update_action_seed": RANDOM_ACTION_SEED,
        "chunk_cache": str(CACHE.relative_to(ROOT)),
        "chunk_manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule_hash,
        "source_file_sha256": manifest["source_sha256"],
        "stream": {
            "features": features,
            "n_chunks": len(schedule),
            "n_rows": int(sum(manifest["chunks"][i]["rows"] for i in schedule)),
            "chunk_membership_fixed": True,
            "chunk_size_target": CHUNK_ROWS,
            "equal_timestamps_kept_together": True,
            "timestamp_storage_unit": "ns",
            "capture_day_order_preserved": True,
            "label_release": "after all arms score each chunk; released labels can affect only later stream positions",
        },
        "policies": {
            name: {"summary": summaries[name], "updates": state["updates"],
                   "chunk_records": state["records"]}
            for name, state in states.items()
        },
        "checks": {
            "static_model_confusion": static_counts,
            "static_matches_reference_frozen_counts": (
                None if expected is None else static_counts == expected
            ),
            "all_labels_released_after_chunk_scoring": all(
                r["labels_released_after_scoring"] for r in trace
            ),
            "schedule_is_permutation_of_fixed_chunk_ids": sorted(schedule) == list(range(len(manifest["chunks"]))),
            "timestamps_match_capture_day_and_chunk_order": timestamps_valid,
        },
        "elapsed_wall_seconds": time.perf_counter() - start,
        "limitations": [
            "One source model and one CSE collection; no independent confirmation stream.",
            "Within-day chunk permutations are counterfactual arrival schedules, not natural chronology.",
            "The RCCDA arm is an update-scheduler port to LightGBM full refits, not an exact paper reproduction or guarantee transfer.",
            "Chunk-level outcomes are dependent; flow count is not the number of independent replicates.",
        ],
    }
    if not result["checks"]["schedule_is_permutation_of_fixed_chunk_ids"]:
        raise AssertionError("Schedule lost or duplicated chunk IDs")
    if expected is not None and static_counts != expected:
        raise AssertionError("Static pooled confusion changed despite invariant target rows")
    suffix = "chronological" if args.order == "chronological" else f"within_day_seed{args.seed}"
    out = RESULTS / f"cse_stream_order_sensitivity_{suffix}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "summary": summaries,
                      "checks": result["checks"],
                      "elapsed_wall_seconds": result["elapsed_wall_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
