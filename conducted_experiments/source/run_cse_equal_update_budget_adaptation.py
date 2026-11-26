"""Matched-update-budget comparison for causal CSE adaptation policies.

Compares the RCCDA-inspired loss-history scheduler with periodic and random
fixed-configuration LightGBM refits. Every adaptive policy gets exactly 24
updates on the same 753-chunk CSE replay; each chunk is scored before its
labels enter the common recent-three-chunk refit buffer. This remains a
development-only comparison on a repeatedly explored CSE stream and is not an
exact reproduction of RCCDA's SGD algorithm or guarantees.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import pandas as pd

import drift_adaptation_experiment as exp
from run_rccda_inspired_cse import binary_log_loss, rccda_decision

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "cse_equal_update_budget_seed42.json"
SEED = 42
RANDOM_SCHEDULE_SEED = 20261102
CHUNK_ROWS = 20_000
MAX_INITIAL_ROWS = 100_000
RECENT_CHUNKS = 3
UPDATE_BUDGET = 24
RCCDA_MEAN_BUDGET_RATE = 0.10
RCCDA_V, RCCDA_KP, RCCDA_KD = 10.0, 1.0, 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(records: list[dict]) -> dict:
    tn = sum(r["tn"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    tp = sum(r["tp"] for r in records)
    pos_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    neg_f1 = 2 * tn / max(1, 2 * tn + fp + fn)
    n = tn + fp + fn + tp
    return {
        "chunks": len(records), "rows": n,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "pooled_macro_f1": (pos_f1 + neg_f1) / 2,
        "pooled_weighted_f1": ((tp + fn) * pos_f1 + (tn + fp) * neg_f1) / max(1, n),
        "pooled_fpr": fp / max(1, fp + tn),
        "pooled_attack_recall": tp / max(1, tp + fn),
        "mean_chunk_macro_f1": float(np.mean([r["macro_f1"] for r in records])),
        "mean_chunk_fpr": float(np.mean([r["false_positive_rate"] for r in records])),
        "mean_chunk_log_loss": float(np.mean([r["log_loss"] for r in records])),
        "updates": sum(bool(r["model_updated_after_chunk"]) for r in records),
        "total_refit_seconds": float(sum(r.get("refit_seconds", 0.0) for r in records)),
        "chunk_records": records,
    }


def main() -> None:
    paths = sorted(exp.RAW_DIR.glob("*_TrafficForML_CICFlowMeter.csv"), key=lambda p: exp.capture_date(p.name))
    initial_paths = [p for p in paths if exp.capture_date(p.name) == exp.INITIAL_DATE]
    if len(initial_paths) != 1:
        raise ValueError(f"Expected one initial CSE capture, found {len(initial_paths)}")
    initial_path = initial_paths[0]
    features = exp.feature_columns(initial_path)
    initial, initial_exclusions = exp.read_sorted_capture(initial_path, features)
    split = int(len(initial) * exp.INITIAL_TRAIN_FRACTION)
    train = initial.iloc[:split]
    core_x, core_y = exp.stratified_cap(train, features, MAX_INITIAL_ROWS, SEED)
    initial_model = exp.fit_reference(core_x, core_y, SEED)
    del initial, train

    # Replay from the immutable timestamp-ordered cache. This avoids reparsing
    # and externally sorting the largest raw capture while preserving the exact
    # feature rows, labels, chunk boundaries, and order in the prior studies.
    cached_manifest = json.loads((ROOT / "data" / "cse_replay_chunk_cache_v2" / "manifest.json").read_text(encoding="utf-8"))
    n_expected = len(cached_manifest["chunks"])
    if n_expected != 753:
        raise ValueError(f"Expected verified 753-chunk v2 stream, found {n_expected}")
    if cached_manifest["features"] != features:
        raise ValueError("Cache feature order differs from the initial-capture feature order")
    periodic_ids = set(np.rint(np.linspace(1, n_expected, UPDATE_BUDGET)).astype(int).tolist())
    if len(periodic_ids) != UPDATE_BUDGET:
        raise ValueError("Periodic schedule did not yield the locked action count")
    rng = random.Random(RANDOM_SCHEDULE_SEED)
    random_ids = set(rng.sample(range(1, n_expected + 1), UPDATE_BUDGET))
    if len(random_ids) != UPDATE_BUDGET:
        raise ValueError("Random schedule did not yield the locked action count")

    states = {
        name: {"model": exp.clone_model(initial_model), "records": [], "updates": []}
        for name in ("static", "rccda_24", "periodic_24", "random_24")
    }
    states["rccda_24"].update({"queue": 0.0, "history": [], "previous_loss": None})
    recent: deque[tuple] = deque(maxlen=RECENT_CHUNKS)
    stream_index = 0
    exclusions = [{"file": initial_path.name, **initial_exclusions}]
    started = perf_counter()
    for chunk_meta in cached_manifest["chunks"]:
        stream_index += 1
        with np.load(ROOT / "data" / "cse_replay_chunk_cache_v2" / chunk_meta["file"], allow_pickle=False) as cache:
            x_array = cache["x"]
            y = np.asarray(cache["y"], dtype=np.int8)
        chunk = pd.DataFrame(x_array, columns=features)
        chunk["target"] = y
        # Evaluation uses only these two endpoints for chunk metadata.
        timestamp_ns = np.full(len(y), chunk_meta["timestamp_min_ns"], dtype=np.int64)
        timestamp_ns[-1] = chunk_meta["timestamp_max_ns"]
        chunk["Timestamp"] = pd.to_datetime(timestamp_ns)
        x = chunk[features].copy()
        actions = {}
        for name, state in states.items():
            pred_record = exp.evaluate_chunk(state["model"], chunk, features, decision_threshold=0.5)
            probabilities = pred_record.pop("_positive_probabilities")
            loss = binary_log_loss(y, probabilities)
            if name == "rccda_24":
                history_best = min(state["history"], default=loss)
                should_update, gain, next_queue = rccda_decision(
                    loss, state["previous_loss"], history_best, state["queue"],
                    1.0, RCCDA_MEAN_BUDGET_RATE, RCCDA_V, RCCDA_KP, RCCDA_KD,
                )
                decision = {"estimated_gain": gain, "queue_before": state["queue"], "queue_after": next_queue}
                state["queue"] = next_queue
                state["history"].append(loss)
                state["previous_loss"] = loss
            elif name == "periodic_24":
                should_update = stream_index in periodic_ids
                decision = {"scheduled_update": should_update}
            elif name == "random_24":
                should_update = stream_index in random_ids
                decision = {"scheduled_update": should_update, "schedule_seed": RANDOM_SCHEDULE_SEED}
            else:
                should_update = False
                decision = {}
            pred_record.update({
                "stream_chunk": stream_index, "capture_file": chunk_meta["capture_file"],
                "contains_attack": bool(y.sum()), "update_requested": bool(should_update),
                "model_updated_after_chunk": False, "log_loss": loss, **decision,
            })
            state["records"].append(pred_record)
            actions[name] = bool(should_update)

        # Only after all policy predictions are saved are these chunk labels released.
        recent.append((x, y.copy()))
        for name, state in states.items():
            if actions[name]:
                fit_seconds = exp.refit(state["model"], core_x, core_y, recent)
                state["updates"].append({"after_chunk": stream_index, "fit_seconds": fit_seconds})
                state["records"][-1]["model_updated_after_chunk"] = True
                state["records"][-1]["refit_seconds"] = fit_seconds
        if stream_index == 1 or stream_index % 100 == 0:
            print(f"Equal-budget replay progress: {stream_index}/{n_expected} chunks", flush=True)
        if stream_index % 100 == 0:
            interim = {
                "status": "running", "completed_chunks": stream_index,
                "elapsed_seconds": perf_counter() - started,
                "update_counts": {name: len(state["updates"]) for name, state in states.items()},
            }
            OUTPUT.with_suffix(".partial.json").write_text(json.dumps(interim, indent=2), encoding="utf-8")

    if stream_index != n_expected:
        raise ValueError(f"Replay built {stream_index} chunks, manifest records {n_expected}")
    summaries = {name: aggregate(state["records"]) for name, state in states.items()}
    if len(states["rccda_24"]["updates"]) != UPDATE_BUDGET:
        raise ValueError(
            f"Locked RCCDA scheduler produced {len(states['rccda_24']['updates'])} updates; "
            f"expected {UPDATE_BUDGET}. Do not interpret this as a matched-count comparison."
        )
    for name, summary in summaries.items():
        summary["refit_schedule"] = [row["after_chunk"] for row in states[name]["updates"]]
    result = {
        "status": "complete",
        "study": "Causal equal-update-budget comparison of fixed-configuration CSE adaptation policies",
        "date": "2026-11-02", "development_only": True,
        "source_paper": "Piaseczny et al., RCCDA, NeurIPS 2025, https://arxiv.org/abs/2505.24149",
        "method_limit": "RCCDA is ported only as a scheduler for LightGBM full refits. No SGD assumptions, optimality or theorem guarantees transfer.",
        "protocol": {
            "seed": SEED, "rows_per_chunk": CHUNK_ROWS,
            "target_cache_manifest_sha256": cached_manifest["manifest_sha256"],
            "target_chunks": n_expected,
            "exact_update_budget_per_adaptive_policy": UPDATE_BUDGET,
            "periodic_schedule": "24 update indices evenly spaced by rounded linspace over chunks 1..753",
            "random_schedule": "uniform sample without replacement of 24 chunk indices; seed 20261102",
            "labels_released_after_chunk_scoring": True,
            "shared_update_action": "refit same fixed LightGBM model on initial core source sample plus most recent three fully labeled chunks",
            "rccda_parameters": {"V": RCCDA_V, "Kp": RCCDA_KP, "Kd": RCCDA_KD,
                                 "mean_budget_rate": RCCDA_MEAN_BUDGET_RATE},
            "metrics_are_descriptive": True,
            "limitations": [
                "CSE has been repeatedly used for development; this is not independent confirmation.",
                "Exactly matching update counts does not guarantee equal CPU cost; elapsed refit time is recorded.",
                "One random 24-update schedule is only one realization; conclusions about randomized scheduling require multiple schedules.",
                "This is not an exact RCCDA reproduction and does not inherit its theoretical guarantees.",
            ],
        },
        "input_capture_hashes": cached_manifest["source_sha256"],
        "replay_exclusions": exclusions,
        "summaries": summaries,
        "wall_seconds": perf_counter() - started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    OUTPUT.with_suffix(".partial.json").unlink(missing_ok=True)
    for name, summary in summaries.items():
        print(name, {k: summary[k] for k in (
            "pooled_macro_f1", "pooled_fpr", "pooled_attack_recall", "updates", "total_refit_seconds"
        )})
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
