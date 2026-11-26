"""Development comparison of RCCDA's paper-level scheduler on CSE-CIC-IDS2018.

This ports the paper's causal loss-history estimator to the project's existing
LightGBM full-refit action.  It is NOT an exact reproduction of RCCDA's SGD
algorithm and carries none of its theorem/guarantees for LightGBM.  Chunk labels
arrive after scoring, and a triggered update can affect only the next chunk.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np

import drift_adaptation_experiment as exp


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "cse_rccda_inspired_equal_update_baselines_seed42.json"
SEED = 42
CHUNK_ROWS = 20_000
MAX_INITIAL_ROWS = 100_000
RECENT_CHUNKS = 3
UPDATE_BUDGET_RATE = 0.10
RCCDA_V = 10.0
RCCDA_KP = 1.0
RCCDA_KD = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
    y = np.asarray(y, dtype=np.int8)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def rccda_decision(loss: float, previous_loss: float | None,
                   historical_best: float, queue: float,
                   cost: float, mean_cost_budget: float,
                   v: float, kp: float, kd: float) -> tuple[bool, float, float]:
    """Paper's practical historical-loss estimator and Lyapunov queue test."""
    prior = loss if previous_loss is None else previous_loss
    best = min(historical_best, loss)
    gain_estimate = kp * (loss - best) + kd * (loss - prior)
    cost_threshold = cost * queue + 0.5 * (cost * cost - 2.0 * mean_cost_budget * cost)
    update = v * gain_estimate >= cost_threshold
    next_queue = max(0.0, queue + cost * int(update) - mean_cost_budget)
    return bool(update), float(gain_estimate), float(next_queue)


def main() -> None:
    manifest_path = exp.MANIFEST_PATH
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing CSE data manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = sorted(exp.RAW_DIR.glob("*_TrafficForML_CICFlowMeter.csv"), key=lambda p: exp.capture_date(p.name))
    initial_paths = [p for p in paths if exp.capture_date(p.name) == exp.INITIAL_DATE]
    if len(initial_paths) != 1:
        raise ValueError(f"Expected one initial capture; found {len(initial_paths)}")
    initial_path = initial_paths[0]
    features = exp.feature_columns(initial_path)
    initial, initial_exclusions = exp.read_sorted_capture(initial_path, features)
    split = int(len(initial) * exp.INITIAL_TRAIN_FRACTION)
    initial_train = initial.iloc[:split]
    core_x, core_y = exp.stratified_cap(initial_train, features, MAX_INITIAL_ROWS, SEED)
    if len(np.unique(core_y)) != 2:
        raise ValueError("Initial training partition must contain both classes")
    initial_model = exp.fit_reference(core_x, core_y, SEED)
    del initial, initial_train

    policies = {
        "static": {"model": exp.clone_model(initial_model), "records": [], "updates": 0, "times": []},
        "rccda_paper_estimator_port": {
            "model": exp.clone_model(initial_model), "records": [], "updates": 0, "times": [],
            "queue": 0.0, "loss_history": [], "previous_loss": None,
        },
        "periodic_10pct": {"model": exp.clone_model(initial_model), "records": [], "updates": 0, "times": []},
        "random_bernoulli_10pct": {"model": exp.clone_model(initial_model), "records": [], "updates": 0, "times": []},
    }
    random_policy = random.Random(20261101)
    recent: deque[tuple] = deque(maxlen=RECENT_CHUNKS)
    exclusions = [{"file": initial_path.name, **initial_exclusions}]
    stream_index = 0
    update_cost = 1.0  # Count one full refit as one resource token.
    queue_budget = UPDATE_BUDGET_RATE * update_cost

    for path in paths:
        if path == initial_path:
            continue
        chunks, bad = exp.capture_chunks(path, features, CHUNK_ROWS)
        exclusions.append({"file": path.name, **bad})
        for chunk in chunks:
            stream_index += 1
            if stream_index == 1 or stream_index % 25 == 0:
                print(f"Replay progress: {stream_index} chunks, latest {path.name}", flush=True)
            y = chunk["target"].to_numpy(dtype=np.int8)
            features_x = chunk[features].copy()
            actions: dict[str, dict] = {}

            for name, state in policies.items():
                pred_record = exp.evaluate_chunk(state["model"], chunk, features, decision_threshold=0.5)
                probs = pred_record.pop("_positive_probabilities")
                loss = binary_log_loss(y, probs)
                should_update = False
                decision_details: dict[str, float | int | bool] = {"loss": loss}

                if name == "rccda_paper_estimator_port":
                    history_best = min(state["loss_history"], default=loss)
                    should_update, gain, next_queue = rccda_decision(
                        loss=loss,
                        previous_loss=state["previous_loss"],
                        historical_best=history_best,
                        queue=state["queue"],
                        cost=update_cost,
                        mean_cost_budget=queue_budget,
                        v=RCCDA_V, kp=RCCDA_KP, kd=RCCDA_KD,
                    )
                    decision_details.update({
                        "estimated_gain": gain,
                        "queue_before": state["queue"],
                        "queue_after": next_queue,
                        "cost_threshold": update_cost * state["queue"] + 0.5 * (update_cost**2 - 2 * queue_budget * update_cost),
                        "budget_rate": UPDATE_BUDGET_RATE,
                    })
                    state["queue"] = next_queue
                    state["loss_history"].append(loss)
                    state["previous_loss"] = loss
                elif name == "periodic_10pct":
                    should_update = (stream_index % round(1 / UPDATE_BUDGET_RATE) == 0)
                elif name == "random_bernoulli_10pct":
                    should_update = random_policy.random() < UPDATE_BUDGET_RATE

                pred_record.update({
                    "stream_chunk": stream_index,
                    "capture_file": path.name,
                    "contains_attack": bool(y.sum()),
                    "first_attack_timestamp": (
                        chunk.loc[chunk["target"].eq(1), "Timestamp"].iloc[0].isoformat()
                        if y.any() else None
                    ),
                    "update_requested": bool(should_update),
                    "model_updated_after_chunk": False,
                    **decision_details,
                })
                state["records"].append(pred_record)
                actions[name] = {"update": bool(should_update), "loss": loss}

            # Current labels become available after scoring; they can only
            # contribute to this update for the next chunk.
            recent.append((features_x, y.copy()))
            for name, state in policies.items():
                if actions[name]["update"]:
                    elapsed = exp.refit(state["model"], core_x, core_y, recent)
                    state["times"].append(elapsed)
                    state["updates"] += 1
                    state["records"][-1]["model_updated_after_chunk"] = True

    summaries = {}
    for name, state in policies.items():
        rows = state["records"]
        tn = sum(r["tn"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        tp = sum(r["tp"] for r in rows)
        attack_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        benign_f1 = 2 * tn / max(1, 2 * tn + fp + fn)
        total_rows = tn + fp + fn + tp
        pooled_weighted_f1 = ((tp + fn) * attack_f1 + (tn + fp) * benign_f1) / max(1, total_rows)
        summaries[name] = {
            "chunks": len(rows), "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "pooled_macro_f1": (attack_f1 + benign_f1) / 2,
            "pooled_weighted_f1": pooled_weighted_f1,
            "row_weighted_mean_chunk_weighted_f1": float(np.average([r["weighted_f1"] for r in rows], weights=[r["rows"] for r in rows])),
            "pooled_fpr": fp / max(1, fp + tn),
            "pooled_attack_recall": tp / max(1, tp + fn),
            "mean_chunk_macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
            "mean_chunk_fpr": float(np.mean([r["false_positive_rate"] for r in rows])),
            "mean_chunk_log_loss": float(np.mean([r["loss"] for r in rows])),
            "updates": state["updates"], "update_rate": state["updates"] / max(1, len(rows)),
            "total_refit_seconds": float(sum(state["times"])),
            "median_refit_seconds": float(np.median(state["times"])) if state["times"] else 0.0,
            "queue_final": state.get("queue"),
            "chunk_records": rows,
        }

    result = {
        "study": "RCCDA-inspired update scheduler port on CSE-CIC-IDS2018",
        "status": "development comparison only; same previously explored CSE schedule",
        "source_paper": "Piaseczny et al., NeurIPS 2025, arXiv:2505.24149v3, https://arxiv.org/abs/2505.24149",
        "official_code_commit": "91d0558f0b77d476b69d4e21aaa16c9e69881831",
        "method_limit": "The paper's single-gradient-step policy is ported only as an update scheduler for fixed LightGBM full refits. Its assumptions, optimality, and resource theorem do not transfer to this experiment.",
        "protocol": {
            "seed": SEED, "chunk_rows": CHUNK_ROWS, "max_initial_train_rows": MAX_INITIAL_ROWS,
            "labels_released_after_scoring_each_chunk": True,
            "update_training_data": "same initial core sample plus most recent 3 fully labeled replay chunks",
            "rccda_estimator": "Kp*(current_logloss - historical_min_logloss) + Kd*(current_logloss - previous_chunk_logloss)",
            "rccda_parameters": {"V": RCCDA_V, "Kp": RCCDA_KP, "Kd": RCCDA_KD,
                                 "update_cost_tokens": update_cost, "average_budget_tokens_per_chunk": queue_budget},
            "baselines": "static, every-10th-chunk periodic, and online Bernoulli 10% updates; identical LightGBM update action and training window",
            "limitations": [
                "CSE is development data, not independent confirmation.",
                "Equal update-count tokens do not represent variable CPU time; actual refit times are separately recorded.",
                "This is not an exact reproduction of RCCDA's SGD model-update experiment and carries no RCCDA theorem guarantee.",
            ],
        },
        "input_manifest_sha256": sha256_file(manifest_path),
        "replay_exclusions": exclusions,
        "summaries": summaries,
        "outputs": {"json": str(OUTPUT.relative_to(ROOT))},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    for name, summary in summaries.items():
        print(name, {k: summary[k] for k in ("pooled_macro_f1", "pooled_fpr", "pooled_attack_recall", "updates", "total_refit_seconds", "queue_final")})
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
