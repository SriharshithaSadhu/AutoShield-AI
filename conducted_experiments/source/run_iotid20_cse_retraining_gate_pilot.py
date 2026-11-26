"""Matched-label source-mixed vs target-only retraining pilot on CSE.

Development-only independent reimplementation inspired by Jeon & Kim (2026).
At one 25% prefix checkpoint, target training labels are sampled uniformly
from the prefix under a fixed 5,000-row budget. Hash-signature folds separate
training, threshold-validation, and future evaluation. Prefix-only and
retrospective full-period threshold gates are compared on the same candidates.
This is not a reproduction of the paper or a confirmatory evaluation.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cse_replay_chunk_cache_v2"
MANIFEST = CACHE / "manifest.json"
SCORE_CACHE = ROOT / "results" / "iotid20_cse_gate_score_cache_seed42.npz"
SCORE_META = SCORE_CACHE.with_suffix(".json")
SOURCE = ROOT / "data" / "iotid20_binary_common77_dev.csv"
MODEL = ROOT / "results" / "end_to_end_models" / "reference_default.joblib"
OUT = ROOT / "results" / "IOTID20_CSE_RETRAINING_GATE_PILOT.json"
REPORT = ROOT / "results" / "IOTID20_CSE_RETRAINING_GATE_PILOT.md"
N_FOLDS = 5
CHECKPOINT_FRACTION = 0.25
TARGET_TRAIN_LABEL_BUDGET = 5_000
VALIDATION_SAMPLE = 10_000
TEST_SAMPLE = 20_000
FPR_CAP = 0.01
TARGET_RECALL = 0.90
SEED = 7403


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add_reservoir(old_x, old_y, old_k, x, y, limit, rng):
    """Uniform reservoir via random priorities; bounded memory per stream."""
    if not len(y):
        return old_x, old_y, old_k
    keys = rng.random(len(y))
    if old_x is not None:
        x = np.concatenate((old_x, x), axis=0)
        y = np.concatenate((old_y, y), axis=0)
        keys = np.concatenate((old_k, keys))
    if len(y) > limit:
        take = np.argpartition(keys, limit - 1)[:limit]
        x, y, keys = x[take], y[take], keys[take]
    return x, y, keys


def wilson_upper(k: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0:
        return 1.0
    p, z2 = k / n, z * z
    den = 1 + z2 / n
    return float((p + z2 / (2 * n) + z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / den)


def calibrate(scores: np.ndarray, y: np.ndarray) -> dict:
    benign = np.sort(np.asarray(scores, dtype=np.float32)[np.asarray(y) == 0])
    attack_scores = np.asarray(scores, dtype=np.float32)[np.asarray(y) == 1]
    if not len(benign):
        return {"threshold": None, "benign_validation_rows": 0, "attack_validation_rows": len(attack_scores),
                "fpr": None, "fpr_wilson_upper": 1.0, "attack_recall": None, "gate_pass": False}
    lo, hi = 0, len(benign)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if wilson_upper(mid, len(benign)) <= FPR_CAP:
            lo = mid
        else:
            hi = mid - 1
    threshold = (float(np.nextafter(benign[0], -np.inf)) if lo == len(benign)
                 else float(np.nextafter(benign[len(benign) - lo - 1], np.inf)))
    fp = int(np.sum(benign >= threshold))
    recall = float(np.mean(attack_scores >= threshold)) if len(attack_scores) else None
    upper = wilson_upper(fp, len(benign))
    return {"threshold": threshold, "benign_validation_rows": len(benign),
            "attack_validation_rows": len(attack_scores), "false_positives": fp,
            "fpr": fp / len(benign), "fpr_wilson_upper": upper,
            "attack_recall": recall,
            "gate_pass": bool(upper <= FPR_CAP and recall is not None and recall >= TARGET_RECALL)}


def metrics(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.uint8)
    p = np.asarray(scores) >= threshold
    benign, attack = y == 0, y == 1
    tn, fp = int(np.sum(benign & ~p)), int(np.sum(benign & p))
    fn, tp = int(np.sum(attack & ~p)), int(np.sum(attack & p))
    return {"rows": len(y), "benign_rows": int(benign.sum()), "attack_rows": int(attack.sum()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fpr": fp / (fp + tn) if fp + tn else None,
            "attack_recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None}


def score_model(model, x: np.ndarray, features: list[str]) -> np.ndarray:
    return model.predict_proba(pd.DataFrame(x, columns=features))[:, 1].astype(np.float32)


def main():
    started = time.perf_counter()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    meta = json.loads(SCORE_META.read_text(encoding="utf-8"))
    if meta["cache_manifest_sha256"] != sha256(MANIFEST) or meta["score_cache_sha256"] != sha256(SCORE_CACHE):
        raise ValueError("Score cache does not match the immutable replay manifest")
    if meta["rows"] != 15_184_359 or meta["chunks"] != 753:
        raise ValueError("Unexpected CSE replay dimensions; inspect before proceeding")
    features = manifest["features"]
    cutoff = int(np.floor(len(manifest["chunks"]) * CHECKPOINT_FRACTION))
    base_rng = np.random.default_rng(SEED)
    train_pool = [None] * N_FOLDS
    prefix_val = [None] * N_FOLDS
    full_val = [None] * N_FOLDS
    future_test = [None] * N_FOLDS
    with np.load(SCORE_CACHE, allow_pickle=False) as z:
        cache_folds = z["folds"]
        cache_chunks = z["chunk_ids"]
        cache_labels = z["labels"]
        offset = 0
        for idx, item in enumerate(manifest["chunks"]):
            with np.load(CACHE / item["file"], allow_pickle=False) as chunk:
                x = chunk["x"]
                y = chunk["y"].astype(np.uint8, copy=False)
            end = offset + len(y)
            if not np.array_equal(cache_labels[offset:end], y) or not np.all(cache_chunks[offset:end] == idx):
                raise ValueError(f"score-cache alignment mismatch in chunk {idx}")
            folds = cache_folds[offset:end]
            is_prefix = idx < cutoff
            for fold in range(N_FOLDS):
                mask = folds == fold
                # Prefix target candidate pool: exact signature group folds
                # exclude each eventual test and validation fold downstream.
                if is_prefix:
                    val_fold = (fold + 1) % N_FOLDS
                    train_mask = (folds != fold) & (folds != val_fold)
                    train_pool[fold] = add_reservoir(
                        *(train_pool[fold] if train_pool[fold] is not None else (None, None, None)),
                        x[train_mask], y[train_mask], 20_000, base_rng)
                    # Validation reservoirs contain each signature fold and
                    # are separately maintained for causal-prefix vs oracle.
                    prefix_val[fold] = add_reservoir(
                        *(prefix_val[fold] if prefix_val[fold] is not None else (None, None, None)),
                        x[mask], y[mask], VALIDATION_SAMPLE, base_rng)
                # Full-period validation oracle includes future validation rows.
                full_val[fold] = add_reservoir(
                    *(full_val[fold] if full_val[fold] is not None else (None, None, None)),
                    x[mask], y[mask], VALIDATION_SAMPLE, base_rng)
                if idx >= cutoff:
                    future_test[fold] = add_reservoir(
                        *(future_test[fold] if future_test[fold] is not None else (None, None, None)),
                        x[mask], y[mask], TEST_SAMPLE, base_rng)
            offset = end
            if (idx + 1) % 100 == 0:
                print(f"Collected causal samples: {idx + 1}/{len(manifest['chunks'])} chunks", flush=True)

    source = pd.read_csv(SOURCE)
    source_y = source["is_attack"].to_numpy(dtype=np.uint8)
    source_x = source[features].to_numpy(dtype=np.float64)
    prototype = joblib.load(MODEL)
    fixed_rows = []
    fold_rows = []
    for test_fold in range(N_FOLDS):
        val_fold = (test_fold + 1) % N_FOLDS
        train_folds = [f for f in range(N_FOLDS) if f not in (test_fold, val_fold)]
        # This is a uniform row reservoir over all prefix records in the
        # three training-role folds, not a chunk- or fold-balanced sample.
        tx, ty = train_pool[test_fold][0], train_pool[test_fold][1]
        rng = np.random.default_rng(SEED + 100 + test_fold)
        chosen = rng.choice(len(ty), min(TARGET_TRAIN_LABEL_BUDGET, len(ty)), replace=False)
        target_x, target_y = tx[chosen], ty[chosen]
        vx, vy = prefix_val[val_fold][0], prefix_val[val_fold][1]
        ovx, ovy = full_val[val_fold][0], full_val[val_fold][1]
        ex, ey = future_test[test_fold][0], future_test[test_fold][1]
        fixed = joblib.load(MODEL)
        fixed_scores = score_model(fixed, ex, features)
        fixed_rows.append(metrics(fixed_scores, ey, 0.5))
        arm_results = {}
        for arm in ("target_only", "source_plus_target"):
            if arm == "target_only":
                fit_x, fit_y = target_x, target_y
            else:
                fit_x = np.concatenate((source_x, target_x), axis=0)
                fit_y = np.concatenate((source_y, target_y), axis=0)
            model = clone(prototype)
            fit_start = time.perf_counter()
            model.fit(pd.DataFrame(fit_x, columns=features), fit_y)
            fit_seconds = time.perf_counter() - fit_start
            prefix_scores = score_model(model, vx, features)
            oracle_scores = score_model(model, ovx, features)
            test_scores = score_model(model, ex, features)
            prefix_gate = calibrate(prefix_scores, vy)
            oracle_gate = calibrate(oracle_scores, ovy)
            arm_results[arm] = {
                "fit_rows": int(len(fit_y)), "target_label_rows_used_for_fit": int(len(target_y)),
                "source_label_rows_used_for_fit": int(len(source_y) if arm == "source_plus_target" else 0),
                "fit_seconds": fit_seconds,
                "prefix_validation_sample": {"rows": int(len(vy)), "benign": int((vy == 0).sum()),
                                              "attack": int((vy == 1).sum())},
                "full_period_validation_sample": {"rows": int(len(ovy)), "benign": int((ovy == 0).sum()),
                                                   "attack": int((ovy == 1).sum())},
                "future_test_sample": {"rows": int(len(ey)), "benign": int((ey == 0).sum()),
                                        "attack": int((ey == 1).sum())},
                "thresholds": {
                    "prefix_only": {"validation": prefix_gate,
                                    "future_metrics": metrics(test_scores, ey, prefix_gate["threshold"])
                                    if prefix_gate["threshold"] is not None else None},
                    "full_period_retrospective": {"validation": oracle_gate,
                                                  "future_metrics": metrics(test_scores, ey, oracle_gate["threshold"])
                                                  if oracle_gate["threshold"] is not None else None},
                },
                "fixed_threshold_0_5_future_metrics": metrics(test_scores, ey, 0.5),
            }
        fold_rows.append({"test_signature_fold": test_fold, "validation_signature_fold": val_fold,
                          "training_signature_folds": train_folds, "training_target_label_budget": len(target_y),
                          "fixed_source_model_future_metrics": fixed_rows[-1], "arms": arm_results})
        print(f"Fold {test_fold + 1}/{N_FOLDS}: train target={len(target_y)}, test rows={len(ey)}; "
              f"target-only fit={arm_results['target_only']['fit_seconds']:.1f}s, "
              f"source+target fit={arm_results['source_plus_target']['fit_seconds']:.1f}s", flush=True)

    def pool(metric_rows):
        sums = {k: sum(r[k] for r in metric_rows) for k in ("tn", "fp", "fn", "tp", "rows", "benign_rows", "attack_rows")}
        return {**sums, "fpr": sums["fp"] / max(1, sums["fp"] + sums["tn"]),
                "attack_recall": sums["tp"] / max(1, sums["tp"] + sums["fn"]),
                "precision": sums["tp"] / max(1, sums["tp"] + sums["fp"])}

    aggregate = {"fixed_source_threshold_0_5": pool(fixed_rows)}
    for arm in ("target_only", "source_plus_target"):
        aggregate[arm] = {"threshold_0_5": pool([f["arms"][arm]["fixed_threshold_0_5_future_metrics"] for f in fold_rows])}
        for gate in ("prefix_only", "full_period_retrospective"):
            aggregate[arm][gate] = pool([f["arms"][arm]["thresholds"][gate]["future_metrics"] for f in fold_rows
                                         if f["arms"][arm]["thresholds"][gate]["future_metrics"] is not None])
            aggregate[arm][gate]["gate_pass_folds"] = sum(
                f["arms"][arm]["thresholds"][gate]["validation"]["gate_pass"] for f in fold_rows)
            aggregate[arm][gate]["future_operational_pass_folds"] = sum(
                bool(f["arms"][arm]["thresholds"][gate]["future_metrics"] and
                     f["arms"][arm]["thresholds"][gate]["future_metrics"]["fpr"] <= FPR_CAP and
                     f["arms"][arm]["thresholds"][gate]["future_metrics"]["attack_recall"] >= TARGET_RECALL)
                for f in fold_rows)

    result = {
        "status": "complete", "date": "2026-11-03", "development_only": True,
        "experiment": "matched-target-label source-mixed vs target-only adaptation and causal threshold gate pilot",
        "closest_work": "Independent reimplementation inspired by Jeon & Kim (2026); not exact reproduction",
        "data": {"manifest_sha256": sha256(MANIFEST), "score_cache_sha256": sha256(SCORE_CACHE),
                 "rows": int(meta["rows"]), "chunks": int(meta["chunks"]), "features": len(features),
                 "checkpoint_fraction": CHECKPOINT_FRACTION, "checkpoint_before_chunk": cutoff,
                 "future_chunks": len(manifest["chunks"]) - cutoff, "signature_folds": N_FOLDS},
        "protocol": {"target_training_label_budget_per_fold": TARGET_TRAIN_LABEL_BUDGET,
                     "target_label_sampling": "uniform row reservoir over prefix rows in the three training signature folds for each outer fold",
                     "source_training_rows": int(len(source_y)),
                     "validation_sample_per_fold": VALIDATION_SAMPLE,
                     "future_test_sample_per_fold": TEST_SAMPLE,
                     "validation": "one held-out signature fold; prefix-only labels vs full-period retrospective labels",
                     "test": "future rows from a separate held-out signature fold; uniform row reservoir",
                     "gate": {"fpr_cap": FPR_CAP, "wilson_upper_bound": "two-sided 95%", "recall_minimum": TARGET_RECALL},
                     "no_same_chunk_label_leakage": True},
        "models": {"target_only": "reference pipeline refit on 5,000 prefix CSE labels",
                   "source_plus_target": "same reference pipeline refit on all IoTID20 source training rows plus the exact same 5,000 prefix CSE labels",
                   "fixed_source_model": "existing IoTID20 reference model, no CSE fit"},
        "aggregate": aggregate, "folds": fold_rows,
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": "Exploratory CSE development evidence only. Outcomes do not establish independent generalization, novelty, or a safe deployment policy. The retrospective gate is an oracle because it uses later validation labels.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# IoTID20→CSE matched-label retraining and gate pilot",
        "",
        "**Status:** complete, development-only pilot; not an exact reproduction of Jeon & Kim (2026).",
        "",
        f"The experiment uses {meta['rows']:,} cached CSE rows in {meta['chunks']} immutable chunks. At the 25% checkpoint (before chunk {cutoff}), each held-out test fold is evaluated only on later rows. Training, threshold-validation, and test feature signatures occupy disjoint folds.",
        "",
        f"Both retraining arms receive exactly {TARGET_TRAIN_LABEL_BUDGET:,} target labels per fold. The source-mixed arm additionally uses the {len(source_y):,} IoTID20 source training rows; the target-only arm does not. Thresholds use either prefix-only validation labels or a retrospective full-period validation oracle. The operational gate is recall ≥0.90 with a 95% Wilson upper FPR bound ≤1%.",
        "",
        "## Pooled future sample outcomes",
        "",
        "| Arm | Threshold policy | FPR | Attack recall | Gate folds passed | Future target folds passed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    def fmt(v): return "—" if v is None else f"{v:.4%}"
    for arm, label in (("fixed_source_threshold_0_5", "Fixed source model (0.5)"),
                       ("target_only", "Target-only refit"), ("source_plus_target", "Source + target refit")):
        if arm.startswith("fixed"):
            m = aggregate[arm]
            lines.append(f"| {label} | fixed 0.5 | {fmt(m['fpr'])} | {fmt(m['attack_recall'])} | — | — |")
        else:
            for gate, gate_label in (("prefix_only", "prefix-only"), ("full_period_retrospective", "full-period oracle")):
                m = aggregate[arm][gate]
                lines.append(f"| {label} | {gate_label} | {fmt(m['fpr'])} | {fmt(m['attack_recall'])} | {m['gate_pass_folds']}/5 | {m['future_operational_pass_folds']}/5 |")
            m = aggregate[arm]["threshold_0_5"]
            lines.append(f"| {label} | fixed 0.5 | {fmt(m['fpr'])} | {fmt(m['attack_recall'])} | — | — |")
    lines += ["", "## Limits", "", "- This reuses a CSE stream repeatedly used during project development; it is not confirmation on independent data.",
              "- The binary task and chunk-level label release differ from the paper’s seven attack-specific tasks and EDG-level budget.",
              "- The 5,000-row target label budget is a project-controlled matched budget, not the paper’s 5–50 EDG budget.",
              "- The full-period gate is intentionally noncausal and only a retrospective oracle; it is not deployable.",
              "- Uniform row reservoirs provide a reproducible bounded sample, but this pilot does not establish rare-attack or capture-specific safety.",
              "- Source and target feature-exporter lineage remains incompletely pinned.", "",
              f"Machine-readable per-fold measurements: `{OUT.relative_to(ROOT).as_posix()}`."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "aggregate": aggregate,
                      "runtime_seconds": result["runtime_seconds"], "report": str(REPORT), "json": str(OUT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
