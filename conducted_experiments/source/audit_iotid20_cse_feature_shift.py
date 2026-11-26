"""Descriptive source-to-target feature-shift diagnostic for the saved transfer model.

Uses the fitted IoTID20 reference model and the fixed v2 CSE replay chunks.
Samples 5% of each intact chunk with a fixed seed; no model fitting, target
selection, or adaptation occurs. Metrics are effect sizes, not causal claims.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "iotid20_binary_common77_dev.csv"
MODEL_PATH = ROOT / "results" / "end_to_end_models" / "reference_default.joblib"
CACHE = ROOT / "data" / "cse_replay_chunk_cache_v2"
OUT_JSON = ROOT / "results" / "iotid20_cse_feature_shift_audit_20261102.json"
OUT_CSV = ROOT / "results" / "iotid20_cse_feature_shift_effects_20261102.csv"
SEED = 20261102
SAMPLE_FRACTION = 0.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def class_shift(source: np.ndarray, target: np.ndarray) -> dict[str, float | int | None]:
    a, b = finite(source), finite(target)
    if not len(a) or not len(b):
        return {"source_n": int(len(a)), "target_n": int(len(b)), "ks": None,
                "wasserstein_over_source_iqr": None, "median_shift_over_source_iqr": None}
    q25, q75 = np.quantile(a, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(a))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return {
        "source_n": int(len(a)),
        "target_n": int(len(b)),
        "ks": float(ks_2samp(a, b, method="auto").statistic),
        "wasserstein_over_source_iqr": float(wasserstein_distance(a, b) / scale),
        "median_shift_over_source_iqr": float((np.median(b) - np.median(a)) / scale),
    }


def within_target_fp_effect(benign: np.ndarray, false_positive: np.ndarray) -> dict[str, float | int | None]:
    a, b = finite(benign), finite(false_positive)
    if not len(a) or not len(b):
        return {"benign_n": int(len(a)), "false_positive_n": int(len(b)),
                "ks": None, "median_shift_over_benign_iqr": None}
    q25, q75 = np.quantile(a, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(a))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return {
        "benign_n": int(len(a)),
        "false_positive_n": int(len(b)),
        "ks": float(ks_2samp(a, b, method="auto").statistic),
        "median_shift_over_benign_iqr": float((np.median(b) - np.median(a)) / scale),
    }


def main() -> None:
    source_df = pd.read_csv(SOURCE)
    if "is_attack" not in source_df:
        raise ValueError("Source reference must include is_attack target")
    features = [column for column in source_df.columns if column != "is_attack"]
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    if manifest["features"] != features:
        raise ValueError("Source feature order differs from the audited CSE cache manifest")

    rng = np.random.default_rng(SEED)
    target_x_parts: list[np.ndarray] = []
    target_y_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    capture_parts: list[np.ndarray] = []
    model = joblib.load(MODEL_PATH)

    for chunk_meta in manifest["chunks"]:
        cache_path = CACHE / chunk_meta["file"]
        with np.load(cache_path, allow_pickle=False) as record:
            x = record["x"]
            y = record["y"].astype(np.int8, copy=False)
            n = len(y)
            take = min(n, max(1, int(round(n * SAMPLE_FRACTION))))
            indexes = np.sort(rng.choice(n, size=take, replace=False))
            xs = np.asarray(x[indexes], dtype=np.float32)
            ys = y[indexes]
        frame = pd.DataFrame(xs, columns=features)
        pred = np.asarray(model.predict(frame), dtype=np.int8)
        target_x_parts.append(xs)
        target_y_parts.append(ys)
        prediction_parts.append(pred)
        capture_parts.append(np.repeat(chunk_meta["capture_file"], len(ys)))

    target_x = np.concatenate(target_x_parts, axis=0)
    target_y = np.concatenate(target_y_parts)
    target_pred = np.concatenate(prediction_parts)
    captures = np.concatenate(capture_parts)
    source_x = source_df[features].to_numpy(dtype=np.float64)
    source_y = source_df["is_attack"].to_numpy(dtype=np.int8)
    false_positive = (target_y == 0) & (target_pred == 1)
    model_importance = np.asarray(model.named_steps["model"].feature_importances_, dtype=np.float64)

    effects = []
    for index, feature in enumerate(features):
        source_values = source_x[:, index]
        target_values = target_x[:, index]
        source_benign = source_values[source_y == 0]
        source_attack = source_values[source_y == 1]
        target_benign = target_values[target_y == 0]
        target_attack = target_values[target_y == 1]
        target_fp = target_values[false_positive]
        source_benign_finite = finite(source_benign)
        target_benign_finite = finite(target_benign)
        source_q = np.quantile(source_benign_finite, [0.05, 0.25, 0.50, 0.75, 0.95])
        target_q = np.quantile(target_benign_finite, [0.05, 0.50, 0.95])
        effect = {
            "feature": feature,
            "source_model_feature_importance": float(model_importance[index]),
            "benign_support_comparison": {
                "source_q05": float(source_q[0]), "source_q25": float(source_q[1]),
                "source_median": float(source_q[2]), "source_q75": float(source_q[3]),
                "source_q95": float(source_q[4]), "cse_benign_q05": float(target_q[0]),
                "cse_benign_median": float(target_q[1]), "cse_benign_q95": float(target_q[2]),
                "cse_benign_below_source_q05_fraction": float(np.mean(target_benign_finite < source_q[0])),
                "cse_benign_above_source_q95_fraction": float(np.mean(target_benign_finite > source_q[4])),
            },
            "source_to_cse_benign": class_shift(source_benign, target_benign),
            "source_to_cse_attack": class_shift(source_attack, target_attack),
            "cse_false_positive_vs_all_benign": within_target_fp_effect(target_benign, target_fp),
        }
        effects.append(effect)

    sample_confusion = {
        "tn": int(((target_y == 0) & (target_pred == 0)).sum()),
        "fp": int(false_positive.sum()),
        "fn": int(((target_y == 1) & (target_pred == 0)).sum()),
        "tp": int(((target_y == 1) & (target_pred == 1)).sum()),
    }
    by_capture = {}
    for capture in sorted(set(captures.tolist())):
        mask = captures == capture
        benign = mask & (target_y == 0)
        attack = mask & (target_y == 1)
        fp = int((benign & (target_pred == 1)).sum())
        tn = int((benign & (target_pred == 0)).sum())
        tp = int((attack & (target_pred == 1)).sum())
        fn = int((attack & (target_pred == 0)).sum())
        by_capture[capture] = {
            "sampled_rows": int(mask.sum()), "sampled_benign": int(benign.sum()),
            "sampled_attack": int(attack.sum()), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "sample_fpr": fp / max(1, fp + tn),
            "sample_attack_recall": tp / max(1, tp + fn),
        }

    effects.sort(
        key=lambda row: (
            row["source_to_cse_benign"]["ks"] or 0,
            row["cse_false_positive_vs_all_benign"]["ks"] or 0,
        ),
        reverse=True,
    )
    flat = []
    for row in effects:
        flat.append({"feature": row["feature"], "comparison": "benign_support_comparison", **row["benign_support_comparison"]})
        flat.append({"feature": row["feature"], "comparison": "source_model_feature_importance", "importance": row["source_model_feature_importance"]})
        for group in ("source_to_cse_benign", "source_to_cse_attack", "cse_false_positive_vs_all_benign"):
            flat.append({"feature": row["feature"], "comparison": group, **row[group]})

    result = {
        "status": "complete",
        "experiment": "Descriptive univariate feature-shift and false-positive association audit",
        "date": "2026-11-02",
        "seed": SEED,
        "sample_fraction_per_chunk": SAMPLE_FRACTION,
        "source": {
            "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE),
            "rows": int(len(source_df)), "features": len(features),
            "benign_rows": int((source_y == 0).sum()), "attack_rows": int((source_y == 1).sum()),
            "source_is_training_sample": True,
        },
        "target": {
            "cache": str(CACHE.relative_to(ROOT)), "manifest_sha256": manifest["manifest_sha256"],
            "source_data_sha256": manifest["source_sha256"], "chunks": len(manifest["chunks"]),
            "population_rows": int(sum(chunk["rows"] for chunk in manifest["chunks"])),
            "sampled_rows": int(len(target_y)), "sampled_benign": int((target_y == 0).sum()),
            "sampled_attack": int((target_y == 1).sum()),
            "captures": sorted(set(captures.tolist())),
            "excludes_initial_cse_capture": True,
        },
        "model": {"path": str(MODEL_PATH.relative_to(ROOT)), "sha256": sha256(MODEL_PATH)},
        "sampled_static_model_confusion": sample_confusion,
        "sampled_static_model_fpr": sample_confusion["fp"] / max(1, sample_confusion["fp"] + sample_confusion["tn"]),
        "sampled_static_model_attack_recall": sample_confusion["tp"] / max(1, sample_confusion["tp"] + sample_confusion["fn"]),
        "capture_metrics": by_capture,
        "top_source_to_cse_benign_shift": [
            {"feature": row["feature"], **row["source_to_cse_benign"]} for row in effects[:15]
        ],
        "top_out_of_source_benign_support": sorted(
            [{"feature": row["feature"], **row["benign_support_comparison"]} for row in effects],
            key=lambda row: row["cse_benign_below_source_q05_fraction"] + row["cse_benign_above_source_q95_fraction"],
            reverse=True,
        )[:15],
        "top_source_model_features": sorted(
            [{"feature": row["feature"], "importance": row["source_model_feature_importance"]} for row in effects],
            key=lambda row: row["importance"], reverse=True,
        )[:15],
        "limitations": [
            "The source reference is the same 31,289-row IoTID20 sample used to fit the model, not an independent source holdout.",
            "CSE sampling is a fixed-seed 5% sample from each intact v2 replay chunk; it covers nine replay captures and excludes the initial CSE capture.",
            "Univariate distribution differences and within-CSE false-positive associations are descriptive and do not establish causal features or semantic equivalence.",
            "The CSE cache inherits its timestamp/label exclusions and preprocessing from the fixed replay manifest; this audit does not rerun feature extraction from PCAP.",
            "No statistical p-values are interpreted as evidence; only effect-size summaries are reported.",
        ],
        "effects_csv": str(OUT_CSV.relative_to(ROOT)),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(flat).to_csv(OUT_CSV, index=False)
    print(json.dumps({
        "output": str(OUT_JSON), "sampled_rows": result["target"]["sampled_rows"],
        "sample_fpr": result["sampled_static_model_fpr"],
        "sample_attack_recall": result["sampled_static_model_attack_recall"],
        "top_shift_features": [row["feature"] for row in result["top_source_to_cse_benign_shift"][:10]],
    }, indent=2))


if __name__ == "__main__":
    main()
