"""Compare sample-trained vs full-IoTID20 fixed LightGBM transfer models.

Development-only diagnostic. The LightGBM config, pipeline, labels, feature
intersection and CSE rows are fixed; the treatment is using the complete
publisher CSV instead of the existing 5% sample. No tuning or fitting uses
CSE data. Labels in each replay chunk are read only for retrospective scoring.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from experiment_protocol import make_leakage_safe_pipeline
from research_runner import make_estimator

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "provenance" / "IoTID20_official_original.csv"
CACHE = ROOT / "data" / "cse_replay_chunk_cache_v2"
SAMPLE_MODEL = ROOT / "results" / "end_to_end_models" / "reference_default.joblib"
FULL_MODEL = ROOT / "results" / "end_to_end_models" / "reference_full_iotid20_seed42.joblib"
OUT = ROOT / "results" / "full_iotid20_source_transfer_audit_20261102.json"
CONFIG = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "class_weight": "balanced",
    "verbosity": -1,
}
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int | float]:
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": fp / max(1, fp + tn),
        "attack_recall": tp / max(1, tp + fn),
    }


def score_model(model, features: list[str], manifest: dict) -> dict[str, object]:
    total_y: list[np.ndarray] = []
    total_pred: list[np.ndarray] = []
    by_capture: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"y": [], "pred": []})
    row_latency: list[float] = []
    inference_seconds = 0.0
    for chunk in manifest["chunks"]:
        capture = chunk["capture_file"]
        with np.load(CACHE / chunk["file"], allow_pickle=False) as data:
            x = data["x"]
            y = np.asarray(data["y"], dtype=np.int8)
        frame = pd.DataFrame(x, columns=features)
        started = perf_counter()
        pred = np.asarray(model.predict(frame), dtype=np.int8)
        elapsed = perf_counter() - started
        inference_seconds += elapsed
        row_latency.append(1000 * elapsed / max(1, len(y)))
        total_y.append(y)
        total_pred.append(pred)
        by_capture[capture]["y"].append(y)
        by_capture[capture]["pred"].append(pred)
    y_all = np.concatenate(total_y)
    pred_all = np.concatenate(total_pred)
    return {
        "pooled": confusion(y_all, pred_all),
        "by_capture": {
            capture: confusion(np.concatenate(values["y"]), np.concatenate(values["pred"]))
            for capture, values in sorted(by_capture.items())
        },
        "rows": int(len(y_all)),
        "inference_seconds_measured": float(inference_seconds),
        "p95_chunk_prediction_latency_ms_per_row": float(np.percentile(row_latency, 95)),
    }


def main() -> None:
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    features = manifest["features"]
    source = pd.read_csv(SOURCE, low_memory=False)
    source_feature_columns = [
        name for name in source.columns
        if name not in {"Flow_ID", "Src_IP", "Dst_IP", "Timestamp", "Label", "Sub_Cat", "Cat"}
        and pd.api.types.is_numeric_dtype(source[name])
    ]
    by_name: dict[str, list[str]] = defaultdict(list)
    for name in source_feature_columns:
        by_name[canonical(name)].append(name)
    collisions = {name: vals for name, vals in by_name.items() if len(vals) != 1}
    mapping = {name: by_name[canonical(name)][0] for name in features
               if len(by_name.get(canonical(name), [])) == 1}
    missing = [name for name in features if name not in mapping]
    if collisions or missing or len(mapping) != len(features):
        raise ValueError(f"Feature mapping is not one-to-one: collisions={collisions}; missing={missing}")
    normal_label = source["Label"].astype(str).str.strip().str.casefold().eq("normal")
    cat_normal = source["Cat"].astype(str).str.strip().str.casefold().eq("normal")
    if not np.array_equal(normal_label.to_numpy(), cat_normal.to_numpy()):
        raise ValueError("Binary Label and Cat Normal mappings disagree in the official source CSV")
    x_full = source[[mapping[name] for name in features]].copy()
    x_full.columns = features
    y_full = (~normal_label).to_numpy(dtype=np.int8)

    model = make_leakage_safe_pipeline(
        make_estimator("lightgbm", CONFIG, SEED, 1), random_state=SEED,
    )
    fit_start = perf_counter()
    model.fit(x_full, y_full)
    fit_seconds = perf_counter() - fit_start
    FULL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, FULL_MODEL, compress=0)

    sample_model = joblib.load(SAMPLE_MODEL)
    results = {
        "status": "complete",
        "experiment": "Fixed-configuration 5%-sample vs full-source IoTID20 transfer comparison",
        "date": "2026-11-02",
        "development_only": True,
        "treatment": "Replace the existing 31,289-row IoTID20 sample with the full 625,783-row publisher CSV; keep LightGBM config, pipeline, feature intersection, seed and CSE target replay fixed.",
        "source": {
            "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE),
            "rows": int(len(source)), "features": len(features),
            "binary_label_rule": "official Label=Normal -> benign; Label=Anomaly -> attack; checked equal to Cat=Normal for every source row",
            "class_counts": {"benign": int((y_full == 0).sum()), "attack": int((y_full == 1).sum())},
            "source_temporal_limits": "The publisher CSV has ten collection dates, with Normal traffic only on 2019-05-20 and attack categories concentrated on separate dates; no random source split is interpreted as temporal confirmation.",
        },
        "model": {
            "config": CONFIG, "seed": SEED, "pipeline": "same fold-adaptive normalization and HybridSMOTEADASYN pipeline as sample-trained reference",
            "full_source_fit_seconds": float(fit_seconds),
            "sample_model_path": str(SAMPLE_MODEL.relative_to(ROOT)),
            "sample_model_sha256": sha256(SAMPLE_MODEL),
            "full_model_path": str(FULL_MODEL.relative_to(ROOT)),
            "full_model_sha256": sha256(FULL_MODEL),
            "full_model_bytes": int(FULL_MODEL.stat().st_size),
        },
        "target": {
            "cache": str(CACHE.relative_to(ROOT)), "manifest_sha256": manifest["manifest_sha256"],
            "population_rows": int(sum(c["rows"] for c in manifest["chunks"])),
            "chunks": len(manifest["chunks"]), "captures": sorted({c["capture_file"] for c in manifest["chunks"]}),
            "no_target_fit_or_tuning": True,
        },
        "results": {
            "5pct_sample_source_model": score_model(sample_model, features, manifest),
            "full_source_model": score_model(model, features, manifest),
        },
        "limitations": [
            "The target CSE collection has already been used for development and is not a confirmatory test.",
            "This is a fixed-configuration source-data-volume comparison; it does not compare adaptation strategies or tune the source model.",
            "IoTID20 labels and scenarios are temporally grouped; no random or full-source fit metric is presented as temporal generalization evidence.",
            "Feature mapping remains canonicalized by field name and the rate/duration invariant audit only validates three relationships.",
            "Retrospective CSE labels are used only to calculate outcomes after prediction; no model fit or threshold tuning uses CSE labels.",
        ],
        "semantic_audit": "results/iotid20_cse_feature_semantics_audit_20261102.json",
    }
    OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT), "source_rows": len(source), "benign": int((y_full == 0).sum()),
        "attack": int((y_full == 1).sum()), "fit_seconds": fit_seconds,
        "results": {name: {"pooled": value["pooled"], "rows": value["rows"]}
                    for name, value in results["results"].items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
