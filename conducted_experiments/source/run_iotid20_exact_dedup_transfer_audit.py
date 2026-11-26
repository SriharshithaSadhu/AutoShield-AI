"""Measure exact-row deduplication's effect on fixed IoTID20→CSE transfer."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from experiment_protocol import make_leakage_safe_pipeline
from research_runner import make_estimator
from run_full_iotid20_source_transfer_audit import (
    CACHE, CONFIG, FULL_MODEL, OUT as FULL_SOURCE_AUDIT, ROOT,
    canonical, score_model, sha256,
)

SOURCE = ROOT / "data" / "provenance" / "IoTID20_official_original.csv"
DEDUP_MODEL = ROOT / "results" / "end_to_end_models" / "reference_iotid20_exact_dedup_seed42.joblib"
OUT = ROOT / "results" / "iotid20_exact_dedup_transfer_audit_20261102.json"
SEED = 42


def main() -> None:
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    features = manifest["features"]
    source = pd.read_csv(SOURCE, low_memory=False)
    source_columns = [
        name for name in source.columns
        if name not in {"Flow_ID", "Src_IP", "Dst_IP", "Timestamp", "Label", "Sub_Cat", "Cat"}
        and pd.api.types.is_numeric_dtype(source[name])
    ]
    by_name: dict[str, list[str]] = {}
    for name in source_columns:
        by_name.setdefault(canonical(name), []).append(name)
    mapping = {name: by_name[canonical(name)][0] for name in features if len(by_name.get(canonical(name), [])) == 1}
    if len(mapping) != len(features):
        raise ValueError("The frozen 77-feature source map is missing or ambiguous")

    normal_label = source["Label"].astype(str).str.strip().str.casefold().eq("normal")
    cat_normal = source["Cat"].astype(str).str.strip().str.casefold().eq("normal")
    if not np.array_equal(normal_label.to_numpy(), cat_normal.to_numpy()):
        raise ValueError("Official binary Label and Cat=Normal disagree")
    x_all = source[[mapping[name] for name in features]].copy()
    x_all.columns = features
    y_all = (~normal_label).to_numpy(dtype=np.int8)

    # Match the prior data-engineering paper's exact-row deduplication rule.
    # All target columns are included, so only fully identical published rows
    # are removed; no feature vector is collapsed solely because it recurs.
    unique_mask = ~source.duplicated(keep="first")
    x_unique = x_all.loc[unique_mask.to_numpy()].reset_index(drop=True)
    y_unique = y_all[unique_mask.to_numpy()]
    if len(source) - len(x_unique) != 164_087:
        raise ValueError("Expected exact-row duplicate count does not match the audited published source")

    model = make_leakage_safe_pipeline(
        make_estimator("lightgbm", CONFIG, SEED, 1), random_state=SEED,
    )
    started = perf_counter()
    model.fit(x_unique, y_unique)
    fit_seconds = perf_counter() - started
    DEDUP_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, DEDUP_MODEL, compress=0)

    full_model = joblib.load(FULL_MODEL)
    scores = {
        "published_full_rows": score_model(full_model, features, manifest),
        "exact_duplicate_rows_removed": score_model(model, features, manifest),
    }
    row_loss = {}
    for name, score in scores.items():
        row_loss[name] = {
            "pooled": score["pooled"],
            "rows": score["rows"],
            "capture_fpr": {capture: values["fpr"] for capture, values in score["by_capture"].items()},
            "capture_attack_recall": {capture: values["attack_recall"] for capture, values in score["by_capture"].items()},
            "p95_chunk_prediction_latency_ms_per_row": score["p95_chunk_prediction_latency_ms_per_row"],
            "inference_seconds_measured": score["inference_seconds_measured"],
        }
    result = {
        "status": "complete",
        "date": "2026-11-02",
        "experiment": "Exact published-row deduplication sensitivity for fixed IoTID20→CSE transfer",
        "development_only": True,
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": sha256(SOURCE),
            "rows_before": int(len(source)),
            "rows_after_exact_deduplication": int(len(x_unique)),
            "exact_duplicate_rows_removed": int(len(source) - len(x_unique)),
            "deduplication_key": "all 86 published columns, including Label, Cat and Sub_Cat (pandas drop_duplicates; keep first)",
            "class_counts_before": {"benign": int((y_all == 0).sum()), "attack": int((y_all == 1).sum())},
            "class_counts_after": {"benign": int((y_unique == 0).sum()), "attack": int((y_unique == 1).sum())},
            "note": "The duplicate count and post-dedup class counts reproduce the published IoTID20 data-engineering result (164,087 removed; 38,598 Normal; 423,098 Anomaly).",
        },
        "model": {
            "fixed_config": CONFIG,
            "seed": SEED,
            "pipeline": "same leakage-safe preprocessing and estimator as the full-source reference model",
            "features": len(features),
            "fit_seconds_exact_dedup_model": float(fit_seconds),
            "full_source_model_path": str(FULL_MODEL.relative_to(ROOT)),
            "full_source_model_sha256": sha256(FULL_MODEL),
            "dedup_model_path": str(DEDUP_MODEL.relative_to(ROOT)),
            "dedup_model_sha256": sha256(DEDUP_MODEL),
            "dedup_model_bytes": int(DEDUP_MODEL.stat().st_size),
        },
        "target": {
            "cache": str(CACHE.relative_to(ROOT)),
            "manifest_sha256": manifest["manifest_sha256"],
            "target_rows": int(sum(chunk["rows"] for chunk in manifest["chunks"])),
            "chunks": len(manifest["chunks"]),
            "captures": sorted({chunk["capture_file"] for chunk in manifest["chunks"]}),
            "no_target_fit_or_tuning": True,
        },
        "results": row_loss,
        "limits": [
            "CSE is a repeatedly explored development target, not independent confirmation.",
            "This tests exact duplicate rows only; it does not collapse recurring feature vectors across different timestamps.",
            "A change after deduplication is a sensitivity result and does not establish that duplication caused the original transfer failure.",
            "The CSE target labels are used only after prediction for retrospective scoring.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "source_rows_before_after": [len(source), len(x_unique)],
        "exact_rows_removed": len(source) - len(x_unique),
        "target_scores": {name: {"pooled": score["pooled"], "rows": score["rows"]} for name, score in scores.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
