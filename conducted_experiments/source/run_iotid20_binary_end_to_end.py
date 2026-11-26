"""Build and compare fixed IoTID20 binary IDS models for CSE transfer.

This is a development run. The integer-to-category alignment is audited
against the official IoTID20 CSV by matching exact numeric traffic signatures.
No AutoML search is run on CSE-CIC-IDS2018.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from deployment_evaluation import DeploymentConstraints
from drift_adaptation_experiment import feature_columns
from experiment_protocol import make_leakage_safe_pipeline
from research_runner import (
    evaluate_fixed_group_holdout,
    file_sha256,
    make_estimator,
    search_lightgbm_mopso,
)

ROOT = Path(__file__).parent
SOURCE = ROOT / "data" / "IoTID20_sample_0.05.csv"
CSE_RAW = ROOT / "data" / "cse_cic_ids2018" / "raw"
PREPARED = ROOT / "data" / "iotid20_binary_common77_dev.csv"
SEARCH_RESULT = ROOT / "results" / "iotid20_binary_common77_mopso_seed42.json"
OUT = ROOT / "results" / "iotid20_binary_common77_end_to_end_seed42.json"
MODEL_DIR = ROOT / "results" / "end_to_end_models"


def canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def main() -> None:
    source = pd.read_csv(SOURCE)
    first_capture = sorted(CSE_RAW.glob("*_TrafficForML_CICFlowMeter.csv"))[0]
    cse_features = feature_columns(first_capture)
    source_features = [name for name in source.columns if name not in {
        "Flow_ID", "Src_IP", "Dst_IP", "Timestamp", "Label", "Sub_Cat", "Cat"
    } and pd.api.types.is_numeric_dtype(source[name])]
    by_canonical: dict[str, list[str]] = {}
    for name in source_features:
        by_canonical.setdefault(canonical(name), []).append(name)
    collisions = {key: values for key, values in by_canonical.items() if len(values) != 1}
    if collisions:
        raise ValueError(f"Ambiguous IoTID20 canonical feature names: {collisions}")
    missing = [name for name in cse_features if canonical(name) not in by_canonical]
    if missing:
        raise ValueError(f"CSE replay features missing from IoTID20: {missing}")
    mapping = {name: by_canonical[canonical(name)][0] for name in cse_features}
    codes = sorted(int(value) for value in source["Label"].dropna().unique())
    if codes != [0, 1, 2, 3, 4]:
        raise ValueError(f"Unexpected IoTID20 label codes: {codes}")
    # The audit directly matches most sample rows to the publisher's textual
    # Cat labels. Published class frequencies corroborate ambiguous/unmatched rows.
    code_to_class = {0: "DoS", 1: "MITM", 2: "Mirai", 3: "Normal", 4: "Scan"}
    prepared = source[[mapping[name] for name in cse_features]].copy()
    prepared.columns = cse_features
    prepared["is_attack"] = source["Label"].astype(int).ne(3).astype(np.int8)
    PREPARED.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(PREPARED, index=False)

    constraints = DeploymentConstraints(
        min_weighted_f1=0.95,
        max_p95_latency_ms_per_row=5.0,
        max_serialized_size_mb=1.5,
    )
    search = search_lightgbm_mopso(
        PREPARED,
        target_column="is_attack",
        constraints=constraints,
        output_path=SEARCH_RESULT,
        seed=42,
        particles=6,
        iterations=4,
        n_jobs=1,
    )
    archive = search.get("pareto_archive", [])
    feasible = [row for row in archive if row.get("feasibility", {}).get("feasible")]
    if not feasible:
        raise RuntimeError("No resource-feasible MOPSO candidate; do not label one constrained-feasible")
    selected = max(feasible, key=lambda row: row["metrics"]["weighted_f1"])
    selected_config = dict(selected["config"])
    baseline_config = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "class_weight": "balanced",
        "verbosity": -1,
    }
    comparisons = {}
    for name, config in (("reference_default", baseline_config),
                         ("mopso_best_feasible", selected_config)):
        comparisons[name] = evaluate_fixed_group_holdout(
            PREPARED,
            target_column="is_attack",
            family="lightgbm",
            config=config,
            constraints=constraints,
            n_splits=5,
            holdout_fold=0,
            seed=42,
            n_jobs=1,
            benchmark_runs=3,
        )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, config in (("reference_default", baseline_config),
                         ("mopso_best_feasible", selected_config)):
        model = make_leakage_safe_pipeline(
            make_estimator("lightgbm", config, 42, 1), random_state=42
        )
        model.fit(prepared[cse_features], prepared["is_attack"].to_numpy())
        joblib.dump(model, MODEL_DIR / f"{name}.joblib", compress=0)

    mapping_audit_path = ROOT / "results" / "iotid20_label_mapping_source_audit.json"
    if not mapping_audit_path.exists():
        raise FileNotFoundError("Run audit_iotid20_label_mapping.py before the transfer experiment")
    mapping_audit = json.loads(mapping_audit_path.read_text(encoding="utf-8"))
    result = {
        "experiment": "IoTID20 binary common-feature end-to-end model preparation",
        "status": "source models fitted; CSE chronological transfer replay is next",
        "development_only": True,
        "source_dataset": SOURCE.name,
        "source_sha256": file_sha256(SOURCE),
        "prepared_sha256": file_sha256(PREPARED),
        "n_rows": int(len(prepared)),
        "features": cse_features,
        "feature_mapping_cse_to_iotid20": mapping,
        "feature_alignment": {
            "cse_features": len(cse_features),
            "one_to_one_aligned": len(mapping),
            "IoTID20_numeric_features_excluded": ["Src_Port", "Dst_Port"],
            "canonicalization": "case-insensitive, retain ASCII alphanumeric characters",
        },
        "raw_label_counts": {str(int(k)): int(v) for k, v in source["Label"].value_counts().sort_index().items()},
        "inferred_code_to_class": code_to_class,
        "label_mapping_evidence": {
            "type": "exact row-signature match to the official IoTID20 source plus frequency correspondence",
            "official_source_sha256": mapping_audit["official_source_sha256"],
            "unambiguous_matched_sample_rows": mapping_audit["matched_unambiguous_rows"],
            "ambiguous_signature_rows": mapping_audit["matched_ambiguous_rows"],
            "unmatched_sample_rows": mapping_audit["unmatched_rows"],
            "mapping_is_authoritative_codebook": False,
            "binary_rule": "code 3 (verified as Normal in unambiguous source matches) -> 0 benign; all other codes -> 1 attack",
            "source": "official IoTID20 Cat labels matched by exact 79-feature traffic signatures; unmatched rows follow the same mapping and published class frequencies; see results/iotid20_label_mapping_source_audit.json",
        },
        "constraints": constraints.__dict__,
        "mopso_search_path": str(SEARCH_RESULT.relative_to(ROOT)),
        "mopso_candidate_evaluations": len(search.get("candidate_records", [])),
        "selected_feasible_candidate": selected,
        "model_configs": {"reference_default": baseline_config,
                          "mopso_best_feasible": selected_config},
        "same_grouped_holdout_comparison": comparisons,
        "fitted_models": {
            "reference_default": str((MODEL_DIR / "reference_default.joblib").relative_to(ROOT)),
            "mopso_best_feasible": str((MODEL_DIR / "mopso_best_feasible.joblib").relative_to(ROOT)),
        },
        "next_step": "Carry both frozen models through the same chronological CSE replay; score each chunk before releasing its labels, then compare no-adaptation and fixed-configuration adaptation.",
    }
    OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "rows": result["n_rows"],
        "features": len(cse_features),
        "candidates": result["mopso_candidate_evaluations"],
        "selected_config": selected_config,
        "holdout": {key: value["holdout_metrics"] for key, value in comparisons.items()},
        "output": str(OUT),
    }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
