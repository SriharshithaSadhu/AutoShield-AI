"""Compare unconstrained and feasible-first selection on a shared candidate pool.

This is a post-hoc selection ablation on the existing random-search candidate
pool. It measures the cost of enforcing budgets; it does not compare optimizer
search trajectories or provide confirmatory performance estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = {
    "cicids2017": Path("results/cicids2017_lightgbm_random_dev_seed42042.json"),
    "iotid20": Path("results/iotid20_lightgbm_random_dev_seed42042.json"),
}


def candidate_metrics(record: dict[str, Any]) -> dict[str, float]:
    folds = record["folds"]
    if folds:
        fold_metrics = [fold["metrics"] for fold in folds]
        weighted = [float(fold["weighted_f1"]) for fold in fold_metrics]
        macro = [float(fold["macro_f1"]) for fold in fold_metrics]
        ece = [float(fold["top_label_ece"]) for fold in fold_metrics]
        latency = [float(fold["p95_latency_ms_per_row"]) for fold in fold_metrics]
        sizes = [float(fold["serialized_size_mb"]) for fold in fold_metrics]
        return {
            "weighted_f1": sum(weighted) / len(weighted),
            "macro_f1": sum(macro) / len(macro),
            "top_label_ece": sum(ece) / len(ece),
            "worst_fold_p95_latency_ms_per_row": max(latency),
            "worst_fold_serialized_size_mb": max(sizes),
        }
    agg = record["aggregate_metrics"]
    return {
        "weighted_f1": float(agg["weighted_f1"]["mean"]),
        "macro_f1": float(agg["macro_f1"]["mean"]),
        "top_label_ece": float(agg["top_label_ece"]["mean"]),
        "worst_fold_p95_latency_ms_per_row": float(agg["p95_latency_ms_per_row"]["max"]),
        "worst_fold_serialized_size_mb": float(agg["serialized_size_mb"]["max"]),
    }


def analyze(source_paths: dict[str, Path], weighted_f1_floor: float = 0.95) -> dict[str, Any]:
    report: dict[str, Any] = {
        "analysis": "shared random-search pool; post-hoc output selection only",
        "selection_rule_unconstrained": "maximize mean grouped-CV weighted F1 across the same 24 candidates",
        "selection_rule_feasible": "maximize mean grouped-CV weighted F1 among candidates meeting the stated quality, worst-fold p95 latency, and worst-fold size limits",
        "weighted_f1_floor": weighted_f1_floor,
        "datasets": {},
    }
    profiles = [
        {"name": f"size_{size:g}mb_latency_5ms", "max_size_mb": size,
         "max_p95_latency_ms_per_row": 5.0}
        for size in (1.5, 2.0, 4.0, 8.0)
    ] + [
        {"name": f"size_4mb_latency_{latency:g}ms", "max_size_mb": 4.0,
         "max_p95_latency_ms_per_row": latency}
        for latency in (0.025, 0.05, 0.1, 0.2)
    ]
    report["profiles"] = profiles
    for dataset, path in source_paths.items():
        source = json.loads(path.read_text(encoding="utf-8"))
        records = source["candidates"]
        if len(records) != 24:
            raise ValueError(f"Expected 24 matched candidates in {path}, found {len(records)}")
        candidates = [{"index": int(record["candidate_index"]),
                       "metrics": candidate_metrics(record)} for record in records]
        unconstrained = max(candidates, key=lambda row: row["metrics"]["weighted_f1"])
        dataset_result = {
            "candidate_source": str(path),
            "candidate_count": len(candidates),
            "unconstrained_best_weighted_f1": unconstrained,
            "profiles": [],
        }
        for profile in profiles:
            feasible = [row for row in candidates
                        if row["metrics"]["weighted_f1"] >= weighted_f1_floor
                        and row["metrics"]["worst_fold_p95_latency_ms_per_row"]
                        <= profile["max_p95_latency_ms_per_row"]
                        and row["metrics"]["worst_fold_serialized_size_mb"]
                        <= profile["max_size_mb"]]
            selected = max(feasible, key=lambda row: row["metrics"]["weighted_f1"]) if feasible else None
            dataset_result["profiles"].append({
                **profile,
                "feasible_count": len(feasible),
                "unconstrained_candidate_feasible": unconstrained in feasible,
                "selected_feasible_candidate": selected,
                "weighted_f1_change_vs_unconstrained": (
                    selected["metrics"]["weighted_f1"] - unconstrained["metrics"]["weighted_f1"]
                    if selected else None
                ),
                "macro_f1_change_vs_unconstrained": (
                    selected["metrics"]["macro_f1"] - unconstrained["metrics"]["macro_f1"]
                    if selected else None
                ),
            })
        report["datasets"][dataset] = dataset_result
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/constraint_selection_effect_seed42.json"))
    args = parser.parse_args()
    result = analyze(DATASETS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for dataset, data in result["datasets"].items():
        print(dataset, "unconstrained", data["unconstrained_best_weighted_f1"]["metrics"])
        for profile in data["profiles"]:
            selected = profile["selected_feasible_candidate"]
            print(profile["name"], "feasible", profile["feasible_count"],
                  "selected", selected["metrics"] if selected else None)
