"""Summarize matched feasibility-first vs constraint-blind MOPSO runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    candidates = result["candidate_records"]
    if len(candidates) != 24:
        raise ValueError(f"Expected 24 candidates in {path}, found {len(candidates)}")
    if not candidates or "metrics" not in candidates[0]:
        raise ValueError(f"No valid MOPSO candidate records in {path}")
    limits = result["constraints"]
    feasible = [candidate for candidate in candidates
                if candidate["feasibility"]["feasible"]]
    best_all = max(candidates, key=lambda candidate: candidate["metrics"]["weighted_f1"])
    best_feasible = max(
        feasible, key=lambda candidate: candidate["metrics"]["weighted_f1"]
    ) if feasible else None
    return {
        "path": str(path),
        "mode": result["search_mode"],
        "seed": result["seed"],
        "dataset": result["dataset"],
        "constraints": limits,
        "dataset_sha256": result["dataset_sha256"],
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible),
        "archive_size": len(result["pareto_archive"]),
        "feasible_archive_count": sum(
            bool(candidate["feasibility"]["feasible"])
            for candidate in result["pareto_archive"]
        ),
        "unconstrained_best_candidate_metrics": best_all["metrics"],
        "best_feasible_candidate_metrics": best_feasible["metrics"] if best_feasible else None,
        "mean_candidate_evaluation_seconds": sum(
            float(item["evaluation_seconds"]) for item in candidates
        ) / len(candidates),
        "total_candidate_evaluation_seconds": sum(
            float(item["evaluation_seconds"]) for item in candidates
        ),
        "iteration_wall_seconds": [float(item["wall_seconds"])
                                   for item in result["iteration_log"]],
        "development_split_hash": candidates[0]["evaluation_metadata"]["development_index_sha256"],
        "reserved_holdout_split_hash": candidates[0]["evaluation_metadata"][
            "holdout_index_sha256_reserved_not_confirmatory"
        ],
        "holdout_qualification": result["outer_partition_note"],
    }


def main(paths: list[Path], output: Path) -> dict[str, Any]:
    summaries = [summarize(path) for path in paths]
    groups: dict[str, dict[str, Any]] = {}
    for row in summaries:
        key = f"{row['dataset']}:{row['seed']}"
        groups.setdefault(key, {})[row["mode"]] = row
    comparisons = []
    for key, modes in groups.items():
        if set(modes) != {"feasibility_first", "constraint_blind"}:
            raise ValueError(f"Incomplete paired modes for {key}: {sorted(modes)}")
        constrained = modes["feasibility_first"]
        blind = modes["constraint_blind"]
        for field in ("dataset_sha256", "candidate_count", "constraints",
                      "development_split_hash", "reserved_holdout_split_hash"):
            if constrained[field] != blind[field]:
                raise ValueError(f"Paired runs disagree on {field} for {key}")
        c = constrained["best_feasible_candidate_metrics"]
        b = blind["best_feasible_candidate_metrics"]
        comparisons.append({
            "dataset": constrained["dataset"],
            "seed": constrained["seed"],
            "constraints": constrained["constraints"],
            "feasibility_first": constrained,
            "constraint_blind": blind,
            "best_feasible_weighted_f1_difference_constrained_minus_blind": (
                c["weighted_f1"] - b["weighted_f1"] if c and b else None
            ),
            "best_feasible_macro_f1_difference_constrained_minus_blind": (
                c["macro_f1"] - b["macro_f1"] if c and b else None
            ),
            "feasible_candidate_count_difference_constrained_minus_blind": (
                constrained["feasible_candidate_count"] - blind["feasible_candidate_count"]
            ),
        })
    report = {
        "analysis_scope": "matched development search; prior grouped folds were inspected; not confirmatory",
        "method_note": (
            "The two search modes use the same metrics, candidate budget, grouped folds, and actual feasibility checks. "
            "Only whether feasibility guides MOPSO archive/personal-best selection differs."
        ),
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("results/mopso_constraint_ablation_analysis.json"))
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    report = main(args.runs, args.output)
    for row in report["comparisons"]:
        print(row["dataset"], "seed", row["seed"],
              "candidate feasibility counts",
              row["feasibility_first"]["feasible_candidate_count"],
              row["constraint_blind"]["feasible_candidate_count"],
              "best-feasible weighted F1 delta",
              row["best_feasible_weighted_f1_difference_constrained_minus_blind"])
