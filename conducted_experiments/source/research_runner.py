"""Small, resumable experiment runner for the leakage-safe research track."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_info

from deployment_evaluation import (
    DeploymentConstraints, assess_feasibility, evaluate_fitted_model, pareto_front,
)
from experiment_protocol import make_leakage_safe_pipeline, numeric_network_features
from pareto_mopso import Dimension, run_pareto_mopso


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_estimator(family: str, config: dict[str, Any], seed: int, n_jobs: int):
    params = dict(config)
    params.update(random_state=seed, n_jobs=n_jobs)
    if family == "lightgbm":
        params.setdefault("verbosity", -1)
        return lgb.LGBMClassifier(**params)
    if family == "xgboost":
        params.setdefault("objective", "multi:softprob")
        params.setdefault("eval_metric", "mlogloss")
        params.setdefault("tree_method", "hist")
        return xgb.XGBClassifier(**params)
    raise ValueError("family must be 'lightgbm' or 'xgboost'")


def evaluate_candidate_cv(
    data_path: str | Path,
    *,
    target_column: str,
    family: str,
    config: dict[str, Any],
    constraints: DeploymentConstraints,
    n_splits: int = 3,
    seed: int = 42,
    n_jobs: int = 1,
    warmup_runs: int = 1,
    benchmark_runs: int = 5,
) -> dict[str, Any]:
    """Evaluate one candidate on identical, fold-local leakage-safe partitions."""
    data_path = Path(data_path)
    data = pd.read_csv(data_path)
    X, y_raw, feature_report = numeric_network_features(data, target_column=target_column)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    smallest_class = int(pd.Series(y).value_counts().min())
    folds = min(n_splits, smallest_class)
    if folds < 2:
        raise ValueError("At least two rows per class are required for stratified CV")
    # Keep repeated/near-identical feature vectors in a single fold. ToN_IoT's
    # released benchmark contains many replicated flow vectors; row-wise CV can
    # otherwise give an optimistic score by placing copies on both sides.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64)
    group_counts = pd.Series(groups).value_counts()
    group_sizes = data.groupby(pd.Series(groups, index=data.index), sort=False).size()
    group_target_counts = pd.DataFrame({"group": groups, "target": y}).groupby("group")["target"].nunique()
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)

    fold_records = []
    for fold_index, (train_indices, validation_indices) in enumerate(cv.split(X, y, groups), start=1):
        estimator = make_estimator(family, config, seed + fold_index - 1, n_jobs)
        pipeline = make_leakage_safe_pipeline(estimator, random_state=seed + fold_index - 1)
        X_train, X_validation = X.iloc[train_indices], X.iloc[validation_indices]
        y_train, y_validation = y[train_indices], y[validation_indices]
        pipeline.fit(X_train, y_train)
        metrics = evaluate_fitted_model(
            pipeline,
            X_validation,
            y_validation,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )
        fold_records.append({"fold": fold_index, "train_rows": len(train_indices),
                             "validation_rows": len(validation_indices), "metrics": metrics})

    metric_names = [
        "weighted_f1", "macro_f1", "mean_true_class_probability", "top_label_ece",
        "median_latency_ms_per_row", "p95_latency_ms_per_row", "serialized_size_mb",
    ]
    aggregate = {
        name: {
            "mean": float(np.mean([fold["metrics"][name] for fold in fold_records])),
            "std": float(np.std([fold["metrics"][name] for fold in fold_records], ddof=1))
              if len(fold_records) > 1 else 0.0,
            "max": float(np.max([fold["metrics"][name] for fold in fold_records])),
        }
        for name in metric_names
    }
    conservative_metrics = {
        "weighted_f1": aggregate["weighted_f1"]["mean"],
        "p95_latency_ms_per_row": aggregate["p95_latency_ms_per_row"]["max"],
        "serialized_size_mb": aggregate["serialized_size_mb"]["max"],
    }
    return {
        "dataset": data_path.name,
        "dataset_sha256": file_sha256(data_path),
        "target_column": target_column,
        "class_names": encoder.classes_.tolist(),
        "feature_columns": X.columns.tolist(),
        "feature_exclusions": feature_report,
        "family": family,
        "config": config,
        "seed": seed,
        "n_splits": folds,
        "duplicate_grouping": {
            "method": "hash of numeric model-feature vector; identical vectors stay in one fold",
            "unique_feature_vectors": int(len(group_counts)),
            "rows_in_repeated_vectors": int(group_sizes[group_sizes > 1].sum()),
            "feature_vectors_with_conflicting_labels": int((group_target_counts > 1).sum()),
        },
        "n_jobs": n_jobs,
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "threadpools": threadpool_info()},
        "aggregate_metrics": aggregate,
        "feasibility_using_mean_f1_and_worst_fold_resources": assess_feasibility(
            conservative_metrics, constraints
        ),
        "folds": fold_records,
    }


def evaluate_fixed_group_holdout(
    data_path: str | Path,
    *,
    target_column: str,
    family: str,
    config: dict[str, Any],
    constraints: DeploymentConstraints,
    n_splits: int = 5,
    holdout_fold: int = 0,
    seed: int = 42,
    n_jobs: int = 1,
    warmup_runs: int = 1,
    benchmark_runs: int = 5,
) -> dict[str, Any]:
    """Fit a fixed, predeclared candidate on development rows and score one reserved fold.

    Identical numeric feature vectors are grouped before splitting to prevent
    exact duplicate flows from appearing on both sides. The reserved fold is
    one StratifiedGroupKFold partition; no parameter search occurs here. If
    these rows were inspected earlier, this is exploratory, not a final test.
    """
    data_path = Path(data_path)
    data = pd.read_csv(data_path)
    X, y_raw, feature_report = numeric_network_features(data, target_column=target_column)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64)
    smallest_class = int(pd.Series(y).value_counts().min())
    folds = min(n_splits, smallest_class)
    if folds < 2:
        raise ValueError("At least two rows per class are required for grouped holdout")
    if not 0 <= holdout_fold < folds:
        raise ValueError(f"holdout_fold must be in [0, {folds - 1}]")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    train_indices, holdout_indices = list(splitter.split(X, y, groups))[holdout_fold]
    if set(groups[train_indices]).intersection(groups[holdout_indices]):
        raise RuntimeError("Grouped holdout contains a feature-vector group in both partitions")

    pipeline = make_leakage_safe_pipeline(
        make_estimator(family, config, seed, n_jobs), random_state=seed
    )
    pipeline.fit(X.iloc[train_indices], y[train_indices])
    metrics = evaluate_fitted_model(
        pipeline,
        X.iloc[holdout_indices],
        y[holdout_indices],
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )
    group_target_counts = pd.DataFrame({"group": groups, "target": y}).groupby("group")["target"].nunique()
    conservative = {
        "weighted_f1": metrics["weighted_f1"],
        "p95_latency_ms_per_row": metrics["p95_latency_ms_per_row"],
        "serialized_size_mb": metrics["serialized_size_mb"],
    }
    return {
        "dataset": data_path.name,
        "dataset_sha256": file_sha256(data_path),
        "target_column": target_column,
        "class_names": encoder.classes_.tolist(),
        "feature_columns": X.columns.tolist(),
        "feature_exclusions": feature_report,
        "family": family,
        "config": config,
        "seed": seed,
        "partition": "single held-out fold of StratifiedGroupKFold; candidate configuration fixed in advance",
        "n_splits": folds,
        "holdout_fold": holdout_fold,
        "train_rows": int(len(train_indices)),
        "holdout_rows": int(len(holdout_indices)),
        "train_class_counts": pd.Series(y[train_indices]).value_counts().sort_index().to_dict(),
        "holdout_class_counts": pd.Series(y[holdout_indices]).value_counts().sort_index().to_dict(),
        "train_index_sha256": hashlib.sha256(np.asarray(train_indices, dtype=np.int64).tobytes()).hexdigest(),
        "holdout_index_sha256": hashlib.sha256(np.asarray(holdout_indices, dtype=np.int64).tobytes()).hexdigest(),
        "duplicate_grouping": {
            "method": "hash of numeric model-feature vector; identical vectors stay within one partition",
            "unique_feature_vectors": int(pd.Series(groups).nunique()),
            "rows_in_repeated_vectors": int(pd.Series(groups).value_counts().loc[lambda v: v > 1].sum()),
            "feature_vectors_with_conflicting_labels": int((group_target_counts > 1).sum()),
        },
        "n_jobs": n_jobs,
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "threadpools": threadpool_info()},
        "holdout_metrics": metrics,
        "feasibility": assess_feasibility(conservative, constraints),
    }


def evaluate_development_candidates(
    data_path: str | Path,
    *,
    target_column: str,
    family: str,
    candidate_configs: list[dict[str, Any]],
    constraints: DeploymentConstraints,
    outer_splits: int = 5,
    holdout_fold: int = 0,
    inner_splits: int = 4,
    seed: int = 42,
    n_jobs: int = 1,
    warmup_runs: int = 1,
    benchmark_runs: int = 3,
) -> dict[str, Any]:
    """Compare fixed candidates on shared development-only grouped folds.

    The outer partition is regenerated from the same seed/fold rule as
    ``evaluate_fixed_group_holdout``. Its reserved outer fold is excluded entirely;
    candidates use identical inner grouped folds over the remaining rows.
    """
    data_path = Path(data_path)
    data = pd.read_csv(data_path)
    X, y_raw, feature_report = numeric_network_features(data, target_column=target_column)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64)
    smallest_class = int(pd.Series(y).value_counts().min())
    outer_count = min(outer_splits, smallest_class)
    if outer_count < 3:
        raise ValueError("At least three grouped outer folds are required for development CV plus holdout")
    if not 0 <= holdout_fold < outer_count:
        raise ValueError(f"holdout_fold must be in [0, {outer_count - 1}]")
    outer_cv = StratifiedGroupKFold(n_splits=outer_count, shuffle=True, random_state=seed)
    outer_partitions = list(outer_cv.split(X, y, groups))
    development_indices = np.concatenate([
        train_idx for fold_idx, (train_idx, _) in enumerate(outer_partitions)
        if fold_idx == holdout_fold
    ])
    heldout_indices = outer_partitions[holdout_fold][1]
    development_indices.sort()
    X_dev = X.iloc[development_indices].reset_index(drop=True)
    y_dev = y[development_indices]
    groups_dev = groups[development_indices]
    inner_count = min(inner_splits, int(pd.Series(y_dev).value_counts().min()))
    if inner_count < 2:
        raise ValueError("Development partition has too few examples per class for inner CV")
    inner_cv = StratifiedGroupKFold(n_splits=inner_count, shuffle=True, random_state=seed)
    inner_partitions = list(inner_cv.split(X_dev, y_dev, groups_dev))

    records = []
    metric_names = [
        "weighted_f1", "macro_f1", "mean_true_class_probability", "top_label_ece",
        "median_latency_ms_per_row", "p95_latency_ms_per_row", "serialized_size_mb",
    ]
    for candidate_index, config in enumerate(candidate_configs):
        fold_records = []
        for fold_index, (train_idx, validation_idx) in enumerate(inner_partitions, start=1):
            pipeline = make_leakage_safe_pipeline(
                make_estimator(family, config, seed + fold_index - 1, n_jobs),
                random_state=seed + fold_index - 1,
            )
            fit_start = perf_counter()
            pipeline.fit(X_dev.iloc[train_idx], y_dev[train_idx])
            fit_seconds = perf_counter() - fit_start
            metrics = evaluate_fitted_model(
                pipeline,
                X_dev.iloc[validation_idx],
                y_dev[validation_idx],
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs,
            )
            fold_records.append({
                "fold": fold_index,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(validation_idx)),
                "fit_seconds": float(fit_seconds),
                "metrics": metrics,
            })
        aggregate = {
            metric: {
                "mean": float(np.mean([fold["metrics"][metric] for fold in fold_records])),
                "std": float(np.std([fold["metrics"][metric] for fold in fold_records], ddof=1))
                    if len(fold_records) > 1 else 0.0,
                "max": float(np.max([fold["metrics"][metric] for fold in fold_records])),
            }
            for metric in metric_names
        }
        conservative = {
            "weighted_f1": aggregate["weighted_f1"]["mean"],
            "p95_latency_ms_per_row": aggregate["p95_latency_ms_per_row"]["max"],
            "serialized_size_mb": aggregate["serialized_size_mb"]["max"],
        }
        records.append({
            "candidate_index": candidate_index,
            "config": dict(config),
            "aggregate_metrics": aggregate,
            "mean_fit_seconds": float(np.mean([fold["fit_seconds"] for fold in fold_records])),
            "feasibility": assess_feasibility(conservative, constraints),
            "folds": fold_records,
        })

    front = pareto_front([
        {"config": record["config"],
         "metrics": {
             **{key: record["aggregate_metrics"][key]["mean"] for key in metric_names},
             "mean_fit_seconds": record["mean_fit_seconds"],
         },
         "feasibility": record["feasibility"],
         "candidate_index": record["candidate_index"]}
        for record in records
    ])
    return {
        "dataset": data_path.name,
        "dataset_sha256": file_sha256(data_path),
        "target_column": target_column,
        "class_names": encoder.classes_.tolist(),
        "feature_columns": X.columns.tolist(),
        "feature_exclusions": feature_report,
        "family": family,
        "seed": seed,
        "outer_partition": "same 5-fold StratifiedGroupKFold rule; selected reserved fold excluded from all candidate fitting/scoring; not necessarily an untouched confirmatory holdout",
        "holdout_fold": holdout_fold,
        "development_rows": int(len(development_indices)),
        "holdout_rows_reserved": int(len(heldout_indices)),
        "development_index_sha256": hashlib.sha256(np.asarray(development_indices, dtype=np.int64).tobytes()).hexdigest(),
        "holdout_index_sha256": hashlib.sha256(np.asarray(heldout_indices, dtype=np.int64).tobytes()).hexdigest(),
        "inner_splits": inner_count,
        "duplicate_grouping": "hash of numeric feature vectors; exact repeats stay together",
        "n_jobs": n_jobs,
        "constraints": constraints.__dict__,
        "candidates": records,
        "pareto_candidate_indices": [record["candidate_index"] for record in front],
    }


def search_lightgbm_mopso(
    data_path: str | Path,
    *,
    target_column: str,
    constraints: DeploymentConstraints,
    output_path: str | Path,
    seed: int = 42,
    particles: int = 6,
    iterations: int = 4,
    n_jobs: int = 1,
    feasibility_first: bool = True,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run constrained or constraint-blind MOPSO on shared grouped folds.

    Both modes use the same candidate metrics and actual feasibility reports;
    ``feasibility_first=False`` ignores feasibility during search and archive
    selection, allowing a matched algorithm ablation.
    """
    data_path = Path(data_path)
    dimensions = [
        Dimension("n_estimators", 50, 200, "int"),
        Dimension("max_depth", 5, 100, "int"),
        Dimension("learning_rate", 0.01, 0.3, "log_float"),
        Dimension("num_leaves", 10, 50, "int"),
        Dimension("min_child_samples", 10, 50, "int"),
    ]
    fixed = {"class_weight": "balanced", "verbosity": -1}

    if resume_from is not None:
        checkpoint = json.loads(Path(resume_from).read_text(encoding="utf-8"))
        cached = checkpoint.get("candidate_records", [])
        if not cached:
            raise ValueError("Cannot resume an MOPSO run without cached candidates")
        metadata = cached[0].get("evaluation_metadata", {})
        if metadata.get("dataset_sha256") != file_sha256(data_path):
            raise ValueError("MOPSO checkpoint dataset hash does not match the requested data")
        if metadata.get("dataset") != data_path.name:
            raise ValueError("MOPSO checkpoint dataset name does not match the requested data")
        if metadata.get("target_column") != target_column or metadata.get("seed") != seed:
            raise ValueError("MOPSO checkpoint target or seed does not match the requested run")
        for row in cached:
            if row.get("feasibility", {}).get("constraints") != constraints.__dict__:
                raise ValueError("MOPSO checkpoint resource profile does not match the requested run")

    def evaluate_config(config: dict[str, Any]) -> dict[str, Any]:
        result = evaluate_development_candidates(
            data_path,
            target_column=target_column,
            family="lightgbm",
            candidate_configs=[config],
            constraints=constraints,
            outer_splits=5,
            holdout_fold=0,
            inner_splits=4,
            seed=seed,
            n_jobs=n_jobs,
            warmup_runs=1,
            benchmark_runs=3,
        )
        record = result["candidates"][0]
        metrics = {
            key: record["aggregate_metrics"][key]["mean"]
            for key in (
                "weighted_f1", "macro_f1", "mean_true_class_probability",
                "top_label_ece", "median_latency_ms_per_row",
            )
        }
        metrics["p95_latency_ms_per_row"] = record["aggregate_metrics"]["p95_latency_ms_per_row"]["max"]
        metrics["serialized_size_mb"] = record["aggregate_metrics"]["serialized_size_mb"]["max"]
        metrics["mean_fit_seconds"] = record["mean_fit_seconds"]
        return {
            "metrics": metrics,
            "feasibility": record["feasibility"],
            "metadata": {
                "dataset": result["dataset"],
                "dataset_sha256": result["dataset_sha256"],
                "target_column": target_column,
                "seed": seed,
                "holdout_index_sha256_reserved_not_confirmatory": result["holdout_index_sha256"],
                "development_index_sha256": result["development_index_sha256"],
                "inner_splits": result["inner_splits"],
                "folds": record["folds"],
            },
        }

    result = run_pareto_mopso(
        evaluate_config,
        dimensions,
        fixed_config=fixed,
        particles=particles,
        iterations=iterations,
        archive_size=32,
        seed=seed,
        output_path=output_path,
        feasibility_first=feasibility_first,
        resume_from=resume_from,
    )
    result.update({
        "dataset": data_path.name,
        "dataset_sha256": file_sha256(data_path),
        "target_column": target_column,
        "n_jobs": n_jobs,
        "constraints": constraints.__dict__,
        "search_mode": "feasibility_first" if feasibility_first else "constraint_blind",
        "outer_partition_note": "Candidate scores use grouped development CV; reserved fold was inspected previously and is not a pristine test.",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "threadpools": threadpool_info()},
    })
    save_result(result, output_path)
    return result


def save_result(result: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
