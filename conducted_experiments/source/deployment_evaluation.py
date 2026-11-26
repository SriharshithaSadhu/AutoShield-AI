"""Deployment-aware evaluation helpers for the MOO-AutoML IDS project.

These helpers are deliberately separate from the paper reproduction notebooks.
They evaluate an already fitted sklearn-compatible classifier without changing
the original notebook state. Run optimization on training/validation data and
reserve the test split for one final report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from time import perf_counter
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


@dataclass(frozen=True)
class DeploymentConstraints:
    """Hard limits for a target deployment profile.

    Latency is measured as p95 batch prediction latency divided by the number
    of rows in the batch. It is a per-row batch-throughput measure, not a
    single-request tail-latency guarantee.
    """

    min_weighted_f1: float = 0.95
    max_p95_latency_ms_per_row: float = 5.0
    max_serialized_size_mb: float = 1.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_weighted_f1 <= 1.0:
            raise ValueError("min_weighted_f1 must be between 0 and 1")
        if self.max_p95_latency_ms_per_row <= 0:
            raise ValueError("max_p95_latency_ms_per_row must be positive")
        if self.max_serialized_size_mb <= 0:
            raise ValueError("max_serialized_size_mb must be positive")


def expected_calibration_error(
    y_true: Iterable[Any], y_pred: Iterable[Any],
    probabilities: np.ndarray, n_bins: int = 10,
) -> float:
    """Top-label ECE using equal-width confidence bins, including empty bins safely."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or len(probabilities) != len(y_true_array):
        raise ValueError("probabilities must be a 2D array aligned with y_true")
    if len(y_pred_array) != len(y_true_array) or not len(y_true_array):
        raise ValueError("y_true and y_pred must be non-empty and aligned")

    confidence = probabilities.max(axis=1)
    correct = (y_pred_array == y_true_array).astype(float)
    bin_ids = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if np.any(mask):
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def evaluate_fitted_model(
    estimator: Any,
    X: Any,
    y: Iterable[Any],
    *,
    warmup_runs: int = 2,
    benchmark_runs: int = 15,
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Measure detection, calibration, batch latency, and serialized size.

    The estimator must implement ``predict`` and ``predict_proba``. Timing
    uses repeated predictions over the supplied batch after warmup. The model
    is serialized in memory with joblib so concurrent evaluations do not share
    or overwrite temporary files.
    """
    if warmup_runs < 0 or benchmark_runs < 2:
        raise ValueError("warmup_runs must be >= 0 and benchmark_runs must be >= 2")
    y_array = np.asarray(y)
    if not len(y_array) or len(X) != len(y_array):
        raise ValueError("X and y must be non-empty and have matching row counts")

    predictions = np.asarray(estimator.predict(X))
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(estimator, "classes_", []))
    if probabilities.ndim != 2 or probabilities.shape[0] != len(y_array):
        raise ValueError("predict_proba returned probabilities with an invalid shape")
    if len(classes) != probabilities.shape[1]:
        raise ValueError("estimator.classes_ must align with predict_proba columns")

    class_to_column = {label: index for index, label in enumerate(classes.tolist())}
    missing = [label for label in np.unique(y_array) if label not in class_to_column]
    if missing:
        raise ValueError(f"Evaluation labels absent from estimator.classes_: {missing!r}")
    true_class_probability = np.fromiter(
        (probabilities[row, class_to_column[label]] for row, label in enumerate(y_array)),
        dtype=float,
        count=len(y_array),
    )
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_array, predictions, labels=classes, zero_division=0
    )
    per_class = {
        str(label): {
            "precision": float(class_precision[index]),
            "recall": float(class_recall[index]),
            "f1": float(class_f1[index]),
            "support": int(class_support[index]),
        }
        for index, label in enumerate(classes.tolist())
    }

    for _ in range(warmup_runs):
        estimator.predict(X)
    elapsed_ms_per_row = np.empty(benchmark_runs, dtype=float)
    for run_index in range(benchmark_runs):
        start = perf_counter()
        estimator.predict(X)
        elapsed_ms_per_row[run_index] = (perf_counter() - start) * 1000.0 / len(y_array)

    buffer = BytesIO()
    joblib.dump(estimator, buffer, compress=0)
    serialized_size_mb = buffer.tell() / (1024.0 * 1024.0)

    return {
        "weighted_f1": float(f1_score(y_array, predictions, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_array, predictions, average="macro", zero_division=0)),
        "mean_true_class_probability": float(true_class_probability.mean()),
        "top_label_ece": expected_calibration_error(
            y_array, predictions, probabilities, n_bins=n_calibration_bins
        ),
        "per_class": per_class,
        "median_latency_ms_per_row": float(np.percentile(elapsed_ms_per_row, 50)),
        "p95_latency_ms_per_row": float(np.percentile(elapsed_ms_per_row, 95)),
        "serialized_size_mb": float(serialized_size_mb),
        "benchmark_runs": int(benchmark_runs),
        "benchmark_batch_rows": int(len(y_array)),
    }


def assess_feasibility(
    metrics: dict[str, Any], constraints: DeploymentConstraints,
) -> dict[str, Any]:
    """Return feasibility and machine-readable constraint violations."""
    checks = {
        "weighted_f1": (
            float(metrics["weighted_f1"]) >= constraints.min_weighted_f1,
            float(metrics["weighted_f1"]),
            constraints.min_weighted_f1,
            "min",
        ),
        "p95_latency_ms_per_row": (
            float(metrics["p95_latency_ms_per_row"]) <= constraints.max_p95_latency_ms_per_row,
            float(metrics["p95_latency_ms_per_row"]),
            constraints.max_p95_latency_ms_per_row,
            "max",
        ),
        "serialized_size_mb": (
            float(metrics["serialized_size_mb"]) <= constraints.max_serialized_size_mb,
            float(metrics["serialized_size_mb"]),
            constraints.max_serialized_size_mb,
            "max",
        ),
    }
    violations = [
        {"metric": name, "observed": observed, "limit": limit, "direction": direction}
        for name, (passed, observed, limit, direction) in checks.items()
        if not passed
    ]
    return {
        "feasible": not violations,
        "violations": violations,
        "constraints": asdict(constraints),
    }


def pareto_front(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return nondominated records over detection, calibration, and resource cost.

    When at least one record is feasible, infeasible records are excluded
    before Pareto filtering. If none is feasible, all records remain eligible
    so the least-bad trade-offs can be inspected instead of silently vanishing.
    Each record must contain ``metrics`` and ``feasibility`` mappings returned
    by this module.
    """
    candidates = [r for r in records if r["feasibility"]["feasible"]]
    no_feasible_candidate = not candidates
    if no_feasible_candidate:
        candidates = list(records)

    def objectives(record: dict[str, Any]) -> tuple[float, ...]:
        m = record["metrics"]
        performance_and_resource = (
            float(m["weighted_f1"]),
            float(m["macro_f1"]),
            -float(m["top_label_ece"]),
            -float(m["p95_latency_ms_per_row"]),
            -float(m["serialized_size_mb"]),
            -float(m.get("mean_fit_seconds", 0.0)),
        )
        if not no_feasible_candidate:
            return performance_and_resource

        # With no feasible solution, retain nondominated trade-offs in the
        # normalized violation of each explicit constraint as well as model
        # quality and cost. This exposes the least-violating frontier instead
        # of accidentally preferring a fast but arbitrarily inaccurate model.
        constraints = record["feasibility"]["constraints"]
        violations = (
            max(0.0, float(constraints["min_weighted_f1"]) - float(m["weighted_f1"]))
            / max(abs(float(constraints["min_weighted_f1"])), 1e-12),
            max(0.0, float(m["p95_latency_ms_per_row"]) - float(constraints["max_p95_latency_ms_per_row"]))
            / max(abs(float(constraints["max_p95_latency_ms_per_row"])), 1e-12),
            max(0.0, float(m["serialized_size_mb"]) - float(constraints["max_serialized_size_mb"]))
            / max(abs(float(constraints["max_serialized_size_mb"])), 1e-12),
        )
        return tuple(-v for v in violations) + performance_and_resource

    front: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_values = objectives(candidate)
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            other_values = objectives(other)
            if all(a >= b for a, b in zip(other_values, candidate_values)) and any(
                a > b for a, b in zip(other_values, candidate_values)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def evaluate_candidates(
    estimator_factory: Any,
    candidate_configs: Iterable[dict[str, Any]],
    X_train: Any,
    y_train: Iterable[Any],
    X_validation: Any,
    y_validation: Iterable[Any],
    *,
    constraints: DeploymentConstraints,
    warmup_runs: int = 2,
    benchmark_runs: int = 15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit and evaluate candidate configurations on one validation partition.

    ``estimator_factory(config)`` must return a fresh, unfitted estimator. Use
    this adapter for candidates proposed by MOPSO or a fixed search. All
    candidate scores are computed on the same validation rows; the untouched
    test set should be passed only to the final selected, refitted pipeline.
    Failed candidates are retained with their error text for auditability.
    Returns ``(all_records, pareto_records)``.
    """
    records: list[dict[str, Any]] = []
    for candidate_index, config in enumerate(candidate_configs):
        record: dict[str, Any] = {
            "candidate_index": candidate_index,
            "config": dict(config),
        }
        try:
            estimator = estimator_factory(dict(config))
            fit_started = perf_counter()
            estimator.fit(X_train, y_train)
            fit_seconds = perf_counter() - fit_started
            metrics = evaluate_fitted_model(
                estimator,
                X_validation,
                y_validation,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs,
            )
            metrics["mean_fit_seconds"] = float(fit_seconds)
            record["metrics"] = metrics
            record["feasibility"] = assess_feasibility(metrics, constraints)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["feasibility"] = {
                "feasible": False,
                "violations": [{"metric": "evaluation", "error": record["error"]}],
                "constraints": asdict(constraints),
            }
        records.append(record)

    evaluable = [record for record in records if "metrics" in record]
    return records, pareto_front(evaluable) if evaluable else []
