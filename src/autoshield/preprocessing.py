"""Leakage-safe data preparation for the MOO-AutoML IDS experiments.

This module is an experimental track alongside the untouched reproduction
notebooks. Split raw rows first, and place every learned transform and sampler
inside an imbalanced-learn pipeline so each CV fold fits them on its training
partition only.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.base import BaseSampler
from scipy.stats import shapiro
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate

DEFAULT_NON_FEATURE_COLUMNS = {
    "flow_id",
    "flowid",
    "src_ip",
    "sourceip",
    "dst_ip",
    "destinationip",
    "timestamp",
    "ts",
    "time",
    "label",
    "type",
    "cat",
}


def numeric_network_features(
    frame: pd.DataFrame,
    *,
    target_column: str,
    exclude_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    """Separate labels/metadata and retain numeric traffic features.

    Address/flow identifiers and timestamps are excluded from the model input
    by default. Non-numeric columns are reported and excluded rather than
    silently coerced to zero; add a fold-safe categorical pipeline before
    including them in a future experiment.
    """
    if target_column not in frame:
        raise KeyError(f"Target column {target_column!r} is missing")
    reserved = {name.casefold() for name in DEFAULT_NON_FEATURE_COLUMNS}
    reserved.update(name.casefold() for name in exclude_columns)
    reserved.discard(target_column.casefold())
    metadata = [
        name
        for name in frame.columns
        if name.casefold() in reserved and name != target_column
    ]
    candidate = frame.drop(columns=[target_column, *metadata], errors="ignore")
    numeric_columns = candidate.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [name for name in candidate.columns if name not in numeric_columns]
    if not numeric_columns:
        raise ValueError("No numeric traffic features remain after exclusions")
    X = candidate.loc[:, numeric_columns].copy()
    y = frame[target_column].copy()
    report = {"excluded_metadata": metadata, "excluded_non_numeric": non_numeric}
    return X, y, report


class FoldAdaptiveNormalizer(BaseEstimator, TransformerMixin):
    """Fit a per-column z-score or min-max transform on training rows only.

    A deterministic sample of at most 5,000 finite training values is used for
    the Shapiro normality heuristic. Non-finite values are imputed using the
    training median. Constant and all-missing columns map to zero. This keeps
    the notebook's adaptive normalization idea while preventing statistics
    from the validation/test partitions from entering the transform.
    """

    def __init__(self, alpha: float = 0.05, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "FoldAdaptiveNormalizer":
        frame = self._as_frame(X)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.medians_: dict[Any, float] = {}
        self.parameters_: dict[Any, tuple[str, float, float]] = {}
        rng = np.random.RandomState(self.random_state)

        for column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            values[~np.isfinite(values)] = np.nan
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            self.medians_[column] = median
            if len(finite) <= 1 or np.ptp(finite) == 0:
                self.parameters_[column] = ("constant", median, 0.0)
                continue

            sample = finite
            if len(sample) > 5000:
                sample = rng.choice(sample, size=5000, replace=False)
            try:
                _, p_value = shapiro(sample)
                use_zscore = bool(p_value > self.alpha)
            except (ValueError, FloatingPointError):
                use_zscore = False

            if use_zscore:
                center = float(np.mean(finite))
                scale = float(np.std(finite))
                self.parameters_[column] = (
                    "zscore",
                    center,
                    scale if scale > 0 else 1.0,
                )
            else:
                low, high = float(np.min(finite)), float(np.max(finite))
                self.parameters_[column] = (
                    "minmax",
                    low,
                    high - low if high > low else 1.0,
                )
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("FoldAdaptiveNormalizer must be fitted before transform")
        frame = self._as_frame(X)
        expected = list(self.feature_names_in_)
        if list(frame.columns) != expected:
            raise ValueError(
                "Input columns or their order differ from the fitted training data"
            )
        output = {}
        for column in expected:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            values[~np.isfinite(values)] = np.nan
            values = np.where(np.isnan(values), self.medians_[column], values)
            kind, center, scale = self.parameters_[column]
            if kind == "constant":
                values = np.zeros_like(values)
            elif kind == "zscore":
                values = (values - center) / scale
            else:
                values = (values - center) / scale
            output[column] = values
        return pd.DataFrame(output, index=frame.index)

    @staticmethod
    def _as_frame(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise ValueError("X must be a two-dimensional feature matrix")
            frame = pd.DataFrame(
                array, columns=[f"feature_{i}" for i in range(array.shape[1])]
            )
        if frame.shape[1] == 0:
            raise ValueError("X must contain at least one feature")
        return frame


class HybridSMOTEADASYN(BaseSampler):
    """Fold-local two-stage oversampling, with a safe SMOTE fallback.

    Minority classes are brought halfway to the notebook's target count with
    SMOTE, then ADASYN completes the target. If ADASYN cannot construct valid
    neighborhoods in a fold, SMOTE completes the remaining targets instead.
    No validation or test rows are ever resampled.
    """

    _sampling_type = "over-sampling"

    def __init__(self, random_state: int = 42, k_neighbors: int = 5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        # BaseSampler's validation expects this public constructor attribute.
        self.sampling_strategy = "auto"

    def _fit_resample(self, X: Any, y: Any):
        counts = Counter(np.asarray(y).tolist())
        if len(counts) < 2:
            return X, y
        target_count = int(sum(counts.values()) / len(counts) / 2)
        final = {
            label: target_count
            for label, count in counts.items()
            if count < target_count
        }
        if not final:
            return X, y

        midpoint = {
            label: max(
                counts[label] + 1, int(np.ceil((counts[label] + target_count) / 2))
            )
            for label in final
        }
        min_neighbors = min(counts[label] for label in final) - 1
        if min_neighbors < 1:
            return X, y
        first_k = min(self.k_neighbors, min_neighbors)
        X_mid, y_mid = SMOTE(
            sampling_strategy=midpoint,
            random_state=self.random_state,
            k_neighbors=first_k,
        ).fit_resample(X, y)
        current = Counter(np.asarray(y_mid).tolist())
        remaining = {
            label: count for label, count in final.items() if current[label] < count
        }
        if not remaining:
            return X_mid, y_mid
        second_k = min(self.k_neighbors, min(current[label] for label in remaining) - 1)
        if second_k >= 1:
            try:
                return ADASYN(
                    sampling_strategy=remaining,
                    random_state=self.random_state,
                    n_neighbors=second_k,
                ).fit_resample(X_mid, y_mid)
            except ValueError:
                pass
        return SMOTE(
            sampling_strategy=remaining,
            random_state=self.random_state,
            k_neighbors=max(
                1, min(self.k_neighbors, min(current[label] for label in remaining) - 1)
            ),
        ).fit_resample(X_mid, y_mid)


def split_development_and_holdout(
    X: Any,
    y: Any,
    *,
    holdout_fraction: float = 0.20,
    random_state: int = 42,
):
    """Split raw observations before fitting normalization or balancing."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    return train_test_split(
        X,
        y,
        test_size=holdout_fraction,
        random_state=random_state,
        stratify=y,
    )


def make_leakage_safe_pipeline(estimator: Any, *, random_state: int = 42) -> Pipeline:
    """Create a fold-safe normalization → balancing → estimator pipeline."""
    return Pipeline(
        steps=[
            ("normalize", FoldAdaptiveNormalizer(random_state=random_state)),
            ("balance", HybridSMOTEADASYN(random_state=random_state)),
            ("model", estimator),
        ]
    )


def cross_validate_pipeline(
    estimator: Any,
    X_development: Any,
    y_development: Any,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Score a candidate only on stratified validation folds of development data."""
    counts = Counter(np.asarray(y_development).tolist())
    smallest_class = min(counts.values())
    folds = min(n_splits, smallest_class)
    if folds < 2:
        raise ValueError("At least two examples per class are needed for stratified CV")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    pipeline = make_leakage_safe_pipeline(estimator, random_state=random_state)
    scores = cross_validate(
        pipeline,
        X_development,
        y_development,
        cv=cv,
        scoring={
            "weighted_f1": "f1_weighted",
            "macro_f1": "f1_macro",
            "weighted_precision": "precision_weighted",
            "weighted_recall": "recall_weighted",
        },
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )
    return {
        "n_splits": folds,
        "fold_scores": {
            key.removeprefix("test_"): values.tolist()
            for key, values in scores.items()
            if key.startswith("test_")
        },
        "mean": {
            key.removeprefix("test_"): float(np.mean(values))
            for key, values in scores.items()
            if key.startswith("test_")
        },
        "std": {
            key.removeprefix("test_"): float(np.std(values, ddof=1))
            for key, values in scores.items()
            if key.startswith("test_")
        },
    }
