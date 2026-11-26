"""Fixed-model chronological replay and supervised adaptation on CSE-CIC-IDS2018.

This is deliberately not an AutoML search. It compares a fixed reference
LightGBM IDS, drift-triggered and periodic fixed-configuration updates, plus
fixed-model threshold policies. Labels for a replay chunk become available
only after that chunk has been scored.
"""

from __future__ import annotations

from collections import deque
import argparse
from datetime import datetime
import gc
from io import BytesIO
import json
import os
import platform
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any, Iterator

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import duckdb
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "cse_cic_ids2018" / "raw"
MANIFEST_PATH = ROOT / "data" / "cse_cic_ids2018" / "manifest.json"
OUTPUT_PATH = ROOT / "results" / "cse_cic_ids2018_adaptation.json"
INITIAL_DATE = datetime(2018, 2, 14)
CHUNK_ROWS = 20_000
INITIAL_TRAIN_FRACTION = 0.70
MAX_INITIAL_TRAIN_ROWS = 100_000
RECENT_LABEL_CHUNKS = 3
PERIODIC_EVERY_CHUNKS = 5
ALERT_CONFIRMATION_CHUNKS = 2
FPR_CAP = 0.001
SEED = 42

EXCLUDED_FEATURES = {
    "timestamp",
    "label",
    "flow id",
    "flow_id",
    "src ip",
    "dst ip",
    "source ip",
    "destination ip",
    "src port",
    "dst port",
    "source port",
    "destination port",
}


def capture_date(filename: str) -> datetime:
    match = re.search(r"(\d{2}-\d{2}-\d{4})", filename)
    if not match:
        raise ValueError(f"Cannot find capture date in {filename}")
    return datetime.strptime(match.group(1), "%d-%m-%Y")


def feature_columns(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    sample = pd.read_csv(path, nrows=10_000, low_memory=False)
    excluded = {name.casefold().strip() for name in EXCLUDED_FEATURES}
    candidates = [name for name in header if name.casefold().strip() not in excluded]
    numeric = [
        name for name in candidates if pd.api.types.is_numeric_dtype(sample[name])
    ]
    if not numeric:
        raise ValueError(
            "No numeric traffic features remain after removing identifiers"
        )
    return numeric


def read_sorted_capture(
    path: Path, features: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load one day at a time and explicitly sort its unordered source rows."""
    frame = pd.read_csv(
        path, usecols=["Timestamp", "Label", *features], low_memory=False
    )
    frame["Timestamp"] = pd.to_datetime(
        frame["Timestamp"].astype(str).str.strip(), dayfirst=True, errors="coerce"
    )
    frame["Label"] = frame["Label"].astype(str).str.strip()
    expected_date = capture_date(path.name).date()
    repeated_headers = frame["Label"].str.casefold().eq("label")
    bad_timestamps = frame["Timestamp"].isna()
    outside_capture_day = frame["Timestamp"].notna() & frame["Timestamp"].dt.date.ne(
        expected_date
    )
    exclusions = {
        "repeated_header_rows": int(repeated_headers.sum()),
        "invalid_timestamp_rows": int(bad_timestamps.sum()),
        "timestamps_outside_capture_day": int(outside_capture_day.sum()),
    }
    keep = ~repeated_headers & ~bad_timestamps & ~outside_capture_day
    frame = frame.loc[keep].copy()
    frame.sort_values("Timestamp", kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    for name in features:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    frame["target"] = frame["Label"].str.casefold().ne("benign").astype(np.int8)
    return frame, exclusions


def make_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    num_leaves=31,
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=seed,
                    verbosity=-1,
                ),
            ),
        ]
    )


def stratified_cap(
    frame: pd.DataFrame, features: list[str], cap: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    target = frame["target"].to_numpy()
    if len(frame) <= cap:
        selected = frame
    else:
        parts = []
        for label, group in frame.groupby("target", sort=True):
            n = max(1, round(cap * len(group) / len(frame)))
            parts.append(group.sample(n=min(n, len(group)), random_state=seed))
        selected = pd.concat(parts).sample(frac=1, random_state=seed)
    return selected[features].copy(), selected["target"].to_numpy(dtype=np.int8)


def time_chunks(frame: pd.DataFrame, rows: int = CHUNK_ROWS) -> Iterator[pd.DataFrame]:
    start = 0
    while start < len(frame):
        end = min(start + rows, len(frame))
        while (
            end < len(frame)
            and frame["Timestamp"].iloc[end] == frame["Timestamp"].iloc[end - 1]
        ):
            end += 1
        yield frame.iloc[start:end].copy()
        start = end


def stream_large_capture_chunks(
    path: Path,
    features: list[str],
    rows: int,
) -> tuple[Iterator[pd.DataFrame], dict[str, int]]:
    """Externally sort a large CSV with DuckDB and stream equal-time-safe chunks.

    The 20-February source file is larger than available RAM when materialized
    as a pandas frame. DuckDB's ORDER BY spills to a project-local temp folder;
    only bounded result batches are converted to pandas.
    """
    expected_date = capture_date(path.name).date().isoformat()
    escaped_path = path.resolve().as_posix().replace("'", "''")
    temp_dir = (ROOT / "data" / "cse_cic_ids2018" / "duckdb_temp").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET memory_limit='2GB'")
    connection.execute("SET threads=1")
    connection.execute(
        f"SET temp_directory='{temp_dir.as_posix().replace(chr(39), chr(39) * 2)}'"
    )
    connection.execute("SET max_temp_directory_size='50GB'")
    # CIC's 20-Feb export is not fully RFC-4180 strict; disable only the
    # strict-state check. Keep ignore_errors=false so malformed rows are not
    # silently discarded, and reconcile this reader's count with the pandas
    # streaming audit before accepting a full replay.
    source = f"read_csv('{escaped_path}', header=true, all_varchar=true, null_padding=true, strict_mode=false, ignore_errors=false, parallel=false)"
    timestamp_expr = "TRY_STRPTIME(TRIM(\"Timestamp\"), '%d/%m/%Y %H:%M:%S')"
    label_expr = 'TRIM("Label")'
    counts = connection.execute(f"""
        SELECT
          SUM(CASE WHEN LOWER({label_expr}) = 'label' THEN 1 ELSE 0 END),
          SUM(CASE WHEN {timestamp_expr} IS NULL THEN 1 ELSE 0 END),
          SUM(CASE WHEN {timestamp_expr} IS NOT NULL
                    AND CAST({timestamp_expr} AS DATE) <> DATE '{expected_date}'
                   THEN 1 ELSE 0 END)
        FROM {source}
    """).fetchone()
    exclusions = {
        "repeated_header_rows": int(counts[0] or 0),
        "invalid_timestamp_rows": int(counts[1] or 0),
        "timestamps_outside_capture_day": int(counts[2] or 0),
    }
    selected = [
        f'{timestamp_expr} AS "Timestamp"',
        f'{label_expr} AS "Label"',
    ]
    for name in features:
        quoted = '"' + name.replace('"', '""') + '"'
        selected.append(f"TRY_CAST(NULLIF(TRIM({quoted}), '') AS DOUBLE) AS {quoted}")
    query = f"""
        SELECT {', '.join(selected)}
        FROM {source}
        WHERE LOWER({label_expr}) <> 'label'
          AND {timestamp_expr} IS NOT NULL
          AND CAST({timestamp_expr} AS DATE) = DATE '{expected_date}'
        ORDER BY 1
    """

    def generate() -> Iterator[pd.DataFrame]:
        pending = pd.DataFrame()
        try:
            connection.execute(query)
            while True:
                batch = connection.fetch_df_chunk(vectors_per_chunk=10)
                if batch.empty:
                    break
                for name in features:
                    batch[name] = batch[name].replace([np.inf, -np.inf], np.nan)
                batch["target"] = (
                    batch["Label"].str.casefold().ne("benign").astype(np.int8)
                )
                pending = (
                    pd.concat((pending, batch), ignore_index=True)
                    if not pending.empty
                    else batch
                )
                while len(pending) > rows:
                    end = rows
                    while (
                        end < len(pending)
                        and pending["Timestamp"].iloc[end]
                        == pending["Timestamp"].iloc[end - 1]
                    ):
                        end += 1
                    # Keep a trailing timestamp group intact across DuckDB batches.
                    if end >= len(pending):
                        break
                    yield pending.iloc[:end].copy()
                    pending = pending.iloc[end:].reset_index(drop=True)
            while len(pending) > rows:
                end = rows
                while (
                    end < len(pending)
                    and pending["Timestamp"].iloc[end]
                    == pending["Timestamp"].iloc[end - 1]
                ):
                    end += 1
                yield pending.iloc[:end].copy()
                pending = pending.iloc[end:].reset_index(drop=True)
            if not pending.empty:
                yield pending
        finally:
            connection.close()

    return generate(), exclusions


def capture_chunks(
    path: Path, features: list[str], rows: int
) -> tuple[Iterator[pd.DataFrame], dict[str, int]]:
    if path.stat().st_size >= 1_000_000_000:
        return stream_large_capture_chunks(path, features, rows)
    frame, exclusions = read_sorted_capture(path, features)
    return time_chunks(frame, rows), exclusions


def fit_reference(train_x: pd.DataFrame, train_y: np.ndarray, seed: int) -> Pipeline:
    model = make_pipeline(seed)
    model.fit(train_x, train_y)
    return model


def psi_setup(
    reference: pd.DataFrame, max_features: int = 12
) -> tuple[list[str], dict[str, tuple[np.ndarray, np.ndarray]]]:
    variances = (
        reference.var(axis=0, skipna=True).fillna(0).sort_values(ascending=False)
    )
    selected = variances.head(min(max_features, len(variances))).index.tolist()
    bins: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in selected:
        values = (
            reference[name]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
        if not len(values):
            continue
        quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        if len(quantiles) < 3:
            edges = np.array([-np.inf, np.inf])
        else:
            edges = np.r_[-np.inf, quantiles[1:-1], np.inf]
        counts, _ = np.histogram(values, bins=edges)
        proportions = counts / max(1, counts.sum())
        bins[name] = (edges, proportions)
    return list(bins), bins


def psi_score(
    frame: pd.DataFrame,
    features: list[str],
    reference: dict[str, tuple[np.ndarray, np.ndarray]],
) -> float:
    values = []
    for name in features:
        edges, expected = reference[name]
        actual_values = (
            frame[name]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
        if not len(actual_values):
            continue
        counts, _ = np.histogram(actual_values, bins=edges)
        observed = counts / max(1, counts.sum())
        expected = np.clip(expected, 1e-6, None)
        observed = np.clip(observed, 1e-6, None)
        values.append(
            float(np.sum((observed - expected) * np.log(observed / expected)))
        )
    return float(np.mean(values)) if values else 0.0


def decision_ece(
    labels: np.ndarray,
    predictions: np.ndarray,
    positive_probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE using confidence in the class selected by the active threshold."""
    confidence = np.where(
        predictions == 1, positive_probability, 1.0 - positive_probability
    )
    correct = (predictions == labels).astype(float)
    bin_ids = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    value = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if np.any(mask):
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def evaluate_chunk(
    model: Pipeline,
    frame: pd.DataFrame,
    features: list[str],
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    X = frame[features]
    y = frame["target"].to_numpy(dtype=np.int8)
    started = perf_counter()
    probabilities = model.predict_proba(X)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    positive_probability = probabilities[:, positive_column]
    prediction = (positive_probability >= decision_threshold).astype(np.int8)
    elapsed = perf_counter() - started
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "rows": len(frame),
        "start_timestamp": frame["Timestamp"].iloc[0].isoformat(),
        "end_timestamp": frame["Timestamp"].iloc[-1].isoformat(),
        "attack_rows": int(y.sum()),
        "benign_rows": int((y == 0).sum()),
        "weighted_f1": float(
            f1_score(y, prediction, average="weighted", zero_division=0)
        ),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "ece": decision_ece(y, prediction, positive_probability, n_bins=10),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "prediction_latency_ms_per_row": elapsed * 1000.0 / max(1, len(frame)),
        "_positive_probabilities": positive_probability,
    }


def recalibrate_threshold(
    probabilities: np.ndarray, labels: np.ndarray, current_threshold: float
) -> float:
    """Choose a macro-F1 threshold from labeled history for use on future chunks."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=np.int8)
    if len(np.unique(labels)) < 2 or len(probabilities) != len(labels):
        return float(current_threshold)
    candidates = np.unique(
        np.r_[
            np.quantile(probabilities, np.linspace(0.0, 1.0, 101)),
            current_threshold,
            0.5,
        ]
    )
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probability = probabilities[order]
    sorted_positive = labels[order].astype(np.int64)
    cumulative_positive = np.cumsum(sorted_positive)
    total_positive = int(cumulative_positive[-1])
    total_negative = len(labels) - total_positive
    split = np.searchsorted(sorted_probability, candidates, side="left")
    predicted_positive = len(labels) - split
    true_positive = total_positive - np.where(
        split > 0, cumulative_positive[np.maximum(split - 1, 0)], 0
    )
    false_positive = predicted_positive - true_positive
    false_negative = total_positive - true_positive
    true_negative = total_negative - false_positive
    f1_positive = (
        2
        * true_positive
        / np.maximum(1, 2 * true_positive + false_positive + false_negative)
    )
    f1_negative = (
        2
        * true_negative
        / np.maximum(1, 2 * true_negative + false_positive + false_negative)
    )
    macro_f1 = (f1_positive + f1_negative) / 2
    # np.argmax is deterministic and prefers the smallest threshold on ties.
    return float(candidates[int(np.argmax(macro_f1))])


def wilson_upper_bound(
    successes: int, trials: int, z: float = 1.959963984540054
) -> float:
    """Two-sided 95% Wilson upper confidence bound for a binomial rate."""
    if trials <= 0:
        return 1.0
    rate = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = rate + z2 / (2.0 * trials)
    margin = z * np.sqrt(rate * (1.0 - rate) / trials + z2 / (4.0 * trials * trials))
    return float((center + margin) / denominator)


def recalibrate_threshold_fpr_capped(
    probabilities: np.ndarray,
    labels: np.ndarray,
    current_threshold: float,
    fpr_cap: float = FPR_CAP,
) -> tuple[float, dict[str, float | int]]:
    """Maximize past attack recall subject to a Wilson-bounded past FPR cap.

    This selects an operating point from already-labeled history only. A
    threshold strictly above the maximum observed score is included to ensure
    there is always a feasible zero-false-positive candidate.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=np.int8)
    negatives = labels == 0
    positives = labels == 1
    if len(probabilities) != len(labels) or not negatives.any():
        return float(current_threshold), {
            "validation_fpr": 0.0,
            "validation_fpr_wilson_upper": 1.0,
            "validation_attack_recall": 0.0,
            "validation_benign_rows": int(negatives.sum()),
        }

    candidates = np.unique(
        np.r_[
            np.quantile(probabilities, np.linspace(0.0, 1.0, 101)),
            current_threshold,
            0.5,
            np.nextafter(float(probabilities.max()), 1.0),
        ]
    )
    benign_scores = probabilities[negatives]
    attack_scores = probabilities[positives]
    benign_false_positives = np.array(
        [(benign_scores >= threshold).sum() for threshold in candidates], dtype=np.int64
    )
    upper_bounds = np.array(
        [
            wilson_upper_bound(int(fp), len(benign_scores))
            for fp in benign_false_positives
        ]
    )
    feasible = upper_bounds <= fpr_cap
    if not feasible.any():  # defensive; nextafter(max(score)) should be feasible
        selected = int(np.argmin(upper_bounds))
    else:
        recalls = np.array(
            [
                (
                    float((attack_scores >= threshold).mean())
                    if len(attack_scores)
                    else 0.0
                )
                for threshold in candidates
            ]
        )
        best_recall = recalls[feasible].max()
        tied = np.flatnonzero(feasible & (recalls == best_recall))
        # Prefer the higher threshold among equal-recall safe options.
        selected = int(tied[-1])

    fp = int(benign_false_positives[selected])
    recall = (
        float((attack_scores >= candidates[selected]).mean())
        if len(attack_scores)
        else 0.0
    )
    return float(candidates[selected]), {
        "validation_fpr": fp / len(benign_scores),
        "validation_fpr_wilson_upper": float(upper_bounds[selected]),
        "validation_attack_recall": recall,
        "validation_benign_rows": int(len(benign_scores)),
    }


def clone_model(model: Pipeline) -> Pipeline:
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    return joblib.load(buffer)


def refit(
    model: Pipeline,
    core_x: pd.DataFrame,
    core_y: np.ndarray,
    recent: deque[tuple[pd.DataFrame, np.ndarray]],
) -> float:
    started = perf_counter()
    recent_x = pd.concat([part[0] for part in recent], ignore_index=True)
    recent_y = np.concatenate([part[1] for part in recent])
    update_x = pd.concat([core_x, recent_x], ignore_index=True)
    update_y = np.concatenate([core_y, recent_y])
    model.fit(update_x, update_y)
    return perf_counter() - started


def main(
    seed: int = SEED,
    chunk_rows: int = CHUNK_ROWS,
    max_initial_train_rows: int = MAX_INITIAL_TRAIN_ROWS,
    output_path: Path = OUTPUT_PATH,
    psi_quantile: float = 0.95,
    error_sd_multiplier: float = 2.0,
    confirmation_chunks: int = ALERT_CONFIRMATION_CHUNKS,
    periodic_every_chunks: int = PERIODIC_EVERY_CHUNKS,
    recent_label_chunks: int = RECENT_LABEL_CHUNKS,
) -> None:
    if not 0.0 < psi_quantile < 1.0:
        raise ValueError("psi_quantile must be between 0 and 1")
    if (
        error_sd_multiplier < 0
        or confirmation_chunks < 1
        or periodic_every_chunks < 1
        or recent_label_chunks < 1
    ):
        raise ValueError(
            "Error multiplier must be nonnegative and chunk intervals must be positive"
        )
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Run download_cse_cic_ids2018.py first: {MANIFEST_PATH}"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    paths = sorted(
        RAW_DIR.glob("*_TrafficForML_CICFlowMeter.csv"),
        key=lambda path: capture_date(path.name),
    )
    initial_paths = [path for path in paths if capture_date(path.name) == INITIAL_DATE]
    if len(initial_paths) != 1:
        raise ValueError(
            f"Expected one initial capture for {INITIAL_DATE.date()}, found {len(initial_paths)}"
        )
    all_features = feature_columns(initial_paths[0])
    initial, initial_timestamp_exclusions = read_sorted_capture(
        initial_paths[0], all_features
    )
    split = int(len(initial) * INITIAL_TRAIN_FRACTION)
    initial_train, calibration = initial.iloc[:split], initial.iloc[split:]
    core_x, core_y = stratified_cap(
        initial_train, all_features, max_initial_train_rows, seed
    )
    if len(np.unique(core_y)) < 2:
        raise ValueError(
            "Initial chronological training window lacks both benign and attack labels"
        )
    reference_model = fit_reference(core_x, core_y, seed)
    psi_features, psi_reference = psi_setup(core_x)

    calibration_scores = []
    calibration_f1 = []
    for chunk in time_chunks(calibration, chunk_rows):
        calibration_scores.append(psi_score(chunk, psi_features, psi_reference))
        calibration_f1.append(
            evaluate_chunk(reference_model, chunk, all_features)["weighted_f1"]
        )
    if not calibration_scores:
        raise ValueError("The initial capture did not yield calibration chunks")
    psi_threshold = float(np.quantile(calibration_scores, psi_quantile))
    error_threshold = (
        float(
            max(
                0.0,
                np.mean(calibration_f1)
                - error_sd_multiplier * np.std(calibration_f1, ddof=1),
            )
        )
        if len(calibration_f1) > 1
        else 0.0
    )
    del initial_train, calibration, initial
    gc.collect()

    policies = {
        name: {
            "model": clone_model(reference_model),
            "records": [],
            "alerts": [],
            "events": [],
            "updates": 0,
            "update_seconds": [],
            "high_streak": 0,
            "decision_threshold": 0.5,
            "threshold_recalibrations": 0,
            "threshold_seconds": [],
        }
        for name in (
            "no_adaptation",
            "feature_psi",
            "label_error",
            "periodic_5_chunks",
            "threshold_recalibration",
            "threshold_fpr_capped",
        )
    }
    recent: deque[tuple[pd.DataFrame, np.ndarray]] = deque(maxlen=recent_label_chunks)
    recent_threshold_data: deque[tuple[np.ndarray, np.ndarray]] = deque(
        maxlen=recent_label_chunks
    )
    stream_index = 0
    timestamp_exclusions = [
        {"file": initial_paths[0].name, **initial_timestamp_exclusions}
    ]
    for path in paths:
        if path == initial_paths[0]:
            continue
        chunks, day_exclusions = capture_chunks(path, all_features, chunk_rows)
        timestamp_exclusions.append({"file": path.name, **day_exclusions})
        for chunk in chunks:
            stream_index += 1
            if stream_index == 1 or stream_index % 25 == 0:
                print(
                    f"Replay progress: {stream_index} chunks; latest capture {path.name}",
                    flush=True,
                )
            y = chunk["target"].to_numpy(dtype=np.int8)
            score = psi_score(chunk, psi_features, psi_reference)
            contains_attack = bool(y.sum())
            threshold_policy_scores = None
            for name, state in policies.items():
                record = evaluate_chunk(
                    state["model"],
                    chunk,
                    all_features,
                    decision_threshold=state["decision_threshold"],
                )
                positive_probabilities = record.pop("_positive_probabilities")
                if name in {"threshold_recalibration", "threshold_fpr_capped"}:
                    threshold_policy_scores = positive_probabilities
                record["decision_threshold"] = state["decision_threshold"]
                record.update(
                    {
                        "stream_chunk": stream_index,
                        "capture_file": path.name,
                        "feature_psi": score,
                        "contains_attack": contains_attack,
                        "first_attack_timestamp": (
                            chunk.loc[chunk["target"].eq(1), "Timestamp"]
                            .iloc[0]
                            .isoformat()
                            if contains_attack
                            else None
                        ),
                        "drift_alert": False,
                        "update_requested": False,
                        "model_updated_after_chunk": False,
                    }
                )
                state["records"].append(record)
                if name == "feature_psi":
                    state["high_streak"] = (
                        state["high_streak"] + 1 if score > psi_threshold else 0
                    )
                    alert = state["high_streak"] >= confirmation_chunks
                elif name == "label_error":
                    state["high_streak"] = (
                        state["high_streak"] + 1
                        if record["weighted_f1"] < error_threshold
                        else 0
                    )
                    alert = state["high_streak"] >= confirmation_chunks
                elif name == "periodic_5_chunks":
                    alert = stream_index % periodic_every_chunks == 0
                else:
                    alert = False
                if alert:
                    record["update_requested"] = True
                    event = {
                        "stream_chunk": stream_index,
                        "timestamp": record["end_timestamp"],
                        "capture_file": path.name,
                        "trigger": name,
                        "contains_attack": contains_attack,
                        "trigger_score": (
                            score
                            if name == "feature_psi"
                            else (
                                record["weighted_f1"] if name == "label_error" else None
                            )
                        ),
                    }
                    state["events"].append(event)
                    if name in {"feature_psi", "label_error"}:
                        record["drift_alert"] = True
                        state["alerts"].append(event)
            # The just-scored chunk becomes labeled only after its predictions;
            # it may train the next chunk's model, never its own predictions.
            recent.append((chunk[all_features].copy(), y.copy()))
            recent_threshold_data.append((threshold_policy_scores.copy(), y.copy()))
            for name, state in policies.items():
                record = state["records"][-1]
                if name in {"threshold_recalibration", "threshold_fpr_capped"}:
                    started = perf_counter()
                    window_probability = np.concatenate(
                        [part[0] for part in recent_threshold_data]
                    )
                    window_labels = np.concatenate(
                        [part[1] for part in recent_threshold_data]
                    )
                    if name == "threshold_recalibration":
                        state["decision_threshold"] = recalibrate_threshold(
                            window_probability,
                            window_labels,
                            state["decision_threshold"],
                        )
                    else:
                        state["decision_threshold"], guard_diagnostics = (
                            recalibrate_threshold_fpr_capped(
                                window_probability,
                                window_labels,
                                state["decision_threshold"],
                                FPR_CAP,
                            )
                        )
                        record["fpr_guard_validation"] = guard_diagnostics
                    elapsed = perf_counter() - started
                    state["threshold_seconds"].append(elapsed)
                    state["threshold_recalibrations"] += 1
                    record["threshold_after_labels_for_next_chunk"] = state[
                        "decision_threshold"
                    ]
                    record["threshold_recalibration_seconds"] = elapsed
                if record["update_requested"]:
                    state["update_seconds"].append(
                        refit(state["model"], core_x, core_y, recent)
                    )
                    state["updates"] += 1
                    record["model_updated_after_chunk"] = True
                    state["high_streak"] = 0

    summaries = {}
    for name, state in policies.items():
        rows = state["records"]
        counts = {
            key: sum(item[key] for item in rows) for key in ("tn", "fp", "fn", "tp")
        }
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        f1_attack = 2 * counts["tp"] / denominator if denominator else 0.0
        denominator_benign = 2 * counts["tn"] + counts["fp"] + counts["fn"]
        f1_benign = 2 * counts["tn"] / denominator_benign if denominator_benign else 0.0
        attack_support = counts["tp"] + counts["fn"]
        benign_support = counts["tn"] + counts["fp"]
        weighted_f1 = (attack_support * f1_attack + benign_support * f1_benign) / max(
            1, attack_support + benign_support
        )
        event_delays = []
        for filename in dict.fromkeys(record["capture_file"] for record in rows):
            day_rows = [record for record in rows if record["capture_file"] == filename]
            attack_chunks = [
                record for record in day_rows if record["first_attack_timestamp"]
            ]
            if not attack_chunks:
                continue
            onset = datetime.fromisoformat(attack_chunks[0]["first_attack_timestamp"])
            alert = next(
                (
                    item
                    for item in state["alerts"]
                    if item["timestamp"] >= onset.isoformat()
                    and item["capture_file"] == filename
                ),
                None,
            )
            event_delays.append(
                {
                    "capture_file": filename,
                    "first_labeled_attack_timestamp": onset.isoformat(),
                    "first_alert_timestamp": alert["timestamp"] if alert else None,
                    "delay_seconds": (
                        (
                            datetime.fromisoformat(alert["timestamp"]) - onset
                        ).total_seconds()
                        if alert
                        else None
                    ),
                }
            )
        per_alert_recovery = []
        for alert in state["alerts"]:
            alert_index = int(alert["stream_chunk"])
            following = [
                record
                for record in rows
                if alert_index < int(record["stream_chunk"]) <= alert_index + 3
            ]
            per_alert_recovery.append(
                {
                    "alert_chunk": alert_index,
                    "next_three_chunk_mean_weighted_f1": (
                        float(np.mean([record["weighted_f1"] for record in following]))
                        if following
                        else None
                    ),
                }
            )
        model_buffer = BytesIO()
        joblib.dump(state["model"], model_buffer, compress=0)
        summaries[name] = {
            "chunks": len(rows),
            "global_confusion_counts": counts,
            "global_weighted_f1_from_confusion": float(weighted_f1),
            "global_macro_f1_from_confusion": float((f1_attack + f1_benign) / 2),
            "global_false_positive_rate": float(
                counts["fp"] / max(1, counts["fp"] + counts["tn"])
            ),
            "mean_chunk_weighted_f1": (
                float(np.mean([r["weighted_f1"] for r in rows])) if rows else None
            ),
            "mean_chunk_macro_f1": (
                float(np.mean([r["macro_f1"] for r in rows])) if rows else None
            ),
            "mean_chunk_ece": (
                float(np.mean([r["ece"] for r in rows])) if rows else None
            ),
            "mean_chunk_false_positive_rate": (
                float(np.mean([r["false_positive_rate"] for r in rows]))
                if rows
                else None
            ),
            "p95_chunk_prediction_ms_per_row": (
                float(
                    np.percentile(
                        [r["prediction_latency_ms_per_row"] for r in rows], 95
                    )
                )
                if rows
                else None
            ),
            "final_serialized_model_size_mb": float(
                model_buffer.tell() / (1024 * 1024)
            ),
            "benign_only_chunk_mean_weighted_f1": (
                float(
                    np.mean(
                        [r["weighted_f1"] for r in rows if not r["contains_attack"]]
                    )
                )
                if any(not r["contains_attack"] for r in rows)
                else None
            ),
            "attack_containing_chunk_mean_weighted_f1": (
                float(np.mean([r["weighted_f1"] for r in rows if r["contains_attack"]]))
                if any(r["contains_attack"] for r in rows)
                else None
            ),
            "alerts": len(state["alerts"]),
            "false_alerts_on_benign_only_chunks": sum(
                not alert["contains_attack"] for alert in state["alerts"]
            ),
            "first_attack_onset_delay_by_capture_day": event_delays,
            "next_three_chunk_f1_after_alert": per_alert_recovery,
            "model_updates": state["updates"],
            "threshold_recalibrations": state["threshold_recalibrations"],
            "total_threshold_recalibration_seconds": float(
                sum(state["threshold_seconds"])
            ),
            "threshold_recalibration_seconds_summary": {
                "count": len(state["threshold_seconds"]),
                "median": (
                    float(np.median(state["threshold_seconds"]))
                    if state["threshold_seconds"]
                    else 0.0
                ),
                "p95": (
                    float(np.percentile(state["threshold_seconds"], 95))
                    if state["threshold_seconds"]
                    else 0.0
                ),
                "maximum": (
                    float(np.max(state["threshold_seconds"]))
                    if state["threshold_seconds"]
                    else 0.0
                ),
            },
            "update_events": len(state["events"]),
            "update_event_records": state["events"],
            "total_update_seconds": float(sum(state["update_seconds"])),
            "update_seconds": state["update_seconds"],
            "update_time_seconds_summary": {
                "count": len(state["update_seconds"]),
                "median": (
                    float(np.median(state["update_seconds"]))
                    if state["update_seconds"]
                    else 0.0
                ),
                "p95": (
                    float(np.percentile(state["update_seconds"], 95))
                    if state["update_seconds"]
                    else 0.0
                ),
                "maximum": (
                    float(np.max(state["update_seconds"]))
                    if state["update_seconds"]
                    else 0.0
                ),
                "maximum_event": (
                    state["events"][int(np.argmax(state["update_seconds"]))]
                    if state["update_seconds"]
                    else None
                ),
            },
            "alert_records": state["alerts"],
            "chunk_records": rows,
        }
        if name == "threshold_recalibration":
            used_thresholds = np.asarray(
                [item["decision_threshold"] for item in rows], dtype=float
            )
            summaries[name]["decision_threshold_summary"] = {
                "count_scored_chunks": int(len(used_thresholds)),
                "minimum": float(np.min(used_thresholds)),
                "median": float(np.median(used_thresholds)),
                "p95": float(np.percentile(used_thresholds, 95)),
                "maximum": float(np.max(used_thresholds)),
                "fraction_below_0_1": float(np.mean(used_thresholds < 0.1)),
                "fraction_above_0_9": float(np.mean(used_thresholds > 0.9)),
            }
        if name == "threshold_fpr_capped":
            guard_records = [
                item["fpr_guard_validation"]
                for item in rows
                if "fpr_guard_validation" in item
            ]
            chunk_fprs = np.asarray(
                [item["false_positive_rate"] for item in rows], dtype=float
            )
            summaries[name]["recent_window_guard_summary"] = {
                "transitions_with_validation_diagnostics": len(guard_records),
                "fraction_selected_thresholds_meeting_wilson_cap": (
                    float(
                        np.mean(
                            [
                                item["validation_fpr_wilson_upper"] <= FPR_CAP
                                for item in guard_records
                            ]
                        )
                    )
                    if guard_records
                    else None
                ),
                "mean_validation_fpr": (
                    float(np.mean([item["validation_fpr"] for item in guard_records]))
                    if guard_records
                    else None
                ),
                "maximum_validation_wilson_upper": (
                    float(
                        max(
                            item["validation_fpr_wilson_upper"]
                            for item in guard_records
                        )
                    )
                    if guard_records
                    else None
                ),
                "replay_chunks_exceeding_fpr_cap": int(np.sum(chunk_fprs > FPR_CAP)),
                "fraction_replay_chunks_exceeding_fpr_cap": (
                    float(np.mean(chunk_fprs > FPR_CAP)) if len(chunk_fprs) else None
                ),
                "mean_replay_chunk_fpr": (
                    float(np.mean(chunk_fprs)) if len(chunk_fprs) else None
                ),
            }
    omitted_files = [
        str(item.get("name", "unknown"))
        for item in manifest.get("excluded_objects", [])
    ]
    limitations = [
        "Daily CSV source order is not chronological; timestamps are parsed day-first and sorted before replay.",
        "This is a binary attack/benign replay over a CIC testbed schedule, not a production network stream.",
        "Label-based error detection assumes labels arrive at chunk end and is not an unlabeled early-warning detector.",
    ]
    if omitted_files:
        limitations.append(
            "Official source files omitted from this run: " + ", ".join(omitted_files)
        )
    result = {
        "dataset": "CSE-CIC-IDS2018 official processed daily flow CSVs",
        "dataset_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_files": manifest["files"],
        "protocol": {
            "scope": "temporal drift/adaptation only; no AutoML optimization on CSE",
            "initial_capture": initial_paths[0].name,
            "initial_train_fraction_by_sorted_timestamp": INITIAL_TRAIN_FRACTION,
            "initial_training_rows_after_cap": len(core_y),
            "features": all_features,
            "psi_features": psi_features,
            "excluded_columns": sorted(EXCLUDED_FEATURES),
            "target": "binary: Benign vs any non-Benign label",
            "chunk_rows_target": chunk_rows,
            "equal_timestamps_kept_together": True,
            "row_exclusions_for_temporal_validity": timestamp_exclusions,
            "replayed_capture_files": [
                item["file"] for item in timestamp_exclusions[1:]
            ],
            "replayed_rows_after_temporal_exclusions": int(
                sum(record["rows"] for record in policies["no_adaptation"]["records"])
            ),
            "labels_available_at_chunk_end": True,
            "model": "LightGBM 100 estimators, learning_rate=0.05, num_leaves=31, CPU n_jobs=1",
            "adaptation_window": f"initial labeled training sample plus most recent {recent_label_chunks} labeled replay chunks",
            "threshold_recalibration_policy": (
                "After each scored chunk's labels arrive, choose a threshold maximizing macro-F1 on "
                f"the most recent {recent_label_chunks} labeled chunks; apply only to later chunks. "
                "Candidate thresholds are unique values from a 101-point score-quantile grid plus "
                "the current threshold and 0.5. No model refit is performed by this policy."
            ),
            "fpr_capped_threshold_policy": (
                "After each scored chunk's labels arrive, select the threshold maximizing attack recall "
                f"on the most recent {recent_label_chunks} labeled chunks, subject to a 95% Wilson "
                f"upper bound on empirical benign FPR <= {FPR_CAP:.4%}; apply only to later chunks. "
                "The candidate set includes the current threshold, 0.5, a 101-point score-quantile "
                "grid, and a threshold above the maximum past score. No model refit is performed."
            ),
            "fpr_capped_threshold_limit": FPR_CAP,
            "feature_psi_threshold": psi_threshold,
            "feature_psi_calibration_quantile": psi_quantile,
            "feature_psi_confirmation_chunks": confirmation_chunks,
            "label_error_threshold": error_threshold,
            "label_error_calibration_sd_multiplier": error_sd_multiplier,
            "label_error_confirmation_chunks": confirmation_chunks,
            "periodic_update_interval_chunks": periodic_every_chunks,
            "limitations": limitations,
        },
        "calibration": {
            "feature_psi_threshold": psi_threshold,
            "feature_psi_quantile": psi_quantile,
            "weighted_f1_mean": float(np.mean(calibration_f1)),
            "weighted_f1_std": (
                float(np.std(calibration_f1, ddof=1))
                if len(calibration_f1) > 1
                else 0.0
            ),
            "error_threshold": error_threshold,
            "error_sd_multiplier": error_sd_multiplier,
            "calibration_chunks": len(calibration_f1),
        },
        "summaries": summaries,
    }
    source_audit = json.loads(
        (ROOT / "data" / "cse_cic_ids2018" / "audit.json").read_text(encoding="utf-8")
    )
    unique_exclusions = sum(
        max(item.get("repeated_header_rows", 0), item.get("invalid_timestamp_rows", 0))
        + item.get("timestamps_outside_capture_day", 0)
        for item in timestamp_exclusions
    )
    result["run_audit"] = {
        "manifest_file_count": len(manifest["files"]),
        "manifest_raw_rows": int(source_audit["total_rows"]),
        "raw_rows_by_file": {
            item["file"]: int(item["rows"]) for item in source_audit["files"]
        },
        "audit_file": "data/cse_cic_ids2018/audit.json",
        "replayed_capture_count": len(result["protocol"]["replayed_capture_files"]),
        "replayed_chunk_count": int(summaries["no_adaptation"]["chunks"]),
        "replayed_rows": int(
            result["protocol"]["replayed_rows_after_temporal_exclusions"]
        ),
        "unique_rows_excluded": int(unique_exclusions),
        "complete_official_collection": len(manifest["files"]) == 10,
    }
    result["protocol"]["random_seed"] = seed
    result["protocol"]["max_initial_training_rows"] = max_initial_train_rows
    result["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "duckdb": duckdb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "lightgbm": lgb.__version__,
        "cpu_threads_for_model": 1,
        "duckdb_memory_limit": "2GB",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Saved temporal adaptation results to {output_path}")
    for name, summary in summaries.items():
        print(
            name,
            {
                key: summary[key]
                for key in (
                    "chunks",
                    "global_weighted_f1_from_confusion",
                    "mean_chunk_macro_f1",
                    "alerts",
                    "model_updates",
                    "total_update_seconds",
                )
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument(
        "--max-initial-train-rows", type=int, default=MAX_INITIAL_TRAIN_ROWS
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--psi-quantile", type=float, default=0.95)
    parser.add_argument("--error-sd-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--confirmation-chunks", type=int, default=ALERT_CONFIRMATION_CHUNKS
    )
    parser.add_argument(
        "--periodic-every-chunks", type=int, default=PERIODIC_EVERY_CHUNKS
    )
    parser.add_argument("--recent-label-chunks", type=int, default=RECENT_LABEL_CHUNKS)
    args = parser.parse_args()
    main(
        args.seed,
        args.chunk_rows,
        args.max_initial_train_rows,
        args.output,
        args.psi_quantile,
        args.error_sd_multiplier,
        args.confirmation_chunks,
        args.periodic_every_chunks,
        args.recent_label_chunks,
    )
