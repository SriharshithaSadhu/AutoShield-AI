"""Compare row-random and 30-second time-block CV on IoTID20 traffic features."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "provenance" / "IoTID20_official_original.csv"
MANIFEST = ROOT / "data" / "cse_replay_chunk_cache_v2" / "manifest.json"
OUT = ROOT / "results" / "iotid20_row_vs_timeblock_cv_20261103.json"
SEED = 42
N_SPLITS = 5
BLOCK_SECONDS = 30
CONFIG = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "class_weight": "balanced",
    "verbosity": -1,
    "n_jobs": 1,
    "random_state": SEED,
}


def canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    benign_n = tn + fp
    attack_n = tp + fn
    return {
        "rows": int(len(y_true)), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": fp / benign_n if benign_n else None,
        "attack_recall": tp / attack_n if attack_n else None,
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float((tn / benign_n + tp / attack_n) / 2) if benign_n and attack_n else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_cv(X: pd.DataFrame, y: np.ndarray, splitter, groups=None) -> dict[str, object]:
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    fold_results = []
    fold_overlap = []
    fit_seconds = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = LGBMClassifier(**CONFIG)
        started = perf_counter()
        model.fit(X.iloc[train_idx], y[train_idx])
        fit_seconds.append(perf_counter() - started)
        pred = model.predict(X.iloc[test_idx]).astype(np.int8)
        fold_results.append({"fold": fold, **metrics(y[test_idx], pred)})
        all_true.append(y[test_idx])
        all_pred.append(pred)

        train_hashes = set(pd.util.hash_pandas_object(X.iloc[train_idx], index=False).to_numpy(dtype=np.uint64).tolist())
        test_hashes = pd.util.hash_pandas_object(X.iloc[test_idx], index=False).to_numpy(dtype=np.uint64)
        fold_overlap.append({
            "fold": fold,
            "test_rows": int(len(test_idx)),
            "test_rows_with_common77_signature_in_train": int(sum(int(value) in train_hashes for value in test_hashes)),
        })
    y_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)
    return {
        "pooled_out_of_fold": metrics(y_all, pred_all),
        "fold_metrics": fold_results,
        "mean_fold_metrics": {
            key: (float(np.mean(values)) if values else None)
            for key in ("fpr", "attack_recall", "macro_f1", "balanced_accuracy", "accuracy", "precision_attack")
            for values in [[row[key] for row in fold_results if row[key] is not None]]
        },
        "common77_signature_overlap_by_fold": fold_overlap,
        "mean_fit_seconds": float(np.mean(fit_seconds)),
        "total_fit_seconds": float(np.sum(fit_seconds)),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    features = manifest["features"]
    header = pd.read_csv(SOURCE, nrows=0).columns.tolist()
    canonical_map: dict[str, list[str]] = {}
    for name in header:
        canonical_map.setdefault(canonical(name), []).append(name)
    mapping = {name: canonical_map[canonical(name)][0] for name in features
               if len(canonical_map.get(canonical(name), [])) == 1}
    if len(mapping) != 77:
        raise ValueError("Frozen common-77 feature mapping is incomplete or ambiguous")

    usecols = [mapping[name] for name in features] + ["Timestamp", "Label"]
    source = pd.read_csv(SOURCE, usecols=usecols, low_memory=False)
    X = source[[mapping[name] for name in features]].copy()
    X.columns = features
    X = X.replace([np.inf, -np.inf], np.nan)
    y = source["Label"].astype(str).str.strip().str.casefold().eq("anomaly").to_numpy(dtype=np.int8)
    timestamps = pd.to_datetime(source["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Unparseable timestamps in full IoTID20 source")
    epoch_aligned = timestamps.astype("int64").to_numpy(dtype=np.int64) // 1_000_000_000 + 240 * 60
    groups = (epoch_aligned // BLOCK_SECONDS).astype(np.int64)

    row_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    block_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    row_result = evaluate_cv(X, y, row_cv)
    block_result = evaluate_cv(X, y, block_cv, groups)

    result = {
        "status": "complete",
        "date": "2026-11-03",
        "study": "Fixed-LightGBM row-random versus time-block grouped IoTID20 source CV",
        "development_only": True,
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": sha256(SOURCE),
            "rows": int(len(source)),
            "features": len(features),
            "binary_label_rule": "Label=Normal -> benign; Label=Anomaly -> attack",
            "benign_capture_constraint": "All 40,073 Normal rows come from one benign capture; time-block CV splits that single capture into blocks and is not independent-capture generalization.",
        },
        "model": {"estimator": "LightGBM", "config": CONFIG, "preprocessing": "none; tree estimator uses the frozen numeric traffic fields, NaN retained, infinities converted to NaN"},
        "protocol": {
            "folds": N_SPLITS,
            "seed": SEED,
            "row_random": "StratifiedKFold with shuffled rows",
            "time_block": f"StratifiedGroupKFold using groups=floor((CSV timestamp + 240 minutes)/{BLOCK_SECONDS} seconds); all rows in each absolute 30-second interval stay together",
            "feature_map": "The same frozen common-77 features in both arms; no IDs, addresses, ports, or timestamps are model inputs.",
            "group_count": int(np.unique(groups).size),
            "test_membership": "All rows receive one out-of-fold prediction in each arm; no hyperparameter selection uses these folds.",
        },
        "results": {"row_random": row_result, "time_block_grouped": block_result},
        "limits": [
            "30-second time blocks reduce within-window train/test overlap but are not capture-independent: different blocks from one source capture can lie in different folds.",
            "The one-benign-capture design prevents an independent benign capture holdout, so this experiment cannot establish source-domain temporal generalization.",
            "This fixed classifier CV is a protocol sensitivity analysis, not a new IDS method or the original MOPSO AutoML score.",
            "IoTID20 has already been explored for development; a locked independent target remains required for confirmatory claims.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT), "rows": len(source), "time_blocks": int(np.unique(groups).size),
        "row_random": row_result["pooled_out_of_fold"],
        "time_block_grouped": block_result["pooled_out_of_fold"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
