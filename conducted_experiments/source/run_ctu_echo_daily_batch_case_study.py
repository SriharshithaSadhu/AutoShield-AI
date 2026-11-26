"""Causal day-batch CTU-Echo case study with chunk-end label release.

This is a fixed-model CPU experiment, not a full AutoML search. Each daily
completed-flow batch is scored before that day's labels enter any update.
Predictions are day-end batch decisions; this script does not claim per-flow
real-time availability or detection latency.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter, process_time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ctu_sme11"
FLOW_DIR = DATA / "extracted"
OUT_JSON = ROOT / "results" / "ctu_echo_daily_batch_case_study.json"
OUT_MD = ROOT / "CTU_ECHO_DAILY_BATCH_CASE_STUDY.md"
SEEDS = (42, 43, 44)
NUMERIC = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts",
           "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
CATEGORICAL = ["proto", "service", "conn_state", "history"]
FEATURES = NUMERIC + CATEGORICAL
TARGETS = {"Benign": 0, "Malicious": 1}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_zeek(path: Path) -> pd.DataFrame:
    fields = None
    with path.open("r", encoding="utf-8", errors="strict") as f:
        for line in f:
            if line.startswith("#fields\t"):
                fields = line.rstrip("\r\n").split("\t")[1:]
                break
    if not fields:
        raise ValueError(f"Missing Zeek schema in {path}")
    frame = pd.read_csv(path, sep="\t", comment="#", names=fields,
                        na_values=["-", "(empty)"], keep_default_na=True,
                        low_memory=False)
    unknown = set(frame["label"].dropna().astype(str)) - set(TARGETS)
    if unknown:
        raise ValueError(f"Unknown labels are never silently mapped: {sorted(unknown)}")
    frame = frame.loc[frame["label"].isin(TARGETS)].copy()
    frame["_ts"] = pd.to_numeric(frame["ts"], errors="coerce")
    if frame["_ts"].isna().any():
        raise ValueError(f"Unparseable timestamps in {path}")
    # Sorting provides deterministic order within a batch; the timestamp is
    # not passed to the model and no within-day prediction is simulated.
    frame = frame.sort_values("_ts", kind="stable").reset_index(drop=True)
    X = frame[FEATURES].copy()
    for column in NUMERIC:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in CATEGORICAL:
        X[column] = X[column].fillna("__MISSING__").astype(str)
    y = frame["label"].map(TARGETS).astype("int8")
    detailed = frame["detailedlabel"].fillna("<missing>").astype(str)
    return pd.concat([X, y.rename("_target"), detailed.rename("_detailedlabel")], axis=1)


def make_model(seed: int) -> Pipeline:
    preprocess = ColumnTransformer(
        [("num", "passthrough", NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="drop",
        sparse_threshold=1.0,
    )
    classifier = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction_seed=seed,
        bagging_seed=seed,
        data_random_seed=seed,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def fit(model: Pipeline, data: pd.DataFrame) -> dict[str, float]:
    wall_started = perf_counter()
    cpu_started = process_time()
    model.fit(data[FEATURES], data["_target"])
    return {"wall_seconds": perf_counter() - wall_started,
            "process_cpu_seconds": process_time() - cpu_started}


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y_true)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision_malicious": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_malicious": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_malicious": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else float("nan"),
    }


def main() -> None:
    archive = DATA / "CTU-SME-11_Honeypot-Assistant-Amazon-Echo1stGen-1_v1.0.0.tar.bz2"
    archive_md5 = md5(archive)
    if archive_md5 != "311cbd100eeddf76c8056919fe35b0cb":
        raise ValueError("Echo archive MD5 is not the official verified value")
    days = []
    paths = sorted(FLOW_DIR.glob("*/zeek/conn.log.labeled"))
    if len(paths) != 7:
        raise ValueError(f"Expected seven daily logs, found {len(paths)}")
    for path in paths:
        days.append((path.parts[-3], read_zeek(path)))
    if any(set(day[1]["_target"].unique()) != {0, 1} for day in days):
        raise ValueError("Every daily batch must contain both binary labels")

    results = {"dataset": {}, "protocol": {}, "seeds": {}}
    dataset_rows = sum(len(frame) for _, frame in days)
    results["dataset"] = {
        "name": "CTU-SME-11 Amazon Echo honeypot device subset",
        "source_doi": "10.5281/zenodo.7958259",
        "official_archive_md5": archive_md5,
        "daily_log_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "capture_days": [day for day, _ in days],
        "rows_used": dataset_rows,
        "label_mapping": TARGETS,
        "scope_limit": "One Echo honeypot device in one seven-day CTU-SME-11 environment; all malicious rows share one detailed Discovery label.",
    }
    results["protocol"] = {
        "evaluation_unit": "Completed daily Zeek flow-log batch",
        "train_days": [days[0][0], days[1][0]],
        "test_days": [day for day, _ in days[2:]],
        "label_delay": "Each daily artifact is scored as one batch before that batch's labels are released; updates affect the next batch. No precise wall-clock label delay is claimed.",
        "model_features": FEATURES,
        "excluded": ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
                     "label", "detailedlabel", "tunnel_parents", "local_orig", "local_resp"],
        "missing_data": "LightGBM native numeric missing-value handling; missing categorical fields receive an explicit token; all one-hot categories are fit on available training days only.",
        "policies": {
            "static": "Fit once on first two days, score all five later days without updating.",
            "expanding_daily_refit": "After each scored day and label release, refit on all available labeled days.",
            "rolling_3day_refit": "After each scored day and label release, refit on the latest three labeled days (or all history until three days exist).",
        },
        "operational_scope": "Predictions are available at the day-batch evaluation point, not at individual flow start or end; no real-time latency claim.",
        "compute": "CPU only; one LightGBM thread; no hyperparameter search or resampling.",
        "model_config": {"objective": "binary", "n_estimators": 250,
                         "learning_rate": 0.05, "num_leaves": 31,
                         "feature_fraction": 0.8, "bagging_fraction": 0.8,
                         "bagging_freq": 1},
    }

    for seed in SEEDS:
        seed_results = {}
        for policy in ("static", "expanding_daily_refit", "rolling_3day_refit"):
            initial = pd.concat([days[0][1], days[1][1]], ignore_index=True)
            model = make_model(seed)
            initial_fit = fit(model, initial)
            history = [days[0][1], days[1][1]]
            y_true_all, y_pred_all = [], []
            per_day = []
            update_fit = []
            for day_name, test_batch in days[2:]:
                pred_start = perf_counter()
                y_pred = model.predict(test_batch[FEATURES]).astype("int8")
                prediction_s = perf_counter() - pred_start
                y_true = test_batch["_target"].to_numpy(dtype="int8")
                benign_support = test_batch.loc[y_true == 0, "_detailedlabel"].value_counts()
                false_positive_counts = test_batch.loc[(y_true == 0) & (y_pred == 1), "_detailedlabel"].value_counts()
                benign_error_breakdown = {
                    str(label): {
                        "benign_rows": int(count),
                        "false_positives": int(false_positive_counts.get(label, 0)),
                        "fpr": float(false_positive_counts.get(label, 0) / count),
                    }
                    for label, count in benign_support.items()
                }
                per_day.append({"day": day_name, "metrics": scores(y_true, y_pred),
                                "predict_batch_seconds": prediction_s,
                                "false_positive_rate_by_benign_detailed_label": benign_error_breakdown,
                                "false_positive_detailed_labels": {
                                    str(k): int(v) for k, v in
                                    test_batch.loc[(y_true == 0) & (y_pred == 1), "_detailedlabel"]
                                    .value_counts().items()
                                }})
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)
                # Label reveal occurs only after this day has been scored.
                history.append(test_batch)
                if policy == "expanding_daily_refit":
                    fit_data = pd.concat(history, ignore_index=True)
                    update_fit.append(fit(model, fit_data))
                elif policy == "rolling_3day_refit":
                    fit_data = pd.concat(history[-3:], ignore_index=True)
                    update_fit.append(fit(model, fit_data))
            seed_results[policy] = {
                "seed": seed,
                "initial_train_rows": len(initial),
                "initial_fit_wall_seconds": initial_fit["wall_seconds"],
                "initial_fit_process_cpu_seconds": initial_fit["process_cpu_seconds"],
                "update_fit_wall_seconds": [fit_result["wall_seconds"] for fit_result in update_fit],
                "update_fit_process_cpu_seconds": [fit_result["process_cpu_seconds"] for fit_result in update_fit],
                "total_update_fit_seconds": float(sum(x["wall_seconds"] for x in update_fit)),
                "total_update_fit_process_cpu_seconds": float(sum(x["process_cpu_seconds"] for x in update_fit)),
                "update_count": len(update_fit),
                "per_day": per_day,
                "pooled_test_metrics": scores(np.concatenate(y_true_all), np.concatenate(y_pred_all)),
                "mean_daily_metrics": {
                    key: float(np.mean([day["metrics"][key] for day in per_day]))
                    for key in ("recall_malicious", "fpr", "macro_f1")
                },
                "worst_daily_fpr": max(day["metrics"]["fpr"] for day in per_day),
                "worst_daily_fpr_day": max(per_day, key=lambda day: day["metrics"]["fpr"])["day"],
            }
        results["seeds"][str(seed)] = seed_results

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# CTU-SME-11 Echo daily-batch adaptation case study",
        "",
        "## Protocol and interpretation",
        "",
        "This CPU-only exploratory experiment trains on the first two capture-day batches, scores each of the next five complete daily flow-log batches, and reveals that batch's labels only after scoring, with no precise wall-clock label delay claimed. The static model never updates; expanding and rolling-three-day policies refit only on labels already released. Predictions are day-batch evaluations, not per-flow real-time detections. Echo is one device capture and every malicious row has the same detailed Discovery class; results cannot establish network-wide robustness or general attack detection.",
        "",
        "Features exclude timestamps, identifiers, ports, and all label fields. Numeric missing values are handled natively by LightGBM; categorical preprocessing is fit inside each training call. Three model seeds are reported as algorithmic sensitivity, not independent dataset replications.",
        "",
        "## Pooled held-out metrics",
        "",
        "| Seed | Policy | Pooled malicious recall | Pooled FPR | Pooled macro F1 | Mean daily FPR | Worst daily FPR (day) | Total update fit (s) |",
        "|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for seed, policies in results["seeds"].items():
        for policy, outcome in policies.items():
            m = outcome["pooled_test_metrics"]
            lines.append(f"| {seed} | {policy} | {m['recall_malicious']:.4f} | {m['fpr']:.4f} | {m['macro_f1']:.4f} | {outcome['mean_daily_metrics']['fpr']:.4f} | {outcome['worst_daily_fpr']:.4f} ({outcome['worst_daily_fpr_day']}) | {outcome['total_update_fit_seconds']:.3f} |")
    lines += ["", "## Per-day results", ""]
    for seed, policies in results["seeds"].items():
        lines += [f"### Seed {seed}", "", "| Policy | Day | Rows | Malicious recall | FPR | Macro F1 | Fit after day (s) |", "|---|---|---:|---:|---:|---:|---:|"]
        for policy, outcome in policies.items():
            for i, day in enumerate(outcome["per_day"]):
                m = day["metrics"]
                fit_s = f"{outcome['update_fit_wall_seconds'][i]:.3f}" if i < len(outcome["update_fit_wall_seconds"]) else "—"
                lines.append(f"| {policy} | {day['day']} | {m['rows']} | {m['recall_malicious']:.4f} | {m['fpr']:.4f} | {m['macro_f1']:.4f} | {fit_s} |")
        lines.append("")
    representative = results["seeds"][str(SEEDS[0])]["static"]
    focal_day = next(day for day in representative["per_day"] if day["day"] == "2023-02-23")
    lines += [
        "## Benign false-positive attribution on 2023-02-23 (static, seed 42)",
        "",
        "| Benign detailed label | Benign rows | False positives | Within-label FPR |",
        "|---|---:|---:|---:|",
    ]
    for label, group in sorted(focal_day["false_positive_rate_by_benign_detailed_label"].items(),
                               key=lambda item: item[1]["false_positives"], reverse=True):
        lines.append(f"| {label} | {group['benign_rows']} | {group['false_positives']} | {group['fpr']:.4f} |")
    lines.append("")
    lines += [
        "## Limitations",
        "",
        "This is a single chronological device capture with five held-out days and one malicious detailed-label family. It is descriptive; no inferential claim is made from the three model seeds. Day-batch evaluation introduces batch-level decision timing without a precise wall-clock availability claim, and the experiment does not measure packet-level or per-flow availability. CTU's flow-start timestamp and label-conditioned missing durations remain a bar to causal per-flow replay. The result is a small external case study only.",
        "",
        f"Reproduce with `\.venv\\Scripts\\python.exe run_ctu_echo_daily_batch_case_study.py`. Raw metrics and source hashes are in `results/{OUT_JSON.name}`.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print("Pooled results:")
    for seed, policies in results["seeds"].items():
        for policy, outcome in policies.items():
            print(seed, policy, json.dumps(outcome["pooled_test_metrics"]),
                  "update_fit_s=", round(outcome["total_update_fit_seconds"], 3))
    print("Runtime:", platform.platform())


if __name__ == "__main__":
    main()
