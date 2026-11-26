"""CPU-only daily-batch adaptation case study on CTU-SME-11 Raspberry Pi.

The protocol is matched to the existing Echo case study: train on the first two
daily artifacts, score each of the next five batches before revealing labels,
then compare frozen, expanding-history, and rolling-three-day refits. The result
is a same-collection device case study, not independent-network confirmation.
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
FLOW_DIR = DATA / "raspberrypi_extracted"
ARCHIVE = DATA / "CTU-SME-11_Honeypot-Minicomputer-RaspberryPi-Gen3-20_v1.0.0.tar.bz2"
OUT_JSON = ROOT / "results" / "ctu_raspberrypi_daily_batch_case_study.json"
OUT_MD = ROOT / "CTU_RASPBERRYPI_DAILY_BATCH_CASE_STUDY.md"
SEEDS = (42, 43, 44)
NUMERIC = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts",
           "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
CATEGORICAL = ["proto", "service", "conn_state", "history"]
FEATURES = NUMERIC + CATEGORICAL
TARGETS = {"Benign": 0, "Malicious": 1}
EXPECTED_ARCHIVE_MD5 = "94285412e422f9983b7431c32acaf5b9"


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_zeek(path: Path) -> pd.DataFrame:
    fields = None
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
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
    frame.sort_values("_ts", kind="stable", inplace=True)
    X = frame[FEATURES].copy()
    for column in NUMERIC:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in CATEGORICAL:
        X[column] = X[column].fillna("__MISSING__").astype(str)
    y = frame["label"].map(TARGETS).astype("int8")
    detailed = frame["detailedlabel"].fillna("<missing>").astype(str)
    return pd.concat([X, y.rename("_target"), detailed.rename("_detailedlabel")], axis=1)


def make_model(seed: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        [("num", "passthrough", NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="drop", sparse_threshold=1.0,
    )
    classifier = lgb.LGBMClassifier(
        objective="binary", n_estimators=250, learning_rate=0.05,
        num_leaves=31, max_depth=-1, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1,
        feature_fraction_seed=seed, bagging_seed=seed,
        data_random_seed=seed, random_state=seed, n_jobs=1, verbosity=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def fit(model: Pipeline, batch: pd.DataFrame) -> dict[str, float]:
    wall_start, cpu_start = perf_counter(), process_time()
    model.fit(batch[FEATURES], batch["_target"])
    return {"wall_seconds": perf_counter() - wall_start,
            "process_cpu_seconds": process_time() - cpu_start}


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y_true)), "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "precision_malicious": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_malicious": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_malicious": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else float("nan"),
    }


def main() -> None:
    archive_md5 = hash_file(ARCHIVE, "md5")
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError(f"Raspberry Pi archive checksum mismatch: {archive_md5}")
    paths = sorted(FLOW_DIR.glob("*/zeek/conn.log.labeled"))
    if len(paths) != 7:
        raise ValueError(f"Expected seven daily logs, found {len(paths)}")
    days = [(path.parts[-3], read_zeek(path)) for path in paths]
    if sum(map(lambda item: len(item[1]), days)) != 199_266:
        raise ValueError("Extracted rows do not match the official Raspberry Pi summary")
    for day_name, frame in days:
        if set(frame["_target"].unique()) != {0, 1}:
            raise ValueError(f"Both classes are required in every day batch: {day_name}")

    result = {
        "dataset": {
            "name": "CTU-SME-11 Raspberry Pi honeypot device subset",
            "source_doi": "10.5281/zenodo.7958259",
            "official_archive_md5": archive_md5,
            "daily_log_sha256": {str(p.relative_to(ROOT)): hash_file(p) for p in paths},
            "capture_days": [name for name, _ in days],
            "rows_used": sum(len(frame) for _, frame in days),
            "label_mapping": TARGETS,
            "duration_missing_by_label": {
                "benign_fraction": float(pd.concat([f for _, f in days])
                                          .loc[lambda x: x["_target"].eq(0), "duration"].isna().mean()),
                "malicious_fraction": float(pd.concat([f for _, f in days])
                                             .loc[lambda x: x["_target"].eq(1), "duration"].isna().mean()),
            },
            "scope_limit": "One Raspberry Pi honeypot device from the CTU-SME-11 collection; not independent network confirmation.",
        },
        "protocol": {
            "evaluation_unit": "Completed daily Zeek flow-log artifact",
            "train_days": [days[0][0], days[1][0]],
            "test_days": [name for name, _ in days[2:]],
            "label_delay": "Score each complete daily artifact before revealing that artifact's labels; labels affect only later day batches. No wall-clock flow delay is claimed.",
            "model_features": FEATURES,
            "excluded": ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "label", "detailedlabel", "tunnel_parents", "local_orig", "local_resp"],
            "policies": {
                "static": "Fit once on first two days; no updates.",
                "expanding_daily_refit": "Refit after each scored day on all labels released so far.",
                "rolling_3day_refit": "Refit after each scored day on the latest three labeled days (or all history until three exist).",
            },
            "compute": "CPU-only LightGBM, one thread, fixed configuration; no AutoML or tuning on replay days.",
            "model_config": {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 31, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1},
        },
        "seeds": {},
    }

    for seed in SEEDS:
        seed_results = {}
        initial = pd.concat([days[0][1], days[1][1]], ignore_index=True)
        for policy in ("static", "expanding_daily_refit", "rolling_3day_refit"):
            model = make_model(seed)
            initial_fit = fit(model, initial)
            history = [days[0][1], days[1][1]]
            truth, predictions, per_day, updates = [], [], [], []
            for day_name, batch in days[2:]:
                start = perf_counter()
                pred = model.predict(batch[FEATURES]).astype("int8")
                prediction_seconds = perf_counter() - start
                y = batch["_target"].to_numpy(dtype="int8")
                per_day.append({"day": day_name, "metrics": metric_dict(y, pred),
                                "prediction_wall_seconds": prediction_seconds})
                truth.append(y)
                predictions.append(pred)
                history.append(batch)  # labels are exposed after this batch is scored
                if policy == "expanding_daily_refit":
                    updates.append(fit(model, pd.concat(history, ignore_index=True)))
                elif policy == "rolling_3day_refit":
                    updates.append(fit(model, pd.concat(history[-3:], ignore_index=True)))
            seed_results[policy] = {
                "initial_train_rows": int(len(initial)),
                "initial_fit_wall_seconds": initial_fit["wall_seconds"],
                "initial_fit_process_cpu_seconds": initial_fit["process_cpu_seconds"],
                "update_count": len(updates),
                "update_fit_wall_seconds": [u["wall_seconds"] for u in updates],
                "update_fit_process_cpu_seconds": [u["process_cpu_seconds"] for u in updates],
                "total_update_fit_seconds": float(sum(u["wall_seconds"] for u in updates)),
                "total_update_fit_process_cpu_seconds": float(sum(u["process_cpu_seconds"] for u in updates)),
                "per_day": per_day,
                "pooled_test_metrics": metric_dict(np.concatenate(truth), np.concatenate(predictions)),
                "mean_daily_metrics": {k: float(np.mean([d["metrics"][k] for d in per_day])
                                                   ) for k in ("recall_malicious", "fpr", "macro_f1")},
                "worst_daily_fpr": max(d["metrics"]["fpr"] for d in per_day),
                "worst_daily_fpr_day": max(per_day, key=lambda d: d["metrics"]["fpr"])["day"],
            }
        result["seeds"][str(seed)] = seed_results

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# CTU-SME-11 Raspberry Pi daily-batch adaptation case study", "",
        "## Protocol and interpretation", "",
        "This is a CPU-only, fixed-configuration case study matched to the Echo daily-batch protocol. The first two device-day artifacts train the model; each of the next five artifacts is scored before its labels are released for future updates. It reports no per-flow latency or wall-clock label-delay claim. Raspberry Pi and Echo are devices from the same CTU-SME-11 collection, not independent networks. Three seeds measure model stochasticity, not dataset replication.", "",
        "Rows are explicitly sorted by `ts` within each daily artifact. Model inputs omit timestamps, identifiers, ports, all label fields, and detailed labels. The fixed numeric/categorical schema and one-hot categories are fit using only currently released training batches. Missing numeric duration is retained for LightGBM native handling; its class-conditioned missingness is reported in `data/ctu_sme11/raspberrypi_temporal_audit.json`.", "",
        "## Pooled held-out metrics", "",
        "| Seed | Policy | Pooled malicious recall | Pooled FPR | Pooled macro F1 | Mean daily FPR | Worst daily FPR (day) | Total update fit (s) |", "|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for seed, policies in result["seeds"].items():
        for policy, outcome in policies.items():
            m = outcome["pooled_test_metrics"]
            lines.append(f"| {seed} | {policy} | {m['recall_malicious']:.4f} | {m['fpr']:.4f} | {m['macro_f1']:.4f} | {outcome['mean_daily_metrics']['fpr']:.4f} | {outcome['worst_daily_fpr']:.4f} ({outcome['worst_daily_fpr_day']}) | {outcome['total_update_fit_seconds']:.3f} |")
    lines += ["", "## Per-day results", ""]
    for seed, policies in result["seeds"].items():
        lines += [f"### Seed {seed}", "", "| Policy | Day | Rows | Malicious recall | FPR | Macro F1 | Fit after day (s) |", "|---|---|---:|---:|---:|---:|---:|"]
        for policy, outcome in policies.items():
            for i, day in enumerate(outcome["per_day"]):
                m = day["metrics"]
                fit_s = f"{outcome['update_fit_wall_seconds'][i]:.3f}" if i < len(outcome["update_fit_wall_seconds"]) else "—"
                lines.append(f"| {policy} | {day['day']} | {m['rows']} | {m['recall_malicious']:.4f} | {m['fpr']:.4f} | {m['macro_f1']:.4f} | {fit_s} |")
        lines.append("")
    lines += [
        "## Limitations", "",
        "This is a one-device, seven-day honeypot capture from the same collection as Echo. It has only five held-out days, class prevalence changes sharply over days, and final-flow features are evaluated as complete daily artifacts rather than causal per-flow predictions. The day-level label-release assumption is coarse. Do not generalize beyond CTU-SME-11 or interpret three model seeds as independent replications.", "",
        f"Reproduce with `\.venv\\Scripts\\python.exe run_ctu_raspberrypi_daily_batch_case_study.py`. Machine-readable results and source hashes are in `results/{OUT_JSON.name}`.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    for seed, policies in result["seeds"].items():
        for policy, outcome in policies.items():
            print(seed, policy, json.dumps(outcome["pooled_test_metrics"]),
                  "updates_s=", round(outcome["total_update_fit_seconds"], 3))
    print("Runtime:", platform.platform())


if __name__ == "__main__":
    main()
