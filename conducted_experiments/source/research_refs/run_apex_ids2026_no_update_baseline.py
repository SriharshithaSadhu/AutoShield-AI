"""Locked, feature-restricted CPU reference on APEX-IDS2026 event-time slices.

See APEX_IDS2026_BASELINE_PROTOCOL_2026-11-07.md. This is a screening run,
not a test suite, adaptation experiment, or claim of independently verified IDS
performance. The data file is scanned once to form a seeded per-day/per-label
training reservoir, then once for development calibration and locked evaluation.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "apex_ids2026" / "apex_ids2026_hc_subset.parquet"
OUT = ROOT / "results" / "apex_ids2026_no_update_seed43.json"
MODEL_OUT = ROOT / "results" / "apex_ids2026_no_update_seed43.txt"
PROTOCOL = ROOT / "APEX_IDS2026_BASELINE_PROTOCOL_2026-11-07.md"
BATCH = 500_000
SEED = 43
CAP_PER_DAY_LABEL = 10_000
FPR_CAP = 0.001
MIN_BIN_SUPPORT = 100
DAY_NS = 86_400_000_000_000
FIVE_MIN_NS = 300_000_000_000
TRAIN_START = "2026-06-21"
TRAIN_END = "2026-07-12"
DEV_START = "2026-07-13"
DEV_END = "2026-07-23"
TEST_START = "2026-07-24"
TEST_END = "2026-08-03"
COLS = ["flow_start", "duration_s", "protocol", "packets", "bytes", "tos",
        "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg",
        "label_binary"]
NUMERIC = ["duration_s", "packets", "bytes", "tos", "flag_syn", "flag_ack",
           "flag_fin", "flag_rst", "flag_psh", "flag_urg"]
FEATURE_NAMES = NUMERIC + ["proto_tcp", "proto_udp", "proto_icmp", "proto_other"]


def make_features(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    cols: list[np.ndarray] = []
    coercions = 0
    for name in NUMERIC:
        raw = frame[name]
        values = pd.to_numeric(raw, errors="coerce")
        coercions += int((values.isna() & raw.notna()).sum())
        cols.append(values.to_numpy(dtype=np.float32, na_value=np.nan))
    proto = frame["protocol"].astype("string").str.upper()
    for name in ("TCP", "UDP", "ICMP"):
        cols.append((proto == name).fillna(False).to_numpy(dtype=np.float32))
    cols.append((~proto.isin(["TCP", "UDP", "ICMP"])).fillna(True).to_numpy(dtype=np.float32))
    return np.column_stack(cols).astype(np.float32, copy=False), coercions


def utc_event_ns(frame: pd.DataFrame) -> np.ndarray:
    start = pd.to_datetime(frame["flow_start"], utc=True, errors="coerce")
    duration = pd.to_numeric(frame["duration_s"], errors="coerce")
    end = start + pd.to_timedelta(duration, unit="s")
    return end.astype("datetime64[ns, UTC]").astype("int64", copy=False).to_numpy()


def split_masks(day_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    date = pd.to_datetime(day_ids * DAY_NS, unit="ns", utc=True).strftime("%Y-%m-%d")
    train = (date >= TRAIN_START) & (date <= TRAIN_END)
    dev = (date >= DEV_START) & (date <= DEV_END)
    test = (date >= TEST_START) & (date <= TEST_END)
    return train, dev, test


def iter_batches_near_dates(pf: pq.ParquetFile, first_day: str, last_day: str):
    """Use verified monotonic row-group date statistics to avoid unrelated days.

    The upstream `date` partition can lag/lead the event timestamp by a day for
    audited boundary rows, so the scan range is widened by one day on each side;
    actual assignment always uses flow_start + duration_s below.
    """
    date_col = pf.schema.names.index("date")
    first = pd.Timestamp(first_day).date() - pd.Timedelta(days=1)
    last = pd.Timestamp(last_day).date() + pd.Timedelta(days=1)
    for rg in range(pf.metadata.num_row_groups):
        stats = pf.metadata.row_group(rg).column(date_col).statistics
        if stats is None or not stats.has_min_max:
            yield from pf.iter_batches(batch_size=BATCH, columns=COLS, use_threads=True,
                                       row_groups=[rg])
            continue
        if stats.max < first:
            continue
        if stats.min > last:
            break
        yield from pf.iter_batches(batch_size=BATCH, columns=COLS, use_threads=True,
                                   row_groups=[rg])


def fit_reservoir(pf: pq.ParquetFile) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    rng = np.random.default_rng(SEED)
    samples: dict[tuple[int, int], np.ndarray] = {}
    labels: dict[tuple[int, int], np.ndarray] = {}
    observed: dict[tuple[int, int], int] = defaultdict(int)
    coercions: dict[str, int] = defaultdict(int)
    rows_seen = 0
    for batch in iter_batches_near_dates(pf, TRAIN_START, TRAIN_END):
        frame = batch.to_pandas()
        event_ns = utc_event_ns(frame)
        valid_time = event_ns >= 0
        day_ids = event_ns // DAY_NS
        in_train, _, _ = split_masks(day_ids)
        take = valid_time & in_train
        if not take.any():
            continue
        x, n_coerce = make_features(frame.loc[take])
        coercions["numeric_nonparseable"] += n_coerce
        y = pd.to_numeric(frame.loc[take, "label_binary"], errors="coerce").to_numpy(dtype=np.int8)
        d = day_ids[take]
        rows_seen += int(take.sum())
        keys = np.unique(np.column_stack([d, y]), axis=0)
        for day, label in keys:
            key = (int(day), int(label))
            idx = np.flatnonzero((d == day) & (y == label))
            new_x, new_y = x[idx], y[idx]
            observed[key] += len(idx)
            if key in samples:
                new_x = np.vstack([samples[key], new_x])
                new_y = np.concatenate([labels[key], new_y])
            if len(new_y) > CAP_PER_DAY_LABEL:
                keep = rng.choice(len(new_y), CAP_PER_DAY_LABEL, replace=False)
                new_x, new_y = new_x[keep], new_y[keep]
            samples[key], labels[key] = new_x, new_y
        if rows_seen and rows_seen % 5_000_000 < BATCH:
            print(f"training-prefix rows encountered: {rows_seen:,}", flush=True)
    if not samples:
        raise RuntimeError("No training rows landed in the preregistered prefix.")
    x_train = np.vstack(list(samples.values()))
    y_train = np.concatenate(list(labels.values()))
    kept: dict[str, int] = defaultdict(int)
    total_by_day: dict[str, int] = defaultdict(int)
    for (day, label), ys in labels.items():
        day_text = pd.to_datetime(day * DAY_NS, unit="ns", utc=True).strftime("%Y-%m-%d")
        kept[f"{day_text}|{label}"] = int(len(ys))
        total_by_day[day_text] += int(observed[(day, label)])
    return x_train, y_train, dict(kept), {"train_prefix_rows": rows_seen, **dict(total_by_day)}


def confusion(scores: np.ndarray, y: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    pred = scores > threshold
    y_pos = y == 1
    tp = int(np.count_nonzero(pred & y_pos))
    fp = int(np.count_nonzero(pred & ~y_pos))
    fn = int(np.count_nonzero(~pred & y_pos))
    tn = int(np.count_nonzero(~pred & ~y_pos))
    return tn, fp, fn, tp


def rates(c: tuple[int, int, int, int]) -> dict[str, float | int]:
    tn, fp, fn, tp = c
    fpr = fp / (fp + tn) if fp + tn else None
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    tpr = recall
    tnr = tn / (tn + fp) if tn + fp else None
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "benign_support": tn + fp,
            "attack_support": fn + tp, "fpr": fpr, "recall": recall, "precision": precision,
            "balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None}


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    test_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    pf = pq.ParquetFile(DATA)
    x_train, y_train, kept, observed = fit_reservoir(pf)
    train_wall, train_cpu = time.perf_counter(), time.process_time()
    model = lgb.LGBMClassifier(objective="binary", n_estimators=200, num_leaves=31,
                               learning_rate=0.05, max_depth=-1, min_child_samples=20,
                               reg_lambda=1.0, random_state=SEED, n_jobs=4,
                               deterministic=True, force_col_wise=True, verbosity=-1)
    model.fit(x_train, y_train, feature_name=FEATURE_NAMES)
    fit_wall, fit_cpu = time.perf_counter(), time.process_time()

    dev_benign_scores: list[np.ndarray] = []
    dev_all_scores: list[np.ndarray] = []
    dev_all_labels: list[np.ndarray] = []
    total_dev = total_test = 0
    for batch in iter_batches_near_dates(pf, DEV_START, TEST_END):
        frame = batch.to_pandas()
        event_ns = utc_event_ns(frame)
        valid_time = event_ns >= 0
        day_ids = event_ns // DAY_NS
        _, is_dev, is_test = split_masks(day_ids)
        for label_period, mask in (("dev", is_dev), ("test", is_test)):
            mask &= valid_time
            if not mask.any():
                continue
            x, _ = make_features(frame.loc[mask])
            y = pd.to_numeric(frame.loc[mask, "label_binary"], errors="coerce").to_numpy(dtype=np.int8)
            event = event_ns[mask]
            scores = model.predict_proba(x, num_threads=4)[:, 1].astype(np.float32, copy=False)
            if label_period == "dev":
                dev_benign_scores.append(scores[y == 0])
                dev_all_scores.append(scores)
                dev_all_labels.append(y)
                total_dev += len(y)
            else:
                # Row order is irrelevant after assigning each completed flow to
                # its UTC 5-minute event-time bin; aggregate after threshold lock.
                test_cache.append((event // FIVE_MIN_NS, y, scores))
                total_test += len(y)
        print(f"scored dev={total_dev:,}, locked-test={total_test:,}", flush=True)

    if not dev_benign_scores:
        raise RuntimeError("No development-period benign records found.")
    benign = np.concatenate(dev_benign_scores)
    benign.sort()
    dev_benign_rows = int(len(benign))
    allowed_fp = int(np.floor(FPR_CAP * len(benign)))
    # Strict score > threshold. The (allowed_fp)-th largest score is a
    # conservative cut: ties at the cut are all treated as negative.
    idx = max(0, len(benign) - allowed_fp - 1)
    threshold = float(benign[idx])
    dev_conf = np.asarray(confusion(np.concatenate(dev_all_scores), np.concatenate(dev_all_labels), threshold),
                          dtype=np.int64)
    del dev_all_scores, dev_all_labels
    del dev_benign_scores, benign

    # Aggregate all evaluation batches in one vectorized pass; this avoids both
    # row-order assumptions and quadratic per-bin mask scans.
    test_bins: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    test_days: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    test_total_conf = np.zeros(4, dtype=np.int64)
    bin_parts, y_parts, score_parts = zip(*test_cache)
    bin_ids = np.concatenate(bin_parts)
    y_test = np.concatenate(y_parts)
    score_test = np.concatenate(score_parts)
    pred = score_test > threshold
    y_pos = y_test == 1
    test_total_conf = np.array([np.count_nonzero(~pred & ~y_pos), np.count_nonzero(pred & ~y_pos),
                                np.count_nonzero(~pred & y_pos), np.count_nonzero(pred & y_pos)], dtype=np.int64)
    bin_keys, bin_inverse = np.unique(bin_ids, return_inverse=True)
    for j, condition in enumerate((~pred & ~y_pos, pred & ~y_pos, ~pred & y_pos, pred & y_pos)):
        test_bin_values = np.bincount(bin_inverse, weights=condition.astype(np.int64), minlength=len(bin_keys)).astype(np.int64)
        for i, value in enumerate(test_bin_values):
            test_bins[int(bin_keys[i])][j] = value
    day_ids = (bin_ids * FIVE_MIN_NS) // DAY_NS
    day_keys, day_inverse = np.unique(day_ids, return_inverse=True)
    day_matrix = np.zeros((len(day_keys), 4), dtype=np.int64)
    for j, condition in enumerate((~pred & ~y_pos, pred & ~y_pos, ~pred & y_pos, pred & y_pos)):
        day_matrix[:, j] = np.bincount(day_inverse, weights=condition.astype(np.int64), minlength=len(day_keys)).astype(np.int64)
    for i, day in enumerate(day_keys):
        test_days[pd.to_datetime(int(day) * DAY_NS, unit="ns", utc=True).strftime("%Y-%m-%d")] = day_matrix[i]
    del test_cache, bin_parts, y_parts, score_parts, bin_ids, y_test, score_test
    eligible = [(k, c) for k, c in test_bins.items()
                if c[0] + c[1] >= MIN_BIN_SUPPORT and c[2] + c[3] >= MIN_BIN_SUPPORT]
    eligible_rates = [(float(rates(tuple(int(v) for v in c))["fpr"]), k, c) for k, c in eligible]
    eligible_over = sum(rate > FPR_CAP for rate, _, _ in eligible_rates)
    worst = max(eligible_rates, key=lambda z: z[0]) if eligible_rates else None
    model.booster_.save_model(str(MODEL_OUT))
    serialized_bytes = MODEL_OUT.stat().st_size
    end_wall, end_cpu = time.perf_counter(), time.process_time()

    day_records = {k: rates(tuple(int(v) for v in c)) for k, c in sorted(test_days.items())}
    bin_records = []
    for k, c in sorted(test_bins.items()):
        record = rates(tuple(int(v) for v in c))
        record["bin_start_utc"] = pd.to_datetime(k * FIVE_MIN_NS, unit="ns", utc=True).isoformat()
        record["eligible_for_worst_bin"] = (record["benign_support"] >= MIN_BIN_SUPPORT and
                                               record["attack_support"] >= MIN_BIN_SUPPORT)
        bin_records.append(record)
    result = {
        "protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_version": 1,
        "dataset": str(DATA.relative_to(ROOT)), "dataset_rows": int(pf.metadata.num_rows),
        "dataset_sha256": "7e7855964ddc8061804fa48acfd6c2a3e1891462466654f9294cdab58744730c",
        "row_group_prefilter": {"partition_date_min_max_monotonic": True,
                                "scan_margin_days_each_side": 1,
                                "max_duration_seconds_from_parquet_row_group_statistics": 59.9900016784668,
                                "event_assignment_uses": "flow_start + duration_s; partition date is only a conservative row-group pruning aid"},
        "task_endpoint": "Attack_Verified vs Benign_Verified within retrospectively selected HC subset",
        "event_time": "UTC flow_start + duration_s proxy; exporter-arrival timestamps absent",
        "split": {"train": [TRAIN_START, TRAIN_END], "development": [DEV_START, DEV_END],
                  "locked_evaluation": [TEST_START, TEST_END]},
        "feature_names": FEATURE_NAMES,
        "excluded_feature_groups": ["IPs", "ports and port categories", "timestamps and IDs", "label/taxonomy/evidence/intelligence", "Zeek/source availability", "precomputed rates and label proxies"],
        "sampling": {"seed": SEED, "cap_per_training_date_label": CAP_PER_DAY_LABEL,
                     "retained_per_date_label": kept, "observed_rows_by_training_date": observed},
        "model": {"name": "LightGBM binary classifier", "objective": "binary", "n_estimators": 200,
                  "num_leaves": 31, "learning_rate": 0.05, "max_depth": -1,
                  "min_child_samples": 20, "reg_lambda": 1.0, "random_state": SEED,
                  "threads": 4, "device": "CPU", "model_bytes": serialized_bytes},
        "threshold_calibration": {"development_benign_rows": dev_benign_rows,
                                  "target_pooled_fpr": FPR_CAP, "frozen_score_threshold_strict_gt": threshold,
                                  "calibration_method": "empirical development benign score order statistic; no guarantee under time shift"},
        "development_locked_threshold_metrics": rates(tuple(int(v) for v in dev_conf)),
        "evaluation": {"rows": int(total_test), "metrics": rates(tuple(int(v) for v in test_total_conf)),
                       "daily": day_records, "five_minute_bins": bin_records,
                       "five_minute_bin_minimum_support_per_class": MIN_BIN_SUPPORT,
                       "eligible_bin_count": len(eligible), "eligible_bins_over_fpr_cap": eligible_over,
                       "eligible_bin_share_over_fpr_cap": eligible_over / len(eligible) if eligible else None,
                       "worst_eligible_bin": None if worst is None else {
                           "bin_start_utc": pd.to_datetime(worst[1] * FIVE_MIN_NS, unit="ns", utc=True).isoformat(),
                           **rates(tuple(int(v) for v in worst[2]))}},
        "compute": {"train_sample_rows": int(len(y_train)), "fit_wall_seconds": fit_wall - train_wall,
                    "fit_process_cpu_seconds": fit_cpu - train_cpu,
                    "training_scan_and_reservoir_wall_seconds": train_wall - started_wall,
                    "training_scan_and_reservoir_process_cpu_seconds": train_cpu - started_cpu,
                    "total_wall_seconds": end_wall - started_wall,
                    "total_process_cpu_seconds": end_cpu - started_cpu,
                    "post_fit_evaluation_aggregation_and_save_wall_seconds": end_wall - fit_wall,
                    "post_fit_evaluation_aggregation_and_save_process_cpu_seconds": end_cpu - fit_cpu,
                    "scored_test_rows": int(total_test), "scored_development_rows": int(total_dev)},
        "interpretation": "One fixed-model, one-seed screening baseline. The final interval is one temporal segment from a single ISP and a selected label subset; no adaptation or novelty claim is supported by this run alone.",
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} and {MODEL_OUT.relative_to(ROOT)}", flush=True)
    print(json.dumps({"threshold": threshold, "dev": result["development_locked_threshold_metrics"],
                      "evaluation": result["evaluation"]["metrics"],
                      "eligible_bins": len(eligible), "eligible_over_cap": eligible_over,
                      "worst_bin": result["evaluation"]["worst_eligible_bin"],
                      "fit_wall_s": result["compute"]["fit_wall_seconds"],
                      "total_wall_s": result["compute"]["total_wall_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
