"""Run the locked fixed-model, flow-completion-ordered NF-v3 replay.

Labels are read for a chunk only after both arms have produced that chunk's
scores. The periodic arm refits after every fifth completed chunk using labels
from the most recent three completed chunks plus the fixed initial sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nf_ton_iot_v3"
SOURCE = DATA / "NF-ToN-IoT-v3_endtime_ordered.csv"
SCHEMA_PATH = ROOT / "results" / "nf_ton_iot_v3_model_schema_audit.json"
OUT = ROOT / "results" / "nf_ton_iot_v3_fixed_model_replay_seed42.json"
PARTIAL = OUT.with_suffix(".partial.json")
SAMPLE_INDEX = ROOT / "results" / "nf_ton_iot_v3_replay_seed42_initial_sample_rows.csv"
MODEL_DIR = ROOT / "results" / "nf_ton_iot_v3_replay_models"

SEED = 42
FIT_END_EXCLUSIVE_MS = 1_556_042_400_000  # 2019-04-23 18:00:00 UTC
REPLAY_START_MS = 1_556_064_000_000     # 2019-04-24 00:00:00 UTC
TRAIN_SAMPLE_PER_CLASS = 250_000
REPLAY_CHUNK_ROWS = 100_000
READ_ROWS = 200_000
REFIT_EVERY_CHUNKS = 5
RECENT_LABELED_CHUNKS = 3
FPR_CAPS = {"fpr_0p1_percent": 0.001, "fpr_1_percent": 0.01}
CATEGORICAL = [
    "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
]
EXCLUDED_FIELDS = {
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Label", "Attack", "DNS_QUERY_ID",
}
EXPECTED_REPLAY_ROWS = 21_677_227
EXPECTED_REPLAY_LABELS = {"0": 11_211_388, "1": 10_465_839}
EXPECTED_FAMILY_SUPPORT = {
    "Backdoor": 203_384,
    "ddos": 4_141_256,
    "dos": 203_456,
    "injection": 381_777,
    "mitm": 6_013,
    "password": 1_594_777,
    "ransomware": 3_971,
    "scanning": 1_096_770,
    "xss": 2_834_435,
}


def input_frame(values: np.ndarray, features: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=features)
    for col in CATEGORICAL:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype("int32")
    for col in features:
        if col not in CATEGORICAL:
            frame[col] = pd.to_numeric(frame[col], errors="raise").astype("float32")
    return frame


def frame_to_array(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = frame.loc[:, features].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Candidate model features contain missing/nonfinite values")
    for col in CATEGORICAL:
        j = features.index(col)
        if not np.equal(values[:, j], np.floor(values[:, j])).all():
            raise ValueError(f"Categorical code field {col} contains noninteger values")
    return values.astype(np.float32, copy=False)


def update_reservoir(state: dict, class_label: int, candidate_x: np.ndarray,
                     candidate_ids: np.ndarray, rng: np.random.Generator) -> None:
    if not len(candidate_x):
        return
    new_priorities = rng.random(len(candidate_x))
    if class_label in state:
        old_x, old_ids, old_priorities = state[class_label]
        all_x = np.concatenate((old_x, candidate_x), axis=0)
        all_ids = np.concatenate((old_ids, candidate_ids), axis=0)
        all_priorities = np.concatenate((old_priorities, new_priorities), axis=0)
    else:
        all_x, all_ids, all_priorities = candidate_x, candidate_ids, new_priorities
    quota = TRAIN_SAMPLE_PER_CLASS
    if len(all_priorities) > quota:
        keep = np.argpartition(all_priorities, quota - 1)[:quota]
        all_x, all_ids, all_priorities = all_x[keep], all_ids[keep], all_priorities[keep]
    state[class_label] = (all_x, all_ids, all_priorities)


def make_pipeline(features: list[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[("codes_one_hot", OneHotEncoder(
            handle_unknown="ignore", sparse_output=True, dtype=np.float32), CATEGORICAL)],
        remainder="passthrough",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )
    estimator = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        max_bin=127,
        n_jobs=1,
        random_state=SEED,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    return Pipeline([("preprocess", preprocessing), ("classifier", estimator)])


def threshold_for_fpr(benign_scores: np.ndarray, cap: float) -> tuple[float, int]:
    scores = np.asarray(benign_scores, dtype=np.float64)
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Need finite calibration scores from benign rows")
    allowed = int(np.floor(cap * len(scores)))
    ascending = np.sort(scores)
    threshold = float(ascending[len(scores) - allowed - 1])
    realized = int(np.count_nonzero(scores > threshold))
    if realized > allowed:
        raise RuntimeError("Strict score threshold exceeded calibration FPR allowance")
    return threshold, realized


def counts_for(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    actual = np.asarray(y_true, dtype=np.int8)
    predicted = np.asarray(scores) > threshold
    positive = actual == 1
    negative = ~positive
    tp = int(np.count_nonzero(predicted & positive))
    fn = int(np.count_nonzero(~predicted & positive))
    fp = int(np.count_nonzero(predicted & negative))
    tn = int(np.count_nonzero(~predicted & negative))
    attack_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    benign_f1 = 2 * tn / max(1, 2 * tn + fp + fn)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "false_positive_rate": fp / max(1, fp + tn),
        "attack_recall": tp / max(1, tp + fn),
        "attack_precision": tp / max(1, tp + fp),
        "attack_f1": attack_f1,
        "benign_f1": benign_f1,
        "macro_f1": (attack_f1 + benign_f1) / 2,
        "weighted_f1": ((tp + fn) * attack_f1 + (tn + fp) * benign_f1) / max(1, tp + tn + fp + fn),
    }


def add_counts(target: dict, value: dict) -> None:
    for key in ("tn", "fp", "fn", "tp"):
        target[key] += int(value[key])


def counts_metrics(counts: dict) -> dict:
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    attack_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    benign_f1 = 2 * tn / max(1, 2 * tn + fp + fn)
    return {
        **counts,
        "false_positive_rate": fp / max(1, fp + tn),
        "attack_recall": tp / max(1, tp + fn),
        "attack_precision": tp / max(1, tp + fp),
        "attack_f1": attack_f1,
        "benign_f1": benign_f1,
        "macro_f1": (attack_f1 + benign_f1) / 2,
        "weighted_f1": ((tp + fn) * attack_f1 + (tn + fp) * benign_f1) / max(1, tp + tn + fp + fn),
    }


def utc_string(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def iter_replay_chunks(features: list[str]):
    usecols = features + ["FLOW_END_MILLISECONDS", "Label", "Attack"]
    carry = None
    staging = None
    previous_chunk_end = None
    last_seen_end = None
    chunk_no = 0

    def take_ready(buffer: pd.DataFrame, final: bool = False):
        nonlocal previous_chunk_end, chunk_no
        ready = []
        while len(buffer) >= REPLAY_CHUNK_ROWS or (final and len(buffer)):
            if len(buffer) <= REPLAY_CHUNK_ROWS:
                cut = len(buffer)
            else:
                cut = REPLAY_CHUNK_ROWS
                boundary = int(buffer["FLOW_END_MILLISECONDS"].iloc[cut - 1])
                end_values = buffer["FLOW_END_MILLISECONDS"].to_numpy(dtype=np.int64)
                while cut < len(buffer) and int(end_values[cut]) == boundary:
                    cut += 1
            part = buffer.iloc[:cut].copy()
            buffer = buffer.iloc[cut:].reset_index(drop=True)
            if not len(part):
                continue
            first_end = int(part["FLOW_END_MILLISECONDS"].iloc[0])
            last_end = int(part["FLOW_END_MILLISECONDS"].iloc[-1])
            if previous_chunk_end is not None and first_end == previous_chunk_end:
                raise RuntimeError("Equal flow-end group was split across replay chunks")
            if previous_chunk_end is not None and first_end < previous_chunk_end:
                raise RuntimeError("Replay chunk order regressed in flow-end time")
            previous_chunk_end = last_end
            chunk_no += 1
            ready.append((chunk_no, part, first_end, last_end))
        return ready, buffer

    for block in pd.read_csv(SOURCE, usecols=usecols, chunksize=READ_ROWS,
                             low_memory=False):
        end_values = pd.to_numeric(block["FLOW_END_MILLISECONDS"], errors="raise").astype("int64")
        block["FLOW_END_MILLISECONDS"] = end_values
        if last_seen_end is not None and len(block) and int(end_values.iloc[0]) < last_seen_end:
            raise RuntimeError("Input end-time ordering regressed during replay")
        if len(block):
            last_seen_end = int(end_values.iloc[-1])
        block = block.loc[end_values >= REPLAY_START_MS].reset_index(drop=True)
        if not len(block):
            continue
        if carry is not None and len(carry):
            block = pd.concat((carry, block), ignore_index=True)
            carry = None
        final_end = int(block["FLOW_END_MILLISECONDS"].iloc[-1])
        safe = block.loc[block["FLOW_END_MILLISECONDS"] < final_end]
        carry = block.loc[block["FLOW_END_MILLISECONDS"] == final_end].copy()
        if staging is None:
            staging = safe.reset_index(drop=True)
        elif len(safe):
            staging = pd.concat((staging, safe), ignore_index=True)
        if staging is not None:
            ready, staging = take_ready(staging, final=False)
            for item in ready:
                yield item
    if staging is None:
        staging = pd.DataFrame(columns=usecols)
    if carry is not None and len(carry):
        staging = pd.concat((staging, carry), ignore_index=True)
    ready, _leftover = take_ready(staging, final=True)
    for item in ready:
        yield item


def main() -> None:
    global SEED, OUT, PARTIAL, SAMPLE_INDEX
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the initial per-class sample and LightGBM")
    args = parser.parse_args()
    SEED = args.seed
    OUT = ROOT / "results" / f"nf_ton_iot_v3_fixed_model_replay_seed{SEED}.json"
    PARTIAL = OUT.with_suffix(".partial.json")
    SAMPLE_INDEX = ROOT / "results" / f"nf_ton_iot_v3_replay_seed{SEED}_initial_sample_rows.csv"
    if OUT.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {OUT}; use --force after review")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if schema["rows"] != 27_520_260 or schema["candidate_feature_count"] != 46:
        raise RuntimeError("Schema audit does not match locked protocol")
    if any(v["missing"] or v["non_numeric"] or v["nonfinite_count"] or v["constant_or_single_valued"]
           for v in schema["feature_statistics"].values()):
        raise RuntimeError("Candidate feature schema gate failed")
    features = schema["candidate_features"]
    if set(CATEGORICAL) - set(features) or set(features) & EXCLUDED_FIELDS:
        raise RuntimeError("Frozen feature/exclusion contract failed")

    rngs = {label: np.random.default_rng(SEED + label) for label in (0, 1)}
    reservoirs = {}
    calibration_x_parts = []
    calibration_y_parts = []
    fit_window_counts = Counter()
    calibration_counts = Counter()
    rows_read = 0
    started = time.perf_counter()
    print("Collecting deterministic initial sample and calibration set...", flush=True)
    first_pass_cols = features + ["FLOW_END_MILLISECONDS", "Label"]
    for frame in pd.read_csv(SOURCE, usecols=first_pass_cols, chunksize=READ_ROWS, low_memory=False):
        ends = pd.to_numeric(frame["FLOW_END_MILLISECONDS"], errors="raise").astype("int64").to_numpy()
        labels = pd.to_numeric(frame["Label"], errors="raise").astype("int8").to_numpy()
        source_rows = rows_read + np.arange(len(frame), dtype=np.int64)
        fit_mask = ends < FIT_END_EXCLUSIVE_MS
        calibration_mask = (ends >= FIT_END_EXCLUSIVE_MS) & (ends < REPLAY_START_MS)
        if fit_mask.any():
            fit_x_all = frame.loc[fit_mask, features]
            fit_y_all = labels[fit_mask]
            fit_ids_all = source_rows[fit_mask]
            fit_x = frame_to_array(fit_x_all, features)
            for label in (0, 1):
                mask = fit_y_all == label
                fit_window_counts[str(label)] += int(mask.sum())
                update_reservoir(reservoirs, label, fit_x[mask], fit_ids_all[mask], rngs[label])
        if calibration_mask.any():
            cal_x = frame_to_array(frame.loc[calibration_mask, features], features)
            cal_y = labels[calibration_mask]
            calibration_x_parts.append(cal_x)
            calibration_y_parts.append(cal_y.copy())
            calibration_counts.update(str(int(y)) for y in cal_y)
        rows_read += len(frame)
        if rows_read % 1_000_000 < READ_ROWS:
            print(f"initial scan rows={rows_read:,}", flush=True)
        if len(ends) and int(ends[-1]) >= REPLAY_START_MS:
            break

    if any(label not in reservoirs or len(reservoirs[label][0]) != TRAIN_SAMPLE_PER_CLASS for label in (0, 1)):
        raise RuntimeError("Insufficient examples for locked initial training sample")
    train_x_parts, train_y_parts, train_id_parts = [], [], []
    for label in (0, 1):
        x, row_ids, _priority = reservoirs[label]
        train_x_parts.append(x)
        train_y_parts.append(np.full(len(x), label, dtype=np.int8))
        train_id_parts.append(row_ids)
    train_x = np.concatenate(train_x_parts, axis=0)
    train_y = np.concatenate(train_y_parts)
    train_ids = np.concatenate(train_id_parts)
    order = np.argsort(train_ids, kind="stable")
    train_x, train_y, train_ids = train_x[order], train_y[order], train_ids[order]
    x_fit = input_frame(train_x, features)

    cal_x = np.concatenate(calibration_x_parts, axis=0)
    cal_y = np.concatenate(calibration_y_parts)
    if len(cal_x) != sum(calibration_counts.values()):
        raise RuntimeError("Calibration row accounting mismatch")

    SAMPLE_INDEX.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source_row_zero_based": train_ids, "Label": train_y}).to_csv(SAMPLE_INDEX, index=False)
    sample_hash = hashlib.sha256(train_ids.astype("<i8", copy=False).tobytes()).hexdigest()
    sample_feature_hash = hashlib.sha256(train_x.astype("<f4", copy=False).tobytes()).hexdigest()

    model_frozen = make_pipeline(features)
    model_periodic = make_pipeline(features)
    fit_start = time.perf_counter()
    model_frozen.fit(x_fit, train_y)
    initial_fit_seconds = time.perf_counter() - fit_start
    model_periodic.fit(x_fit, train_y)
    initial_fit_seconds_periodic_copy = time.perf_counter() - fit_start - initial_fit_seconds
    print(f"initial fit complete in {initial_fit_seconds:.2f}s; scoring calibration", flush=True)

    x_cal_frame = input_frame(cal_x, features)
    cal_start = time.perf_counter()
    cal_scores = model_frozen.predict_proba(x_cal_frame)[:, 1]
    calibration_predict_seconds = time.perf_counter() - cal_start
    benign_scores = cal_scores[cal_y == 0]
    thresholds, calibration_threshold_results = {}, {}
    for name, cap in FPR_CAPS.items():
        threshold, false_alarms = threshold_for_fpr(benign_scores, cap)
        thresholds[name] = threshold
        calibration_threshold_results[name] = {
            "cap": cap,
            "benign_rows": int(len(benign_scores)),
            "allowed_false_alarms": int(np.floor(cap * len(benign_scores))),
            "observed_false_alarms": false_alarms,
            "observed_fpr": false_alarms / len(benign_scores),
            "score_threshold": threshold,
        }
    del cal_x, x_cal_frame, cal_scores, benign_scores

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    initial_model_path = MODEL_DIR / f"seed{SEED}_initial_frozen.joblib"
    joblib.dump(model_frozen, initial_model_path, compress=3)

    pooled = {(arm, cap): Counter({"tn": 0, "fp": 0, "fn": 0, "tp": 0})
              for arm in ("frozen", "periodic_refit") for cap in FPR_CAPS}
    by_day = defaultdict(Counter)
    by_family = defaultdict(Counter)
    chunk_records = []
    fit_records = []
    prediction_latency_ms_per_row = {"frozen": [], "periodic_refit": []}
    recent: deque[tuple[pd.DataFrame, np.ndarray]] = deque(maxlen=RECENT_LABELED_CHUNKS)
    replay_rows = 0
    replay_labels = Counter()
    replay_family_counts = Counter()
    replay_chunks = 0
    previous_output_end = None
    replay_started = time.perf_counter()

    for chunk_no, frame, first_end, last_end in iter_replay_chunks(features):
        if previous_output_end is not None and first_end == previous_output_end:
            raise RuntimeError("Equal completion-time rows were split between chunks")
        previous_output_end = last_end
        x_values = frame_to_array(frame, features)
        x_frame = input_frame(x_values, features)
        predictions = {}
        latency = {}
        for arm, model in (("frozen", model_frozen), ("periodic_refit", model_periodic)):
            before = time.perf_counter()
            predictions[arm] = model.predict_proba(x_frame)[:, 1]
            latency[arm] = time.perf_counter() - before
            prediction_latency_ms_per_row[arm].append(1000 * latency[arm] / len(frame))

        # Label release point: no target label is read until both arms have scored the chunk.
        y = pd.to_numeric(frame["Label"], errors="raise").to_numpy(dtype=np.int8)
        family = frame["Attack"].astype("string").fillna("<MISSING>").str.strip().to_numpy()
        if not np.isin(y, [0, 1]).all():
            raise RuntimeError(f"Nonbinary replay label in chunk {chunk_no}")
        y_counts = Counter(str(int(value)) for value in y)
        replay_labels.update(y_counts)
        replay_rows += len(y)
        replay_chunks += 1
        chunk_date = datetime.fromtimestamp(last_end / 1000, timezone.utc).date().isoformat()
        families_in_chunk = Counter(str(value) for value in family)
        replay_family_counts.update({k: v for k, v in families_in_chunk.items() if k != "Benign"})
        record = {
            "chunk": chunk_no,
            "rows": int(len(frame)),
            "start_end_utc": utc_string(first_end),
            "last_end_utc": utc_string(last_end),
            "date_utc": chunk_date,
            "benign_rows": int(y_counts.get("0", 0)),
            "attack_rows": int(y_counts.get("1", 0)),
            "family_support": {k: int(v) for k, v in sorted(families_in_chunk.items())},
            "arms": {},
        }
        for arm in ("frozen", "periodic_refit"):
            record["arms"][arm] = {
                "prediction_seconds": latency[arm],
                "prediction_ms_per_row": 1000 * latency[arm] / len(frame),
                "threshold_metrics": {},
            }
            for cap_name, threshold in thresholds.items():
                metrics = counts_for(y, predictions[arm], threshold)
                record["arms"][arm]["threshold_metrics"][cap_name] = metrics
                aggregate_counts = {key: 0 for key in ("tn", "fp", "fn", "tp")}
                add_counts(aggregate_counts, metrics)
                add_counts(pooled[(arm, cap_name)], aggregate_counts)
                add_counts(by_day[(chunk_date, arm, cap_name)], aggregate_counts)
                for fam in np.unique(family):
                    fam_mask = family == fam
                    fam_counts = counts_for(y[fam_mask], predictions[arm][fam_mask], threshold)
                    add_counts(by_family[(str(fam), arm, cap_name)], fam_counts)
        chunk_records.append(record)

        # Only after post-score evaluation are chunk labels admitted to the update buffer.
        recent.append((x_frame.copy(), y.copy()))
        if chunk_no % REFIT_EVERY_CHUNKS == 0:
            recent_x = pd.concat([part[0] for part in recent], ignore_index=True)
            recent_y = np.concatenate([part[1] for part in recent])
            update_x = pd.concat([x_fit, recent_x], ignore_index=True)
            update_y = np.concatenate((train_y, recent_y))
            update_start = time.perf_counter()
            model_periodic = make_pipeline(features)
            model_periodic.fit(update_x, update_y)
            fit_seconds = time.perf_counter() - update_start
            fit_record = {
                "after_chunk": chunk_no,
                "labels_released_through_chunk": chunk_no,
                "predicts_from_chunk": chunk_no + 1,
                "fit_seconds": fit_seconds,
                "training_rows": int(len(update_y)),
                "training_label_counts": {str(int(k)): int(v) for k, v in Counter(update_y).items()},
                "recent_labeled_chunks": min(chunk_no, RECENT_LABELED_CHUNKS),
            }
            fit_records.append(fit_record)
            del recent_x, recent_y, update_x, update_y

        if chunk_no == 1 or chunk_no % 10 == 0:
            print(f"replay_chunks={chunk_no} rows={replay_rows:,} "
                  f"date={chunk_date} periodic_updates={len(fit_records)} "
                  f"elapsed_s={time.perf_counter()-replay_started:.1f}", flush=True)
        if chunk_no % 20 == 0:
            PARTIAL.write_text(json.dumps({
                "status": "running",
                "completed_chunks": chunk_no,
                "rows_scored": replay_rows,
                "labels_scored": dict(replay_labels),
                "chunk_records": chunk_records,
                "fit_records": fit_records,
                "thresholds": calibration_threshold_results,
            }, indent=2), encoding="utf-8")
        del x_values, x_frame, predictions, y, family, frame

    if replay_rows != EXPECTED_REPLAY_ROWS:
        raise RuntimeError(f"Replay rows {replay_rows:,} != expected {EXPECTED_REPLAY_ROWS:,}")
    if dict(replay_labels) != EXPECTED_REPLAY_LABELS:
        raise RuntimeError(f"Replay labels differ: {dict(replay_labels)}")
    if dict(replay_family_counts) != EXPECTED_FAMILY_SUPPORT:
        raise RuntimeError(f"Attack-family support differs: {dict(replay_family_counts)}")

    summaries = {
        f"{arm}:{cap}": counts_metrics(dict(counts))
        for (arm, cap), counts in pooled.items()
    }
    day_summary = [
        {"date_utc": day, "arm": arm, "threshold": cap, **counts_metrics(dict(counts))}
        for (day, arm, cap), counts in sorted(by_day.items())
    ]
    family_summary = [
        {"attack_or_benign_family": family_name, "arm": arm, "threshold": cap,
         "support": int(counts["tn"] + counts["fn"] + counts["fp"] + counts["tp"]),
         **counts_metrics(dict(counts))}
        for (family_name, arm, cap), counts in sorted(by_family.items())
    ]
    update_model_path = MODEL_DIR / f"seed{SEED}_periodic_final.joblib"
    joblib.dump(model_periodic, update_model_path, compress=3)
    model_file_sizes = {
        "initial_frozen_bytes": initial_model_path.stat().st_size,
        "final_periodic_refit_bytes": update_model_path.stat().st_size,
    }
    result = {
        "status": "complete",
        "protocol": "NF_TON_IOT_V3_REPLAY_PROTOCOL_2026-11-05.md",
        "seed": SEED,
        "development_only": True,
        "experiment": "fixed LightGBM frozen versus periodic label-released supervised refit on NF-ToN-IoT-v3 flow-end stream",
        "source_sha256": json.loads((ROOT / "results" / "nf_ton_iot_v3_endtime_ordered_manifest.json").read_text(encoding="utf-8"))["output_sha256"],
        "schema_audit": SCHEMA_PATH.relative_to(ROOT).as_posix(),
        "rows": replay_rows,
        "labels": dict(replay_labels),
        "attack_family_support": dict(replay_family_counts),
        "feature_count": len(features),
        "features": features,
        "excluded_fields": sorted(EXCLUDED_FIELDS),
        "one_hot_code_fields": CATEGORICAL,
        "initial_fit_window": {"end_exclusive_utc": "2019-04-23T18:00:00+00:00", "rows": int(sum(fit_window_counts.values())), "label_counts": dict(fit_window_counts)},
        "initial_sample": {
            "strategy": "seeded per-class random-priority reservoir, then ordered by source-row index",
            "seed": SEED,
            "sample_per_class": TRAIN_SAMPLE_PER_CLASS,
            "rows": int(len(train_y)),
            "label_counts": {str(int(k)): int(v) for k, v in Counter(train_y).items()},
            "row_id_file": SAMPLE_INDEX.relative_to(ROOT).as_posix(),
            "row_ids_sha256": sample_hash,
            "feature_matrix_sha256_float32": sample_feature_hash,
        },
        "calibration": {
            "window": "2019-04-23T18:00:00+00:00 to 2019-04-24T00:00:00+00:00",
            "rows": int(len(cal_y)),
            "label_counts": dict(calibration_counts),
            "calibration_prediction_seconds": calibration_predict_seconds,
            "thresholds_from_benign_scores_only": calibration_threshold_results,
        },
        "model": {
            "class": "lightgbm.LGBMClassifier",
            "parameters": {
                "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31,
                "min_child_samples": 20, "max_bin": 127, "n_jobs": 1,
                "random_state": SEED, "deterministic": True, "force_col_wise": True,
            },
            "initial_fit_seconds_frozen": initial_fit_seconds,
            "initial_fit_seconds_periodic_copy": initial_fit_seconds_periodic_copy,
            "model_file_sizes": model_file_sizes,
            "versions": {"python": sys.version.split()[0], "lightgbm": lgb.__version__,
                          "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "replay_protocol": {
            "start_utc": "2019-04-24T00:00:00+00:00",
            "target_chunk_rows": REPLAY_CHUNK_ROWS,
            "equal_flow_end_rows_kept_together": True,
            "adaptation": "refit after every fifth scored chunk using fixed initial sample plus last three fully labeled chunks",
            "labels_for_current_chunk_used_for_current_prediction": False,
            "model_thresholds_change_during_replay": False,
        },
        "chunk_count": replay_chunks,
        "pooled_metrics": summaries,
        "per_day_metrics": day_summary,
        "per_family_metrics": family_summary,
        "per_chunk_metrics": chunk_records,
        "updates": fit_records,
        "prediction_latency_ms_per_row_by_chunk": {
            arm: values for arm, values in prediction_latency_ms_per_row.items()
        },
        "wall_seconds_total_including_initial_sample_fit_and_replay": time.perf_counter() - started,
        "notes": [
            "This is a single-seed development replay on an already inspected schedule, not blind confirmation.",
            "The collection belongs to the same ToN-IoT testbed family and the attacks are scheduled campaigns, not natural production drift.",
            "The publisher's timestamp-plus-five-tuple label join is not independently re-annotated in this experiment.",
            "No AutoML search or optimizer comparison was conducted on NF-v3.",
            "Any pooled gain must be interpreted alongside per-day/family false-positive rates, label budget, and measured refit time."
        ],
        "wall_seconds_after_replay_start": time.perf_counter() - replay_started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    PARTIAL.unlink(missing_ok=True)
    print(json.dumps({"status": result["status"], "rows": replay_rows,
                      "chunks": replay_chunks, "updates": len(fit_records),
                      "pooled_metrics": summaries, "output": OUT.relative_to(ROOT).as_posix()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
