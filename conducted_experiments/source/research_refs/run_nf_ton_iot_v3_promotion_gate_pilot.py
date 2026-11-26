"""Real-data pilot comparing update gates on the NF-ToN-IoT-v3 stream.

This is a deliberately narrow development experiment. Candidates are
trained only on already-labeled, strictly earlier chunks. The newest completed
chunk is held out for the promotion decision; it never enters that candidate's
training matrix. All arms use the same candidate, fixed score threshold, update
schedule, and future chunks. The flow end timestamp is an event-time lower bound
on availability, not the exporter emission timestamp.

Methods: always promote (periodic baseline), BRIDGE's one-candidate bootstrap
lower-bound rule (balanced accuracy), the VBC-SG stratified EB confidence-sequence
decision logic, and a project FPR-cap + paired recall-improvement rule. The last
is a feasibility pilot, not a proven guarantee: its binomial/McNemar assumptions
are not established for autocorrelated, drifting network flows.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "nf_ton_iot_v3"
BASELINE_SCRIPT = ROOT / "research_refs" / "run_nf_ton_iot_v3_fixed_model_replay.py"
BASE = RESULTS / "nf_ton_iot_v3_fixed_model_replay_seed42.json"
SAMPLE = RESULTS / "nf_ton_iot_v3_replay_seed42_initial_sample_rows.csv"
MODEL = RESULTS / "nf_ton_iot_v3_replay_models" / "seed42_initial_frozen.joblib"
OUT = RESULTS / "nf_ton_iot_v3_promotion_gate_comparison_seed42.json"
PARTIAL = RESULTS / "nf_ton_iot_v3_promotion_gate_comparison_seed42.partial.json"
SUMMARY = RESULTS / "nf_ton_iot_v3_promotion_gate_comparison_seed42.md"
BRIDGE_SRC = (ROOT / "research_refs" / "bridge_gate_artifact_2026-10-28" /
              "DwiSunaryono-BRIDGE-ML-f26f2ee" / "pkg" / "src")
BRIDGE_REPLICATES = 20_000
UPDATE_EVERY = 5
RECENT_TRAIN_CHUNKS = 3
GATES = ("periodic", "bridge", "vbc_sg", "fpr_recall")


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("nf_ton_baseline", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the existing replay helper")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_bridge_core():
    sys.path.insert(0, str(BRIDGE_SRC))
    from bridge_gate import core
    return core


def cs_lower_bound_eb(d: np.ndarray, alpha: float) -> float:
    """Exact BRIDGE/VBC-SG artifact implementation (MIT; see code-audit note)."""
    x = (np.asarray(d, float) + 1.0) / 2.0
    n = len(x)
    if not n:
        return -math.inf
    loginv = math.log(1.0 / alpha)
    num = den = sx = sv = 0.0
    for i, xi in enumerate(x, start=1):
        muhat = (0.5 + sx) / i
        sighat = (0.25 + sv) / i
        lam = min(0.75, math.sqrt(2.0 * loginv /
                                   max(sighat * i * math.log(i + 1), 1e-12)))
        v = (float(xi) - muhat) ** 2
        psi = -math.log(1.0 - lam) - lam
        num += lam * float(xi) - psi * v
        den += lam
        sx += float(xi)
        sv += v
    return 2.0 * ((num - loginv) / den) - 1.0


def vbc_sg_decision(y: np.ndarray, incumbent: np.ndarray, candidate: np.ndarray,
                    rng: np.random.Generator, alpha: float = 0.10,
                    seq_block: int = 16, max_steps: int = 5) -> dict:
    """VBC-SG's stratified, without-replacement cohort probe on this causal holdout.

    The full current validation chunk is a later-time probe than candidate training.
    For each step, the rule draws 16 examples per class without replacement and
    applies the source's alpha/4 empirical-Bernstein CS commit/reject split.
    Unresolved proposals are conservatively rejected at the five-step cap.
    """
    y = np.asarray(y, dtype=np.int8)
    inc = np.asarray(incumbent, dtype=np.int8)
    cand = np.asarray(candidate, dtype=np.int8)
    pools = {c: np.flatnonzero(y == c) for c in (0, 1)}
    if any(len(v) == 0 for v in pools.values()):
        return {"accept": False, "decision": "reject_single_class", "labels": 0,
                "steps": 0, "lower_ba_gain": None, "upper_ba_gain": None}
    for c in (0, 1):
        rng.shuffle(pools[c])
    used = {0: 0, 1: 0}
    diffs = {0: [], 1: []}
    lower = upper = None
    for step in range(1, max_steps + 1):
        for c in (0, 1):
            lo = used[c]
            hi = min(lo + seq_block, len(pools[c]))
            ix = pools[c][lo:hi]
            used[c] = hi
            if len(ix):
                d = ((cand[ix] == y[ix]).astype(float)
                     - (inc[ix] == y[ix]).astype(float))
                diffs[c].extend(d.tolist())
        if any(not diffs[c] for c in (0, 1)):
            continue
        lbs = [cs_lower_bound_eb(np.asarray(diffs[c]), alpha / 4.0) for c in (0, 1)]
        ubs = [-cs_lower_bound_eb(-np.asarray(diffs[c]), alpha / 4.0) for c in (0, 1)]
        lower, upper = float(0.5 * sum(lbs)), float(0.5 * sum(ubs))
        if lower > 0.0:
            return {"accept": True, "decision": "commit", "labels": sum(map(len, diffs.values())),
                    "steps": step, "lower_ba_gain": lower, "upper_ba_gain": upper}
        if upper < 0.0:
            return {"accept": False, "decision": "reject", "labels": sum(map(len, diffs.values())),
                    "steps": step, "lower_ba_gain": lower, "upper_ba_gain": upper}
    return {"accept": False, "decision": "defer_cap_reject",
            "labels": sum(map(len, diffs.values())), "steps": max_steps,
            "lower_ba_gain": lower, "upper_ba_gain": upper}


def fpr_recall_gate(y: np.ndarray, incumbent: np.ndarray, candidate: np.ndarray,
                    alpha: float, fpr_cap: float = 0.001) -> dict:
    """Pilot rule: exact-binomial benign-FPR UCB plus one-sided paired recall test."""
    y = np.asarray(y, dtype=np.int8)
    inc = np.asarray(incumbent, dtype=np.int8)
    cand = np.asarray(candidate, dtype=np.int8)
    benign = y == 0
    attacks = y == 1
    n_benign = int(benign.sum())
    fp = int(np.count_nonzero((cand == 1) & benign))
    fpr_ucb = (1.0 if fp == n_benign else
               float(beta.ppf(1.0 - alpha, fp + 1, n_benign - fp)))
    better = int(np.count_nonzero(attacks & (cand == 1) & (inc == 0)))
    worse = int(np.count_nonzero(attacks & (cand == 0) & (inc == 1)))
    discordant = better + worse
    p_recall = (1.0 if discordant == 0 else
                float(binomtest(better, discordant, 0.5, alternative="greater").pvalue))
    fpr_ok = bool(fpr_ucb <= fpr_cap)
    recall_ok = bool(better > worse and p_recall < alpha)
    return {
        "accept": fpr_ok and recall_ok,
        "decision": "commit" if fpr_ok and recall_ok else "reject",
        "fpr_cap": fpr_cap, "benign_n": n_benign, "false_positives": fp,
        "fpr_ucb": fpr_ucb, "alpha_per_endpoint": alpha,
        "recall_better": better, "recall_worse": worse,
        "recall_gain_p_one_sided": p_recall,
        "fpr_pass": fpr_ok, "recall_gain_pass": recall_ok,
        "assumption_note": "Exact binomial/McNemar calculations are nominal here; serial dependence and future drift invalidate a general guarantee unless separately controlled.",
    }


def score_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    return load_baseline_module().counts_for(y, scores, threshold)


def extract_initial_training(baseline, features: list[str], expected: dict):
    sample = pd.read_csv(SAMPLE)
    row_ids = sample["source_row_zero_based"].to_numpy(dtype=np.int64)
    sample_y = sample["Label"].to_numpy(dtype=np.int8)
    if not np.all(np.diff(row_ids) > 0):
        raise RuntimeError("Stored initial sample row IDs are not strictly ordered")
    found_x: list[np.ndarray] = []
    found_y: list[np.ndarray] = []
    read_rows = 0
    usecols = features + ["Label"]
    # Row IDs were generated while scanning the verified end-time-ordered
    # replay source, not the publisher's start-time-ordered CSV.
    for frame in pd.read_csv(baseline.SOURCE, usecols=usecols,
                             chunksize=200_000, low_memory=False):
        lo, hi = read_rows, read_rows + len(frame)
        left = int(np.searchsorted(row_ids, lo, side="left"))
        right = int(np.searchsorted(row_ids, hi, side="left"))
        if right > left:
            offsets = row_ids[left:right] - lo
            x = baseline.frame_to_array(frame.iloc[offsets][features], features)
            y = pd.to_numeric(frame.iloc[offsets]["Label"], errors="raise").to_numpy(dtype=np.int8)
            found_x.append(x)
            found_y.append(y)
        read_rows = hi
        if read_rows > int(row_ids[-1]):
            break
        if read_rows % 1_000_000 < 200_000:
            print(f"initial end-ordered row extraction={read_rows:,}", flush=True)
    if sum(len(x) for x in found_x) != len(row_ids):
        raise RuntimeError("Could not recover every stored initial training row")
    train_x = np.concatenate(found_x)
    train_y = np.concatenate(found_y)
    if not np.array_equal(train_y, sample_y):
        raise RuntimeError("Recovered feature rows do not match stored training labels")
    import hashlib
    row_hash = hashlib.sha256(row_ids.astype("<i8", copy=False).tobytes()).hexdigest()
    feature_hash = hashlib.sha256(train_x.astype("<f4", copy=False).tobytes()).hexdigest()
    if row_hash != expected["initial_sample"]["row_ids_sha256"]:
        raise RuntimeError("Initial sample row-ID checksum differs from locked replay")
    if feature_hash != expected["initial_sample"]["feature_matrix_sha256_float32"]:
        raise RuntimeError("Initial sample feature checksum differs from locked replay")
    return baseline.input_frame(train_x, features), train_y, read_rows


def main() -> None:
    global OUT, PARTIAL, SUMMARY
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, choices=(42, 43), default=42,
                        help="use an existing locked seed-42 or seed-43 model/replay")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="development pilot truncation; 0 runs the full fixed stream")
    parser.add_argument("--shadow-previous", action="store_true")
    args = parser.parse_args()
    stem = (f"nf_ton_iot_v3_promotion_gate_smoke{args.max_chunks}_seed{args.seed}"
            if args.max_chunks else f"nf_ton_iot_v3_promotion_gate_comparison_seed{args.seed}")
    if args.shadow_previous:
        stem += "_shadow_previous"
    OUT = RESULTS / f"{stem}.json"
    PARTIAL = RESULTS / f"{stem}.partial.json"
    SUMMARY = RESULTS / f"{stem}.md"
    global BASE, SAMPLE, MODEL
    BASE = RESULTS / f"nf_ton_iot_v3_fixed_model_replay_seed{args.seed}.json"
    SAMPLE = RESULTS / f"nf_ton_iot_v3_replay_seed{args.seed}_initial_sample_rows.csv"
    MODEL = RESULTS / "nf_ton_iot_v3_replay_models" / f"seed{args.seed}_initial_frozen.joblib"
    if OUT.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {OUT}; use --force after review")
    if args.max_chunks < 0:
        raise SystemExit("--max-chunks must be nonnegative")

    baseline = load_baseline_module()
    bridge_core = load_bridge_core()
    expected = json.loads(BASE.read_text(encoding="utf-8"))
    schema = json.loads((RESULTS / "nf_ton_iot_v3_model_schema_audit.json").read_text(encoding="utf-8"))
    features = schema["candidate_features"]
    threshold = float(expected["calibration"]["thresholds_from_benign_scores_only"]["fpr_0p1_percent"]["score_threshold"])
    frozen = joblib.load(MODEL)
    print("Recovering the exact locked initial training sample...", flush=True)
    x_fit, y_fit, source_rows_scanned = extract_initial_training(baseline, features, expected)
    print(f"recovered {len(y_fit):,} rows; building gate replay", flush=True)

    arms = {
        "no_update": frozen,
        "periodic": frozen,
        "bridge": frozen,
        "vbc_sg": frozen,
        "fpr_recall": frozen,
    }
    previous = frozen
    methods = tuple(arms)
    pool = {m: Counter(tn=0, fp=0, fn=0, tp=0) for m in methods}
    day_pool = defaultdict(Counter)
    family_pool = defaultdict(Counter)
    chunk_rows = []
    decisions = []
    fit_seconds_total = 0.0
    gate_seconds = Counter()
    gate_label_use = Counter()
    recent: deque[tuple[pd.DataFrame, np.ndarray]] = deque(maxlen=RECENT_TRAIN_CHUNKS)
    pending_updates: list[dict] = []
    started = time.perf_counter()
    alpha_each = 0.05 / (2 * 43)  # two endpoints, Bonferroni over 43 scheduled proposals
    arm_names = ("no_update", "periodic", "bridge", "vbc_sg", "fpr_recall")

    for chunk_no, frame, first_end, last_end in baseline.iter_replay_chunks(features):
        if args.max_chunks and chunk_no > args.max_chunks:
            break
        x_values = baseline.frame_to_array(frame, features)
        x_frame = baseline.input_frame(x_values, features)
        scores_by_arm = {}
        predict_start = time.perf_counter()
        for method in arm_names:
            scores_by_arm[method] = arms[method].predict_proba(x_frame)[:, 1]
        prediction_seconds = time.perf_counter() - predict_start
        if args.shadow_previous:
            shadow_start = time.perf_counter()
            shadow_cpu_start = time.process_time()
            shadow_scores = previous.predict_proba(x_frame)[:, 1]
            shadow_wall = time.perf_counter() - shadow_start
            shadow_cpu = time.process_time() - shadow_cpu_start
        # Label release follows all incumbent predictions for this chunk.
        y = pd.to_numeric(frame["Label"], errors="raise").to_numpy(dtype=np.int8)
        family = frame["Attack"].astype("string").fillna("<MISSING>").str.strip().to_numpy()
        date = baseline.utc_string(last_end)[:10]
        record = {"chunk": chunk_no, "rows": int(len(y)), "date_utc": date,
                  "first_end_utc": baseline.utc_string(first_end),
                  "last_end_utc": baseline.utc_string(last_end),
                  "prediction_seconds_all_arms": prediction_seconds, "arms": {}}
        if args.shadow_previous:
            record["previous_shadow"] = baseline.counts_for(y, shadow_scores, threshold)
            record["previous_shadow_wall_seconds"] = shadow_wall
            record["previous_shadow_cpu_seconds"] = shadow_cpu
        predictions = {}
        for method in arm_names:
            scores = scores_by_arm[method]
            metrics = baseline.counts_for(y, scores, threshold)
            pred = (scores > threshold).astype(np.int8)
            predictions[method] = pred
            record["arms"][method] = metrics
            for key in ("tn", "fp", "fn", "tp"):
                pool[method][key] += metrics[key]
                day_pool[(date, method)][key] += metrics[key]
            for fam in np.unique(family):
                mask = family == fam
                vals = baseline.counts_for(y[mask], scores[mask], threshold)
                for key in ("tn", "fp", "fn", "tp"):
                    family_pool[(str(fam), method)][key] += vals[key]
        # Score prospective outcomes for candidate decisions already made.
        for update in pending_updates:
            # The incumbent candidate remains active while the next scheduled
            # chunk is scored; include that chunk before its next decision.
            if update["decision_chunk"] < chunk_no <= update["decision_chunk"] + UPDATE_EVERY:
                update["future_chunks"].append({
                    m: {k: record["arms"][m][k] for k in ("false_positive_rate", "attack_recall", "fp", "tp", "fn", "tn")}
                    for m in update["methods"]
                })
        chunk_rows.append(record)

        # Candidate fit uses only the preceding three fully labeled chunks;
        # current chunk is the later-time approval window and is never trained on.
        if chunk_no % UPDATE_EVERY == 0 and len(recent) == RECENT_TRAIN_CHUNKS:
            recent_x = pd.concat([x for x, _ in recent], ignore_index=True)
            recent_y = np.concatenate([yy for _, yy in recent])
            train_x = pd.concat([x_fit, recent_x], ignore_index=True)
            train_y = np.concatenate((y_fit, recent_y))
            candidate = baseline.make_pipeline(features)
            t0 = time.perf_counter()
            candidate.fit(train_x, train_y)
            fit_s = time.perf_counter() - t0
            fit_seconds_total += fit_s
            candidate_scores = candidate.predict_proba(x_frame)[:, 1]
            candidate_pred = (candidate_scores > threshold).astype(np.int8)
            result = {"after_chunk": chunk_no,
                      "validation_chunk": chunk_no,
                      "candidate_train_chunks": [int(v) for v in range(chunk_no - RECENT_TRAIN_CHUNKS, chunk_no)],
                      "validation_is_strictly_after_train": True,
                      "training_rows": int(len(train_y)),
                      "candidate_fit_seconds": fit_s,
                      "validation_rows": int(len(y)),
                      "validation_candidate": baseline.counts_for(y, candidate_scores, threshold),
                      "decisions": {}}
            # All policies compare the same candidate on the same later-time
            # validation chunk, against their own incumbent predictions.
            for method in ("periodic", "bridge", "vbc_sg", "fpr_recall"):
                inc_pred = predictions[method]
                if method == "periodic":
                    gate_result = {"accept": True, "decision": "always_promote",
                                   "labels_used_by_gate": 0}
                elif method == "bridge":
                    t0 = time.perf_counter()
                    decision = bridge_core.gate(y, inc_pred, candidate_pred,
                                                metric="bacc", seed=args.seed + chunk_no,
                                                margin=0.005)
                    gate_seconds[method] += time.perf_counter() - t0
                    gate_label_use[method] += len(y)
                    gate_result = {"accept": bool(decision.accept),
                                   "decision": "commit" if decision.accept else "reject",
                                   "lower_bacc_gain_bootstrap": float(decision.lower_bound),
                                   "point_bacc_gain": float(decision.gain_pool),
                                   "labels_used_by_gate": int(len(y)),
                                   "assumption": "BRIDGE row-bootstrap; temporal exchangeability is not established."}
                elif method == "vbc_sg":
                    t0 = time.perf_counter()
                    rng = np.random.default_rng(args.seed * 1000 + chunk_no)
                    gate_result = vbc_sg_decision(y, inc_pred, candidate_pred, rng)
                    gate_seconds[method] += time.perf_counter() - t0
                    gate_label_use[method] += int(gate_result["labels"])
                    gate_result["labels_used_by_gate"] = int(gate_result["labels"])
                    gate_result["assumption"] = "VBC-SG stratified EB-CS port; its martingale/conditional assumptions are not validated for this stream. Per-proposal alpha=0.10, matching its chronological baseline configuration; no lifetime correction in this pilot."
                else:
                    t0 = time.perf_counter()
                    gate_result = fpr_recall_gate(y, inc_pred, candidate_pred, alpha_each)
                    gate_seconds[method] += time.perf_counter() - t0
                    gate_label_use[method] += len(y)
                    gate_result["labels_used_by_gate"] = int(len(y))
                result["decisions"][method] = gate_result
                if gate_result["accept"]:
                    if method == "fpr_recall":
                        previous = arms[method]
                    arms[method] = candidate
            decisions.append(result)
            pending_updates.append({"decision_chunk": chunk_no,
                                    "methods": list(result["decisions"]),
                                    "accepted": {m: bool(v["accept"]) for m, v in result["decisions"].items()},
                                    "validation": result["validation_candidate"],
                                    "future_chunks": []})
            del recent_x, recent_y, train_x, train_y

        # Current labels become eligible for future candidates after the
        # validation decision has been made.
        recent.append((x_frame.copy(), y.copy()))
        del x_values, x_frame, scores_by_arm, predictions, y, family, frame
        if chunk_no == 1 or chunk_no % 10 == 0:
            print(f"chunks={chunk_no} fits={len(decisions)} rows={sum(r['rows'] for r in chunk_rows):,} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if chunk_no % 20 == 0:
            PARTIAL.write_text(json.dumps({"status": "running", "completed_chunks": chunk_no,
                                           "decisions": decisions, "chunk_rows": chunk_rows,
                                           "fit_seconds_total": fit_seconds_total}, indent=2), encoding="utf-8")

    pooled = {m: baseline.counts_metrics(dict(c)) for m, c in pool.items()}
    per_day = [{"date_utc": date, "method": method, **baseline.counts_metrics(dict(c))}
               for (date, method), c in sorted(day_pool.items())]
    per_family = [{"family": fam, "method": method, **baseline.counts_metrics(dict(c))}
                  for (fam, method), c in sorted(family_pool.items())]
    update_summary = []
    for update_ix, update in enumerate(pending_updates):
        row = {"decision_chunk": update["decision_chunk"],
               "accepted": update["accepted"],
               "validation_candidate_fpr": update["validation"].get("false_positive_rate"),
               "validation_candidate_recall": update["validation"].get("attack_recall"),
               "future_chunks_evaluated": len(update["future_chunks"]),
               "future_by_method": {}, "deployed_interval_by_method": {}}
        for method in update["methods"]:
            future = [c[method] for c in update["future_chunks"]]
            row["future_by_method"][method] = {
                "max_chunk_fpr": max((v["false_positive_rate"] for v in future), default=None),
                "mean_chunk_fpr": (float(np.mean([v["false_positive_rate"] for v in future])) if future else None),
                "mean_chunk_recall": (float(np.mean([v["attack_recall"] for v in future])) if future else None),
                "any_future_chunk_over_fpr_cap": any(v["false_positive_rate"] > 0.001 for v in future),
            }
            if update["accepted"].get(method):
                next_accept = next((later["decision_chunk"] for later in pending_updates[update_ix + 1:]
                                    if later["accepted"].get(method)), len(chunk_rows))
                lo = update["decision_chunk"] + 1
                hi = max(lo - 1, min(int(next_accept), len(chunk_rows)))
                deployed = [chunk_rows[idx - 1]["arms"][method]
                            for idx in range(lo, hi + 1)]
                row["deployed_interval_by_method"][method] = {
                    "first_chunk": lo if deployed else None,
                    "last_chunk": hi if deployed else None,
                    "chunks_evaluated": len(deployed),
                    "max_chunk_fpr": max((v["false_positive_rate"] for v in deployed), default=None),
                    "any_future_chunk_over_fpr_cap": any(v["false_positive_rate"] > 0.001 for v in deployed),
                    "over_cap_chunk_count": sum(v["false_positive_rate"] > 0.001 for v in deployed),
                }
            else:
                row["deployed_interval_by_method"][method] = None
        update_summary.append(row)
    result = {
        "status": "pilot_truncated" if args.max_chunks else "complete",
        "seed": args.seed, "development_only": True,
        "protocol": "causal current-chunk holdout; candidate trained on prior three labeled chunks; evaluated on next future chunks",
        "source_sha256": expected["source_sha256"],
        "initial_model_sha256": expected["model"].get("initial_model_sha256"),
        "rows": int(sum(r["rows"] for r in chunk_rows)),
        "chunks": len(chunk_rows), "threshold": threshold, "fpr_cap": 0.001,
        "methods": list(methods), "pooled_metrics": pooled,
        "per_day_metrics": per_day, "per_family_metrics": per_family,
        "candidate_decisions": decisions, "prospective_update_outcomes": update_summary,
        "gate_label_use_total": dict(gate_label_use),
        "gate_runtime_seconds": dict(gate_seconds), "candidate_fit_runtime_seconds_total": fit_seconds_total,
        "total_wall_seconds": time.perf_counter() - started,
        "initial_sample_rows_recovered_through_source_row": source_rows_scanned,
        "limits": [
            "Single seed on an inspected NF-ToN-IoT-v3 campaign stream; not independent confirmation.",
            "The end timestamp is the last packet event time, not exporter emission time.",
            "BRIDGE bootstrap and exact-binomial/McNemar pilot tests assume row-level exchangeability/independence that this flow stream does not establish.",
            "VBC-SG is ported from its MIT artifact with causal later-time validation; alpha is per proposal, with no lifetime correction in this one-seed feasibility pilot.",
            "Per-chunk and per-day false-positive outcomes are descriptive and not independent observations.",
            "The FPR/recall gate is a candidate policy under investigation, not a formal future-FPR guarantee.",
        ],
        "artifact_references": {
            "BRIDGE": "research_refs/BRIDGE_CODE_AUDIT_2026-11-05.md",
            "VBC-SG": "research_refs/validate_before_commit_v1_24/roberto-fernandez-barrios-Validate-Before-Commit-8e31ea8/src/experiments/run_paper2_temporal_stream.py",
        },
        "chunk_metrics": chunk_rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    PARTIAL.unlink(missing_ok=True)
    lines = [f"# NF-ToN-IoT-v3 causal promotion-gate pilot (seed {args.seed})", "",
             "Development-only one-seed comparison. Candidates use only the three labeled chunks before the validation chunk; the current validation chunk is held out and scored before its labels are used for acceptance. Prospective outcomes cover the next five incumbent-scored chunks, including the chunk on which the next candidate is considered.", "",
             "## Pooled results", "",
             "| Method | FPR | Attack recall | Macro F1 |", "|---|---:|---:|---:|"]
    for method, m in pooled.items():
        lines.append(f"| {method} | {m['false_positive_rate']:.4%} | {m['attack_recall']:.2%} | {m['macro_f1']:.4f} |")
    lines += ["", "## Promotion decisions", "",
              "| Method | Accepted / proposed | Proposals followed by any over-cap future chunk | Labels examined for gate | Gate time (s) |",
              "|---|---:|---:|---:|---:|"]
    nprop = len(update_summary)
    for method in ("periodic", "bridge", "vbc_sg", "fpr_recall"):
        accepted = sum(bool(u["accepted"].get(method)) for u in update_summary)
        harms = sum(bool((u.get("deployed_interval_by_method", {}).get(method) or {}).get("any_future_chunk_over_fpr_cap"))
                    for u in update_summary if u["accepted"].get(method))
        lines.append(f"| {method} | {accepted}/{nprop} | {harms} | {gate_label_use.get(method, 0):,} | {gate_seconds.get(method, 0.0):.3f} |")
    lines += ["", "## Limits", "",
              "The pilot uses one inspected stream and one seed. BRIDGE's multinomial bootstrap and exact binomial/McNemar bounds are shown as published decision baselines, not as valid guarantees for dependent flows. VBC-SG uses a later-time validation chunk here; the source paper's temporal runner uses older probe windows, so this is a causal protocol adaptation, not a bit-for-bit reproduction of its replay experiment. The FPR-cap gate's nominal confidence calculations do not account for serial dependence or arbitrary future drift.", "",
              f"Machine-readable detail: `{OUT.relative_to(ROOT).as_posix()}`. Protocol runner: `research_refs/run_nf_ton_iot_v3_promotion_gate_pilot.py`."]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
