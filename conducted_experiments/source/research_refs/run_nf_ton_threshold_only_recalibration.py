"""Run a causal threshold-only comparator on the NF-ToN-IoT-v3 stream.

At every fifth chunk, the frozen seed model scores first. Once that chunk's
labels are released, an empirical benign-score threshold is selected so a
one-sided exact-binomial upper bound meets the original 0.1% FPR cap. The new
threshold is used only on later chunks. The model is never refit. This is a
development baseline; flow dependence and temporal exchangeability are not
established by the binomial calculation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASELINE_SCRIPT = ROOT / "research_refs" / "run_nf_ton_iot_v3_fixed_model_replay.py"
UPDATE_EVERY = 5
FPR_CAP = 0.001
ALPHA_EACH = 0.05 / (2 * 43)  # match original gate's per-endpoint Bonferroni alpha


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("nf_ton_threshold_baseline", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the existing replay helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def threshold_from_labeled_window(y: np.ndarray, scores: np.ndarray,
                                  current_threshold: float) -> dict:
    """Choose the lowest threshold satisfying the nominal FPR UCB on released labels."""
    benign_scores = np.asarray(scores)[np.asarray(y) == 0]
    n = int(len(benign_scores))
    if n == 0:
        return {"updated": False, "reason": "no_benign_labels", "threshold": current_threshold,
                "benign_n": 0, "false_positives": None, "fpr_ucb": None}

    # Find the largest calibration false-positive count whose one-sided
    # Clopper-Pearson upper bound is under the target.
    max_empirical_fp = min(n, int(np.floor(FPR_CAP * n)))
    k = np.arange(max_empirical_fp + 1, dtype=int)
    bounds = beta.ppf(1.0 - ALPHA_EACH, k + 1, n - k)
    acceptable = np.flatnonzero(bounds <= FPR_CAP)
    if len(acceptable) == 0:
        return {"updated": False, "reason": "no_threshold_meets_bound", "threshold": current_threshold,
                "benign_n": n, "false_positives": None, "fpr_ucb": None}

    allowed_fp = int(acceptable[-1])
    ordered = np.sort(benign_scores)
    threshold = float(ordered[n - allowed_fp - 1])
    realized_fp = int(np.count_nonzero(benign_scores > threshold))
    # Ties can only make the realized FPR more conservative than the bound.
    realized_ucb = float(beta.ppf(1.0 - ALPHA_EACH,
                                  realized_fp + 1, n - realized_fp))
    return {
        "updated": True, "reason": "threshold_selected",
        "threshold": threshold, "previous_threshold": current_threshold,
        "threshold_change": threshold - current_threshold,
        "benign_n": n, "allowed_false_positives": allowed_fp,
        "false_positives": realized_fp,
        "empirical_fpr": realized_fp / n,
        "fpr_ucb": realized_ucb,
        "alpha": ALPHA_EACH,
        "calibration_attack_recall": float(np.count_nonzero((np.asarray(y) == 1) &
                                                               (np.asarray(scores) > threshold)) /
                                                   max(1, int(np.count_nonzero(np.asarray(y) == 1)))),
        "assumption_note": "Nominal exact-binomial bound; flow dependence and drift are not controlled.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(42, 43), default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--window-chunks", type=int, choices=(1, 3), default=1,
                        help="number of latest scored/labeled chunks used for threshold calibration")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="truncate for an early feasibility pilot; 0 runs the full stream")
    args = parser.parse_args()
    if args.max_chunks < 0:
        raise SystemExit("--max-chunks must be nonnegative")
    suffix = "" if args.window_chunks == 1 else f"_window{args.window_chunks}"
    stem = (f"nf_ton_iot_v3_threshold_only_smoke{args.max_chunks}_seed{args.seed}{suffix}"
            if args.max_chunks else f"nf_ton_iot_v3_threshold_only{suffix}_seed{args.seed}")
    out = RESULTS / f"{stem}.json"
    summary = RESULTS / f"{stem}.md"
    partial = RESULTS / f"{stem}.partial.json"
    if out.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {out}; use --force after review")

    baseline = load_baseline_module()
    baseline_result_path = RESULTS / f"nf_ton_iot_v3_fixed_model_replay_seed{args.seed}.json"
    baseline_result = json.loads(baseline_result_path.read_text(encoding="utf-8"))
    schema = json.loads((RESULTS / "nf_ton_iot_v3_model_schema_audit.json").read_text(encoding="utf-8"))
    features = schema["candidate_features"]
    model_path = RESULTS / "nf_ton_iot_v3_replay_models" / f"seed{args.seed}_initial_frozen.joblib"
    frozen = joblib.load(model_path)
    threshold = float(baseline_result["calibration"]["thresholds_from_benign_scores_only"]
                      ["fpr_0p1_percent"]["score_threshold"])
    initial_threshold = threshold
    methods = ("fixed_threshold", "threshold_only")
    pooled = {m: Counter(tn=0, fp=0, fn=0, tp=0) for m in methods}
    day_pool = defaultdict(Counter)
    family_pool = defaultdict(Counter)
    chunk_rows = []
    decisions = []
    calibration_window = deque(maxlen=args.window_chunks)
    inference_seconds = 0.0
    calibration_seconds = 0.0
    started = time.perf_counter()

    for chunk_no, frame, first_end, last_end in baseline.iter_replay_chunks(features):
        if args.max_chunks and chunk_no > args.max_chunks:
            break
        x = baseline.frame_to_array(frame, features)
        x_frame = baseline.input_frame(x, features)
        t0 = time.perf_counter()
        scores = frozen.predict_proba(x_frame)[:, 1]
        inference_seconds += time.perf_counter() - t0
        y = pd.to_numeric(frame["Label"], errors="raise").to_numpy(dtype=np.int8)
        family = frame["Attack"].astype("string").fillna("<MISSING>").str.strip().to_numpy()
        date = baseline.utc_string(last_end)[:10]
        fixed = baseline.counts_for(y, scores, initial_threshold)
        adaptive = baseline.counts_for(y, scores, threshold)
        metrics_by_method = {"fixed_threshold": fixed, "threshold_only": adaptive}
        for method, metrics in metrics_by_method.items():
            for key in ("tn", "fp", "fn", "tp"):
                pooled[method][key] += metrics[key]
                day_pool[(date, method)][key] += metrics[key]
            for fam in np.unique(family):
                mask = family == fam
                fam_metrics = baseline.counts_for(y[mask], scores[mask],
                                                 initial_threshold if method == "fixed_threshold" else threshold)
                for key in ("tn", "fp", "fn", "tp"):
                    family_pool[(str(fam), method)][key] += fam_metrics[key]
        row = {
            "chunk": chunk_no, "rows": len(y), "date_utc": date,
            "first_end_utc": baseline.utc_string(first_end),
            "last_end_utc": baseline.utc_string(last_end),
            "threshold_used": threshold,
            "arms": metrics_by_method,
        }
        chunk_rows.append(row)
        calibration_window.append((scores.copy(), y.copy()))

        # Labels become available only after the frozen incumbent scores this chunk.
        if chunk_no % UPDATE_EVERY == 0:
            t0 = time.perf_counter()
            window_scores = np.concatenate([entry[0] for entry in calibration_window])
            window_y = np.concatenate([entry[1] for entry in calibration_window])
            decision = threshold_from_labeled_window(window_y, window_scores, threshold)
            calibration_seconds += time.perf_counter() - t0
            decision["after_chunk"] = chunk_no
            decision["validation_chunk"] = chunk_no
            decision["calibration_chunks"] = [max(1, chunk_no - len(calibration_window) + 1), chunk_no]
            decision["calibration_rows"] = int(len(window_y))
            decision["applies_from_chunk"] = chunk_no + 1
            if decision["updated"]:
                threshold = float(decision["threshold"])
            decision["threshold_after_decision"] = threshold
            decisions.append(decision)

        if chunk_no == 1 or chunk_no % 20 == 0:
            print(f"seed={args.seed} chunks={chunk_no} rows={sum(r['rows'] for r in chunk_rows):,} "
                  f"threshold={threshold:.6f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if chunk_no % 20 == 0:
            partial.write_text(json.dumps({
                "status": "running", "seed": args.seed, "completed_chunks": chunk_no,
                "threshold": threshold, "threshold_decisions": decisions,
                "chunk_metrics": chunk_rows,
            }, indent=2), encoding="utf-8")
        del x, x_frame, scores, y, family, frame

    pooled_metrics = {m: baseline.counts_metrics(dict(c)) for m, c in pooled.items()}
    per_day = [{"date_utc": date, "method": method, **baseline.counts_metrics(dict(c))}
               for (date, method), c in sorted(day_pool.items())]
    per_family = [{"family": family, "method": method, **baseline.counts_metrics(dict(c))}
                  for (family, method), c in sorted(family_pool.items())]
    result = {
        "status": "pilot_truncated" if args.max_chunks else "complete",
        "seed": args.seed, "development_only": True,
        "protocol": f"frozen model; threshold updated after each fifth scored chunk using only the latest {args.window_chunks} scored chunks' released labels; new threshold applies from the next chunk",
        "source_sha256": baseline_result["source_sha256"],
        "initial_model_sha256": baseline_result["model"].get("initial_model_sha256"),
        "rows": int(sum(r["rows"] for r in chunk_rows)), "chunks": len(chunk_rows),
        "initial_threshold": initial_threshold, "fpr_cap": FPR_CAP,
        "update_every_chunks": UPDATE_EVERY, "calibration_window_chunks": args.window_chunks,
        "alpha_per_threshold_update": ALPHA_EACH,
        "methods": list(methods), "pooled_metrics": pooled_metrics,
        "per_day_metrics": per_day, "per_family_metrics": per_family,
        "threshold_decisions": decisions,
        "threshold_updates_accepted": sum(d["updated"] for d in decisions),
        "inference_seconds": inference_seconds,
        "threshold_calibration_seconds": calibration_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "limits": [
            "Two-seed repetitions use the same inspected stream and are not independent confirmation.",
            "The exact-binomial calibration bound assumes independent/exchangeable benign flows; serial dependence is not controlled.",
            "The threshold is selected on one released chunk and evaluated only on later chunks; later distribution shifts can still breach the cap.",
            "This is a threshold-only baseline, not a novel method or a deployment guarantee.",
        ],
        "chunk_metrics": chunk_rows,
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)

    lines = [f"# NF-ToN-IoT-v3 causal threshold-only baseline (seed {args.seed})", "",
             f"Development-only. The initial model is frozen. At each five-chunk checkpoint, the current chunk is scored with the threshold selected before its labels were released. After release, a new threshold is chosen from the latest {args.window_chunks} scored chunk(s) using a one-sided exact-binomial bound; it applies only to subsequent chunks. No later labels enter the decision.", "",
             "## Pooled results", "", "| Policy | FPR | Attack recall | Macro F1 |", "|---|---:|---:|---:|"]
    for method, m in pooled_metrics.items():
        lines.append(f"| {method} | {m['false_positive_rate']:.4%} | {m['attack_recall']:.2%} | {m['macro_f1']:.4f} |")
    lines += ["", f"Threshold updates accepted: {result['threshold_updates_accepted']}/{len(decisions)}.",
              f"Frozen-model inference time: {inference_seconds:.3f}s; threshold calibration time: {calibration_seconds:.3f}s.", "",
              "This comparison is causal with respect to chunk label release, but its binomial bound is nominal under flow dependence and cannot guarantee future FPR under drift. It is a comparator only.", "",
              f"Threshold calibration fallbacks: {sum(not d['updated'] for d in decisions)}/{len(decisions)}.", "",
              f"Machine-readable detail: `results/{out.name}`; runner: `research_refs/run_nf_ton_threshold_only_recalibration.py`."]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
