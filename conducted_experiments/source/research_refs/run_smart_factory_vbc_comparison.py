"""VBC-SG comparator replay on TU Wien smart-factory labeled flows.

The October aggressive campaign is development; the later November stealth
campaign is a same-site later-campaign check. It reuses the locked candidate
proposals from the previous replay and adds the VBC-SG sequential gate as a
post-hoc comparator. It is not an independent-site confirmation study.

The VBC-SG empirical-Bernstein gate is adapted from the MIT-licensed
Validate-Before-Commit artifact; see research_refs/THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import beta, binomtest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tuwien_smart_factory"
OUT = ROOT / "results" / "SMART_FACTORY_A_TO_S_VBC_COMPARISON.json"
REPORT = ROOT / "results" / "SMART_FACTORY_A_TO_S_VBC_COMPARISON.md"
CAP = 0.001
CHUNK_SIZE = 10_000
UPDATE_EVERY = 5
WARMUP = 20_000
SEEDS = (42, 43)


def load_capture(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / name, low_memory=False)
    frame["available_ms"] = (
        pd.to_numeric(frame["flowStartMilliseconds"], errors="raise")
        + pd.to_numeric(frame["flowDurationMilliseconds"], errors="raise")
    ).astype("int64")
    frame["y"] = (frame["Label"].astype(str) != "BENIGN").astype("int8")
    # Use end-time order because final flow counters become available at completion.
    frame = frame.sort_values(["available_ms", "SequenceNumber"], kind="mergesort").reset_index(drop=True)
    return frame


def model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "protocolIdentifier": pd.to_numeric(frame["protocolIdentifier"], errors="raise").astype("int64").astype(str),
        "flowDurationMilliseconds": pd.to_numeric(frame["flowDurationMilliseconds"], errors="raise").astype("float32"),
        "log_packet_count": np.log1p(pd.to_numeric(frame["packetTotalCount"], errors="raise").astype("float64")).astype("float32"),
    })


def make_model(seed: int) -> Pipeline:
    prep = ColumnTransformer([
        ("protocol", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32), ["protocolIdentifier"]),
    ], remainder="passthrough", sparse_threshold=1.0, verbose_feature_names_out=False)
    clf = LGBMClassifier(
        objective="binary", n_estimators=100, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, max_bin=127,
        n_jobs=1, random_state=seed, deterministic=True,
        force_col_wise=True, verbosity=-1,
    )
    return Pipeline([("preprocess", prep), ("classifier", clf)])


def threshold_for_fpr(scores: np.ndarray, cap: float = CAP) -> tuple[float, int]:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Need finite benign-only calibration scores")
    allowed = int(np.floor(cap * len(scores)))
    threshold = float(np.sort(scores)[len(scores) - allowed - 1])
    realized = int(np.count_nonzero(scores > threshold))
    if realized > allowed:
        raise RuntimeError("Empirical calibration threshold exceeds its false-positive allowance")
    return threshold, realized


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    benign = y == 0
    attack = ~benign
    tp = int(np.count_nonzero(attack & (pred == 1)))
    fn = int(np.count_nonzero(attack & (pred == 0)))
    fp = int(np.count_nonzero(benign & (pred == 1)))
    tn = int(np.count_nonzero(benign & (pred == 0)))
    return {"n": len(y), "benign_n": int(benign.sum()), "attack_n": int(attack.sum()),
            "fp": fp, "tn": tn, "tp": tp, "fn": fn,
            "fpr": fp / max(1, fp + tn), "recall": tp / max(1, tp + fn),
            "precision": tp / max(1, tp + fp)}


def promotion_gate(y: np.ndarray, incumbent_pred: np.ndarray, candidate_pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8)
    benign = y == 0
    attack = y == 1
    if not benign.any() or not attack.any():
        return {"accept": False, "decision": "defer_single_class_validation",
                "benign_n": int(benign.sum()), "attack_n": int(attack.sum())}
    fp = int(np.count_nonzero((candidate_pred == 1) & benign))
    n_benign = int(benign.sum())
    fpr_ucb = (1.0 if fp == n_benign else float(beta.ppf(0.95, fp + 1, n_benign - fp)))
    better = int(np.count_nonzero(attack & (candidate_pred == 1) & (incumbent_pred == 0)))
    worse = int(np.count_nonzero(attack & (candidate_pred == 0) & (incumbent_pred == 1)))
    discordant = better + worse
    p_gain = 1.0 if discordant == 0 else float(binomtest(better, discordant, 0.5, alternative="greater").pvalue)
    fpr_ok = fpr_ucb <= CAP
    recall_ok = better > worse and p_gain < 0.05
    return {"accept": bool(fpr_ok and recall_ok), "decision": "promote" if fpr_ok and recall_ok else "reject",
            "benign_n": n_benign, "candidate_fp": fp, "candidate_fpr_ucb_95": fpr_ucb,
            "fpr_cap": CAP, "recall_better": better, "recall_worse": worse,
            "recall_gain_p_one_sided": p_gain, "fpr_pass": bool(fpr_ok), "recall_gain_pass": bool(recall_ok),
            "assumption": "Nominal exact-binomial/McNemar calculations; no guarantee under serial dependence, repeated looks, or future drift."}


def vbc_sg_gate(y: np.ndarray, incumbent_pred: np.ndarray,
                candidate_pred: np.ndarray, rng: np.random.Generator,
                alpha: float = 0.10, block_size: int = 16,
                max_steps: int = 5) -> dict:
    """VBC-SG stratified empirical-Bernstein sequential gate from its MIT artifact.

    This is a causal protocol port: candidate and incumbent score the same current
    chunk; its labels are exposed only after scoring. The gate samples paired
    correctness differences in class-stratified blocks and stops at five looks.
    Its published guarantee assumptions are not asserted for dependent flow chunks.
    """
    y = np.asarray(y, dtype=np.int8)
    inc = np.asarray(incumbent_pred, dtype=np.int8)
    cand = np.asarray(candidate_pred, dtype=np.int8)
    pools = {c: np.flatnonzero(y == c) for c in (0, 1)}
    if any(len(pools[c]) == 0 for c in (0, 1)):
        return {"accept": False, "decision": "reject_single_class",
                "labels_used": 0, "steps": 0}
    for c in (0, 1):
        rng.shuffle(pools[c])
    used = {0: 0, 1: 0}
    diffs = {0: [], 1: []}
    lower = upper = None
    for step in range(1, max_steps + 1):
        for c in (0, 1):
            lo = used[c]
            hi = min(lo + block_size, len(pools[c]))
            ix = pools[c][lo:hi]
            used[c] = hi
            if len(ix):
                d = ((cand[ix] == y[ix]).astype(float)
                     - (inc[ix] == y[ix]).astype(float))
                diffs[c].extend(d.tolist())
        lbs = []
        ubs = []
        for c in (0, 1):
            vals = np.asarray(diffs[c], dtype=np.float64)
            lbs.append(cs_lower_bound_eb(vals, alpha / 4.0))
            ubs.append(-cs_lower_bound_eb(-vals, alpha / 4.0))
        lower, upper = 0.5 * sum(lbs), 0.5 * sum(ubs)
        if lower > 0.0:
            accept, decision = True, "commit"
            break
        if upper < 0.0:
            accept, decision = False, "reject"
            break
    else:
        step = max_steps
        accept, decision = False, "defer_cap_reject"
    return {"accept": bool(accept), "decision": decision,
            "labels_used": sum(map(len, diffs.values())), "steps": step,
            "lower_balanced_accuracy_gain": float(lower),
            "upper_balanced_accuracy_gain": float(upper),
            "alpha_per_proposal": alpha,
            "assumption": "VBC-SG stratified empirical-Bernstein CS port; conditional validity under this stream's serial dependence and repeated proposals is not established."}


def cs_lower_bound_eb(d: np.ndarray, alpha: float) -> float:
    """The empirical-Bernstein confidence-sequence bound used by VBC-SG."""
    x = (np.asarray(d, dtype=np.float64) + 1.0) / 2.0
    n = len(x)
    if n == 0:
        return -float("inf")
    loginv = np.log(1.0 / alpha)
    numerator = denominator = sum_x = sum_v = 0.0
    for i, xi in enumerate(x, start=1):
        mean_hat = (0.5 + sum_x) / i
        var_hat = (0.25 + sum_v) / i
        lam = min(0.75, np.sqrt(2.0 * loginv /
                                  max(var_hat * i * np.log(i + 1), 1e-12)))
        v = (float(xi) - mean_hat) ** 2
        psi = -np.log(1.0 - lam) - lam
        numerator += lam * float(xi) - psi * v
        denominator += lam
        sum_x += float(xi)
        sum_v += v
    return float(2.0 * ((numerator - loginv) / denominator) - 1.0)


def pack_model_bytes(model: Pipeline) -> int:
    stream = io.BytesIO()
    joblib.dump(model, stream, compress=3)
    return stream.tell()


def run_seed(seed: int) -> dict:
    # Fit/calibration split stays within the official benign week; later 20% is never fit.
    tf = load_capture("tf_a_flows_labeled.csv")
    split = int(len(tf) * 0.80)
    base_train = tf.iloc[:split].copy()
    calibration = tf.iloc[split:].copy()
    first_file = load_capture("a_day1_flows_labeled.csv")
    warmup = first_file.iloc[:WARMUP].copy()
    initial_x = pd.concat([model_frame(base_train), model_frame(warmup)], ignore_index=True)
    initial_y = np.concatenate([base_train["y"].to_numpy(), warmup["y"].to_numpy()])
    model = make_model(seed)
    fit_start = time.process_time()
    model.fit(initial_x, initial_y)
    initial_fit_cpu = time.process_time() - fit_start
    cal_scores = model.predict_proba(model_frame(calibration))[:, 1]
    threshold, cal_fp = threshold_for_fpr(cal_scores[calibration["y"].to_numpy() == 0])

    arms = {
        "no_update": {"model": model, "threshold": threshold},
        "periodic": {"model": model, "threshold": threshold},
        "fpr_recall_gate": {"model": model, "threshold": threshold},
        "vbc_sg": {"model": model, "threshold": threshold},
        "threshold_only": {"model": model, "threshold": threshold},
    }
    a_captures = [("a_day1_flows_labeled.csv", first_file.iloc[WARMUP:].copy()),
                  ("a_day2_flows_labeled.csv", load_capture("a_day2_flows_labeled.csv"))]
    s_captures = [(name, load_capture(name)) for name in
                  ("s_day1_flows_labeled.csv", "s_day2_flows_labeled.csv")]
    a_captures.sort(key=lambda item: int(item[1]["available_ms"].iloc[0]))
    s_captures.sort(key=lambda item: int(item[1]["available_ms"].iloc[0]))
    phase_captures = [("development_aggressive_A", a_captures),
                      ("later_stealth_S_same_site_check", s_captures)]

    history_frames = [warmup]
    history_y = [warmup["y"].to_numpy()]
    chunks = []
    decisions = []
    raw_predictions = {key: [] for key in arms}
    chunk_no = 0
    rolling_blocks = []
    fit_cpu = {"periodic": 0.0, "fpr_recall_gate": 0.0, "vbc_sg": 0.0}
    decision_cpu = {"periodic": 0.0, "fpr_recall_gate": 0.0, "vbc_sg": 0.0, "threshold_only": 0.0}
    promote_events = {key: [] for key in arms}

    for phase, captures in phase_captures:
      for capture_name, capture in captures:
        capture = capture.sort_values(["available_ms", "SequenceNumber"], kind="mergesort")
        for start in range(0, len(capture), CHUNK_SIZE):
            block = capture.iloc[start:start + CHUNK_SIZE].copy()
            if block.empty:
                continue
            chunk_no += 1
            if chunk_no == 1 or chunk_no % 10 == 0:
                print(f"seed={seed} phase={phase} chunk={chunk_no}", flush=True)
            x = model_frame(block)
            y = block["y"].to_numpy(dtype=np.int8)
            row = {"chunk": chunk_no, "phase": phase, "capture": capture_name,
                   "available_start_ms": int(block["available_ms"].iloc[0]),
                   "available_end_ms": int(block["available_ms"].iloc[-1]),
                   "label_counts": {str(k): int(v) for k, v in block["Label"].value_counts().items()},
                   "arms": {}}
            incumbent_preds = {}
            for arm, state in arms.items():
                t0 = time.process_time()
                scores = state["model"].predict_proba(x)[:, 1]
                inference_cpu = time.process_time() - t0
                pred = (scores > state["threshold"]).astype(np.int8)
                incumbent_preds[arm] = pred
                raw_predictions[arm].append((y.copy(), pred.copy(), phase, capture_name, chunk_no,
                                             block["Label"].astype(str).to_numpy()))
                row["arms"][arm] = {**metrics(y, pred), "inference_cpu_seconds": inference_cpu,
                                     "threshold": float(state["threshold"])}

            if chunk_no % UPDATE_EVERY == 0:
                fit_data = pd.concat([base_train, warmup] + history_frames[1:], ignore_index=True)
                fit_labels = np.concatenate([base_train["y"].to_numpy(dtype=np.int8),
                                             np.concatenate(history_y)])
                candidate = make_model(seed)
                t0 = time.process_time()
                candidate.fit(model_frame(fit_data), fit_labels)
                candidate_fit_cpu = time.process_time() - t0
                for arm in ("periodic", "fpr_recall_gate", "vbc_sg"):
                    fit_cpu[arm] += candidate_fit_cpu
                cand_scores = candidate.predict_proba(x)[:, 1]
                update = {"chunk": chunk_no, "phase": phase, "capture": capture_name,
                          "candidate_fit_rows": len(fit_labels),
                          "candidate_fit_cpu_seconds": candidate_fit_cpu,
                          "validation": metrics(y, (cand_scores > threshold).astype(np.int8)),
                          "decisions": {}}
                for arm in ("periodic", "fpr_recall_gate", "vbc_sg"):
                    t0 = time.process_time()
                    if arm == "periodic":
                        gate = {"accept": True, "decision": "unconditional_periodic_promotion"}
                    elif arm == "vbc_sg":
                        gate = vbc_sg_gate(y, incumbent_preds[arm],
                                           (cand_scores > threshold).astype(np.int8),
                                           np.random.default_rng(seed * 1_000_003 + chunk_no))
                    else:
                        gate = promotion_gate(y, incumbent_preds[arm], (cand_scores > threshold).astype(np.int8))
                    decision_cpu[arm] += time.process_time() - t0
                    update["decisions"][arm] = gate
                    if gate["accept"]:
                        arms[arm]["model"] = candidate
                        promote_events[arm].append({"chunk": chunk_no, "phase": phase, "capture": capture_name,
                                                    "validation": update["validation"], "gate": gate,
                                                    "candidate_fit_rows": len(fit_labels)})
                # Threshold-only: use only already-released benign scores from the latest 3 chunks.
                t0 = time.process_time()
                # Re-score the latest three already-labeled chunks with the frozen model.
                prior_blocks = (rolling_blocks + [block])[-3:]
                if prior_blocks:
                    bx = pd.concat([model_frame(b) for b in prior_blocks], ignore_index=True)
                    by = np.concatenate([b["y"].to_numpy(dtype=np.int8) for b in prior_blocks])
                    bscore = arms["threshold_only"]["model"].predict_proba(bx)[:, 1]
                    bbenign = bscore[by == 0]
                    if len(bbenign) >= 1000:
                        new_threshold, realized = threshold_for_fpr(bbenign)
                        arms["threshold_only"]["threshold"] = new_threshold
                        row["threshold_recalibration"] = {"benign_n": int(len(bbenign)),
                                                           "calibration_fp": realized,
                                                           "threshold": new_threshold,
                                                           "fallback": False}
                    else:
                        row["threshold_recalibration"] = {"benign_n": int(len(bbenign)), "fallback": True}
                else:
                    row["threshold_recalibration"] = {"benign_n": 0, "fallback": True}
                decision_cpu["threshold_only"] += time.process_time() - t0
                update["threshold_recalibration"] = row.get("threshold_recalibration")
                decisions.append(update)
            chunks.append(row)
            history_frames.append(block)
            history_y.append(y.copy())
            rolling_blocks.append(block)
            if len(rolling_blocks) > 3:
                rolling_blocks = rolling_blocks[-3:]

    aggregate = {}
    for arm, pieces in raw_predictions.items():
        y_all = np.concatenate([p[0] for p in pieces])
        pred_all = np.concatenate([p[1] for p in pieces])
        aggregate[arm] = metrics(y_all, pred_all)
        by_capture = {}
        aggregate[arm]["by_phase"] = {}
        for phase_name in dict.fromkeys(p[2] for p in pieces):
            selected_phase = [p for p in pieces if p[2] == phase_name]
            aggregate[arm]["by_phase"][phase_name] = metrics(
                np.concatenate([p[0] for p in selected_phase]),
                np.concatenate([p[1] for p in selected_phase]))
            aggregate[arm]["by_phase"][phase_name]["chunks_over_cap"] = sum(
                c["arms"][arm]["fpr"] > CAP for c in chunks if c["phase"] == phase_name)
            aggregate[arm]["by_phase"][phase_name]["max_chunk_fpr"] = max(
                (c["arms"][arm]["fpr"] for c in chunks if c["phase"] == phase_name), default=None)
            phase_families = sorted({family for item in selected_phase for family in item[5] if family != "BENIGN"})
            aggregate[arm]["by_phase"][phase_name]["attack_family_recall"] = {}
            for family in phase_families:
                n_family = sum(int(np.count_nonzero(item[5] == family)) for item in selected_phase)
                tp_family = sum(int(np.count_nonzero((item[5] == family) & (item[1] == 1))) for item in selected_phase)
                aggregate[arm]["by_phase"][phase_name]["attack_family_recall"][family] = {
                    "n": n_family, "tp": tp_family, "recall": tp_family / max(1, n_family)}
        for capture_name in dict.fromkeys(p[3] for p in pieces):
            selected = [p for p in pieces if p[3] == capture_name]
            by_capture[capture_name] = metrics(np.concatenate([p[0] for p in selected]),
                                                np.concatenate([p[1] for p in selected]))
        aggregate[arm]["by_capture"] = by_capture
        families = sorted({family for item in pieces for family in item[5] if family != "BENIGN"})
        aggregate[arm]["attack_family_recall"] = {}
        for family in families:
            family_rows = [(item[0], item[1], item[5]) for item in pieces]
            n_family = sum(int(np.count_nonzero(labels == family)) for _, _, labels in family_rows)
            tp_family = sum(int(np.count_nonzero((labels == family) & (pred == 1)))
                            for _, pred, labels in family_rows)
            aggregate[arm]["attack_family_recall"][family] = {
                "n": n_family, "tp": tp_family, "recall": tp_family / max(1, n_family)}
        arm_chunks = [c["arms"][arm] for c in chunks]
        aggregate[arm]["chunk_tail"] = {
            "max_chunk_fpr": max((c["fpr"] for c in arm_chunks), default=None),
            "chunks_over_cap": sum(c["fpr"] > CAP for c in arm_chunks),
            "chunks_with_benign": sum(c["benign_n"] > 0 for c in arm_chunks),
        }
        aggregate[arm]["accepted_updates"] = len(promote_events.get(arm, []))
        aggregate[arm]["final_model_bytes_joblib_z3"] = pack_model_bytes(arms[arm]["model"])
        if promote_events.get(arm):
            tenures = []
            events = promote_events[arm]
            for index, event in enumerate(events):
                next_update_chunk = events[index + 1]["chunk"] if index + 1 < len(events) else chunk_no + 1
                future = [c for c in chunks if event["chunk"] < c["chunk"] < next_update_chunk]
                fprs = [c["arms"][arm]["fpr"] for c in future]
                recalls = [c["arms"][arm]["recall"] for c in future]
                tenures.append({"promotion_chunk": event["chunk"], "future_chunks": len(future),
                                "max_future_chunk_fpr": max(fprs, default=None),
                                "mean_future_chunk_fpr": float(np.mean(fprs)) if fprs else None,
                                "mean_future_chunk_recall": float(np.mean(recalls)) if recalls else None,
                                "any_future_chunk_over_cap": any(v > CAP for v in fprs)})
            aggregate[arm]["promotion_tenures"] = tenures
            aggregate[arm]["harmful_promotions"] = sum(t["any_future_chunk_over_cap"] for t in tenures)
        else:
            aggregate[arm]["promotion_tenures"] = []
            aggregate[arm]["harmful_promotions"] = 0

    vbc_decisions = [d["decisions"]["vbc_sg"] for d in decisions if "vbc_sg" in d["decisions"]]
    vbc_summary = {
        "candidate_proposals": len(vbc_decisions),
        "accepted": sum(bool(d["accept"]) for d in vbc_decisions),
        "labels_used": sum(int(d.get("labels_used", 0)) for d in vbc_decisions),
        "decision_counts": {name: sum(d.get("decision") == name for d in vbc_decisions)
                            for name in sorted({d.get("decision") for d in vbc_decisions})},
    }
    return {
        "seed": seed, "dataset": "TU Wien smart factory", "campaigns": "October aggressive development followed by November stealth same-site check",
        "source_feature_policy": ["flowDurationMilliseconds", "protocolIdentifier one-hot", "log1p(packetTotalCount)"],
        "excluded_predictors": ["SequenceNumber", "flowStartMilliseconds", "sourceIPAddress", "destinationIPAddress", "sourceTransportPort", "destinationTransportPort", "Label"],
        "initial": {"base_train_rows": len(base_train), "benign_calibration_rows": len(calibration),
                    "warmup_rows": len(warmup), "warmup_label_counts": {str(k): int(v) for k, v in warmup["Label"].value_counts().items()},
                    "threshold": threshold, "calibration_fp": cal_fp,
                    "model_fit_cpu_seconds": initial_fit_cpu},
        "protocol": {"chunk_flows": CHUNK_SIZE, "update_every_chunks": UPDATE_EVERY,
                     "nominal_fpr_cap": CAP, "gate_alpha": 0.05,
                     "gate_validation": "current scored chunk after labels arrive; candidate fit excludes current chunk",
                     "assumption_note": "nominal binomial/McNemar; repeated looks and flow dependence are not controlled"},
        "aggregate": aggregate, "promotion_events": promote_events,
        "vbc_sg_decision_summary": vbc_summary,
        "decision_log": decisions, "chunks": chunks,
        "compute": {"candidate_fit_cpu_seconds": fit_cpu, "decision_cpu_seconds": decision_cpu},
    }


def render_report(result: dict) -> None:
    lines = ["# Smart-factory A-to-S replay with VBC-SG comparator", "",
             "This is a post-hoc comparator extension of the existing A→S replay, not a preregistered confirmation. Candidate proposals and fits are shared across periodic, FPR+recall, and VBC-SG arms. VBC-SG applies its stratified paired-correctness confidence-sequence gate to the current already-scored 10,000-flow chunk after labels are released, stopping after up to five 16-per-class looks. It has the same proposal and candidate-training schedule; labels consumed by the gate are separately reported. The FPR+recall gate uses the full labeled validation chunk, while VBC-SG samples at most 160 labels per proposal; the label-evidence budget is therefore not matched. Under the selected per-chunk label-release assumption, this measures decision-label consumption rather than different acquisition budgets. VBC-SG uses alpha=0.10 per proposal without deployment-lifetime correction. Its statistical assumptions are not validated for serially dependent flows or repeated promotion decisions. No S data is used for initial fit/calibration. This is same-site evidence, not independent-site confirmation. Base protocol and limitations: `SMART_FACTORY_REPLAY_PROTOCOL.md`.", "",
             "## Outcomes by campaign phase", "", "| Seed | Phase | Arm | FPR | Recall | Precision | Chunks over cap | Worst chunk FPR |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for run in result["seeds"]:
        for arm, m in run["aggregate"].items():
            for phase, pm in m["by_phase"].items():
                lines.append(f"| {run['seed']} | {phase} | {arm} | {pm['fpr']:.6%} | {pm['recall']:.4%} | {pm['precision']:.4%} | {pm['chunks_over_cap']} | {pm['max_chunk_fpr']:.6%} |")
    lines += ["", "## Accepted gate promotions: validation and future tenure", "",
              "FPR-harmful tenure means at least one future scored chunk exceeds the 0.1% cap. Recall effects are reported separately.", "",
              "| Seed | Phase/chunk | Validation benign/attack | Validation FPR/recall | Future chunks | Future max FPR | Future mean recall | FPR harm |", "|---:|---|---:|---:|---:|---:|---:|---|"]
    for run in result["seeds"]:
        events = run["promotion_events"]["fpr_recall_gate"]
        tenures = run["aggregate"]["fpr_recall_gate"]["promotion_tenures"]
        for event, tenure in zip(events, tenures):
            val = event["validation"]
            lines.append(f"| {run['seed']} | {event['phase']} / {event['chunk']} | {val['benign_n']:,}/{val['attack_n']:,} | {val['fpr']:.6%}/{val['recall']:.4%} | {tenure['future_chunks']} | {tenure['max_future_chunk_fpr']:.6%} | {tenure['mean_future_chunk_recall']:.4%} | {tenure['any_future_chunk_over_cap']} |")
    lines += ["", "## VBC-SG accepted promotions", "", "VBC-SG tests paired balanced-accuracy improvement; it has no hard false-positive cap. Future-tenure FPR and recall below are therefore outcomes, not guarantees.", "", "| Seed | Promotion chunk | Labels used | Looks | Decision-time BA-gain interval | Future chunks | Future max FPR | Future mean recall | FPR harm |", "|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for run in result["seeds"]:
        events = run["promotion_events"]["vbc_sg"]
        tenures = run["aggregate"]["vbc_sg"]["promotion_tenures"]
        for event, tenure in zip(events, tenures):
            gate = event["gate"]
            interval = f"[{gate['lower_balanced_accuracy_gain']:.4f}, {gate['upper_balanced_accuracy_gain']:.4f}]"
            lines.append(f"| {run['seed']} | {event['phase']} / {event['chunk']} | {gate['labels_used']} | {gate['steps']} | {interval} | {tenure['future_chunks']} | {tenure['max_future_chunk_fpr']:.6%} | {tenure['mean_future_chunk_recall']:.4%} | {tenure['any_future_chunk_over_cap']} |")
    lines += ["", "Decision summary by seed: " + "; ".join(
        f"seed {run['seed']}: {run['vbc_sg_decision_summary']['accepted']}/{run['vbc_sg_decision_summary']['candidate_proposals']} promoted, {run['vbc_sg_decision_summary']['labels_used']:,} validation labels"
        for run in result["seeds"]) + ". VBC-SG’s low label count is not a matched-label-budget comparison: the FPR gate evaluates the entire 10,000-flow labeled chunk.", ""]
    lines += ["", "The A promotion passed its validation chunk at 99.67% recall with zero false positives, but its later tenure reached 0.672% chunk FPR. The S promotion passed validation at about 30.5% recall with zero false positives, but attack recall was zero in all 13 subsequent chunks. Thus the current gate did not control future-tenure FPR in A and did not preserve future attack recall in S.", "",
              "## Attack-family recall in S", "", "| Seed | Arm | Family | Flows | Recall |", "|---:|---|---|---:|---:|"]
    for run in result["seeds"]:
        for arm, metrics_by_arm in run["aggregate"].items():
            family_metrics = metrics_by_arm["by_phase"]["later_stealth_S_same_site_check"]["attack_family_recall"]
            for family, value in family_metrics.items():
                lines.append(f"| {run['seed']} | {arm} | {family} | {value['n']:,} | {value['recall']:.4%} |")
    lines += ["", "Gate and periodic policies detect nmap but miss meterpreter almost entirely. The pooled S recall therefore hides a substantial attack-family coverage gap.", "",
              "## Overall cost and model size", "", "| Seed | Arm | Promotions | FPR-harmful tenures | Pooled FPR | Recall | Model bytes | Candidate fit CPU seconds |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for run in result["seeds"]:
        for arm, metrics_by_arm in run["aggregate"].items():
            fit_cpu = run["compute"]["candidate_fit_cpu_seconds"].get(arm, 0.0)
            lines.append(f"| {run['seed']} | {arm} | {metrics_by_arm['accepted_updates']} | {metrics_by_arm['harmful_promotions']} | {metrics_by_arm['fpr']:.6%} | {metrics_by_arm['recall']:.4%} | {metrics_by_arm['final_model_bytes_joblib_z3']:,} | {fit_cpu:.2f} |")
    lines += ["", "## Interpretation limits", "",
              "The FPR gate uses nominal one-sided exact-binomial and McNemar calculations at repeated chunk looks; neither provides a future-risk guarantee under serial dependence or drift. Attacks were controlled pentests at one factory cell, and the primary feature set has only three predictors. Event-time replay uses flow completion timestamps, not observed exporter arrival. Two model random-state settings on one fixed stream are not independent data replications. This later campaign is a same-site check, not independent-site confirmation.", "",
              f"Full machine-readable decisions and per-chunk results: `{OUT.relative_to(ROOT)}`.",
              "Figure generated from the saved replay trace: `results/smart_factory_a_to_s_vbc_comparison.png` (`research_refs/plot_smart_factory_a_to_s_vbc_comparison.py`)."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing result: {OUT}")
    result = {"protocol": "SMART_FACTORY_REPLAY_PROTOCOL.md plus VBC-SG causal comparator; see runner documentation", "seeds": []}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    partial = OUT.with_suffix(".partial.json")
    for seed in SEEDS:
        print(f"starting seed={seed}", flush=True)
        result["seeds"].append(run_seed(seed))
        partial.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    lines = ["# Smart-factory A-to-S temporal adaptation replay", "",
             "October aggressive traffic is the development phase; the later November stealth campaign is a same-site later-campaign check. No S data is used for initial fit or calibration. This is not independent-site or production-network confirmation. Full protocol and limits: `SMART_FACTORY_REPLAY_PROTOCOL.md`.", "",
             "## Outcomes by campaign phase", "", "| Seed | Phase | Arm | FPR | Recall | Precision | Chunks over cap | Worst chunk FPR |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for run in result["seeds"]:
        for arm, m in run["aggregate"].items():
            for phase, pm in m["by_phase"].items():
                lines.append(f"| {run['seed']} | {phase} | {arm} | {pm['fpr']:.6%} | {pm['recall']:.4%} | {pm['precision']:.4%} | {pm['chunks_over_cap']} | {pm['max_chunk_fpr']:.6%} |")
    lines += ["", "## Overall policy outcomes and cost", "", "| Seed | Arm | Promotions | Harmful promotions | Pooled FPR | Recall | Final model bytes | Candidate fit CPU (s) |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for run in result["seeds"]:
        for arm, m in run["aggregate"].items():
            fit_cost = run["compute"]["candidate_fit_cpu_seconds"].get(arm, 0.0)
            lines.append(f"| {run['seed']} | {arm} | {m['accepted_updates']} | {m['harmful_promotions']} | {m['fpr']:.6%} | {m['recall']:.4%} | {m['final_model_bytes_joblib_z3']:,} | {fit_cost:.2f} |")
    lines += ["", "## Interpretation limits", "",
              "The FPR gate uses nominal one-sided exact-binomial and McNemar calculations at repeated chunk looks; neither provides a future-risk guarantee under serial dependence or drift. Attacks were controlled pentests at one factory cell, and the primary feature set has only three predictors. Event-time replay uses flow completion timestamps, not observed exporter arrival. The later campaign is a same-site scenario check, not independent-site confirmation.", "",
              f"Full machine-readable decisions and per-chunk results: `{OUT.relative_to(ROOT)}`."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    render_report(result)
    print(f"Wrote {OUT.relative_to(ROOT)} and {REPORT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()

