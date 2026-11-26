"""Audit retrospective versus prefix-only threshold certification on CSE.

This is a development-only, threshold-only sensitivity experiment. It keeps
the source model and target stream fixed, withholds exact common-feature hash
groups from calibration, and compares calibration using labels available by a
checkpoint against calibration using validation labels from the full stream.
The latter is intentionally retrospective and is an oracle, not a deployable
policy or a reproduction of the complete Jeon-Kim method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "cse_replay_chunk_cache_v2"
MANIFEST_PATH = CACHE_DIR / "manifest.json"
MODEL_PATH = ROOT / "results" / "end_to_end_models" / "reference_default.joblib"
SCORE_CACHE = ROOT / "results" / "iotid20_cse_gate_score_cache_seed42.npz"
SCORE_META = SCORE_CACHE.with_suffix(".json")
OUTPUT = ROOT / "results" / "iotid20_cse_prefix_vs_fullperiod_threshold_gate_seed42.json"
REPORT = ROOT / "results" / "IOTID20_CSE_PREFIX_VS_FULLPERIOD_GATE.md"
PLOT = ROOT / "results" / "IOTID20_CSE_THRESHOLD_GATE_TRADEOFF.png"
FPR_CAPS = (0.001, 0.01)
CHECKPOINT_FRACTIONS = (0.25, 0.50, 0.75)
N_GROUP_FOLDS = 5
TARGET_RECALL = 0.90


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wilson_upper(k: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0:
        return 1.0
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    margin = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return float((center + margin) / den)


def choose_threshold(scores: np.ndarray, y: np.ndarray, fpr_cap: float) -> dict:
    scores = np.asarray(scores, dtype=np.float32)
    y = np.asarray(y, dtype=np.uint8)
    benign = np.sort(scores[y == 0])
    attack = scores[y == 1]
    if benign.size == 0:
        return {"threshold": None, "benign_rows": 0, "attack_rows": int(attack.size),
                "validation_fpr": None, "validation_fpr_wilson_upper": 1.0,
                "validation_attack_recall": None, "gate_pass": False}

    lo, hi = 0, int(benign.size)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if wilson_upper(mid, int(benign.size)) <= fpr_cap:
            lo = mid
        else:
            hi = mid - 1
    max_false_positives = lo
    if max_false_positives == benign.size:
        threshold = float(np.nextafter(benign[0], -np.inf))
    else:
        # Permit the maximum statistically feasible number of top benign
        # scores; a threshold just above the next excluded benign score
        # maximizes validation attack recall subject to that constraint.
        excluded_score = benign[benign.size - max_false_positives - 1]
        threshold = float(np.nextafter(excluded_score, np.inf))
    fp = int(np.sum(benign >= threshold))
    recall = float(np.mean(attack >= threshold)) if attack.size else None
    upper = wilson_upper(fp, int(benign.size))
    return {
        "threshold": threshold,
        "benign_rows": int(benign.size),
        "attack_rows": int(attack.size),
        "validation_fp": fp,
        "validation_fpr": fp / int(benign.size),
        "validation_fpr_wilson_upper": upper,
        "validation_attack_recall": recall,
        "gate_pass": bool(upper <= fpr_cap and recall is not None and recall >= TARGET_RECALL),
    }


def score_metrics(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.uint8)
    pred = np.asarray(scores >= threshold, dtype=np.uint8)
    benign = y == 0
    attack = y == 1
    tn = int(np.sum(benign & (pred == 0)))
    fp = int(np.sum(benign & (pred == 1)))
    fn = int(np.sum(attack & (pred == 0)))
    tp = int(np.sum(attack & (pred == 1)))
    return {
        "rows": int(y.size), "benign_rows": int(benign.sum()), "attack_rows": int(attack.sum()),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": fp / (tn + fp) if tn + fp else None,
        "attack_recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "accuracy": (tp + tn) / y.size if y.size else None,
    }


def build_score_cache(manifest: dict) -> tuple[dict, dict]:
    features = manifest["features"]
    rows = int(sum(chunk["rows"] for chunk in manifest["chunks"]))
    scores = np.empty(rows, dtype=np.float32)
    labels = np.empty(rows, dtype=np.uint8)
    folds = np.empty(rows, dtype=np.uint8)
    chunk_ids = np.empty(rows, dtype=np.uint16)
    capture_ids = np.empty(rows, dtype=np.uint8)
    capture_map = {name: i for i, name in enumerate(dict.fromkeys(c["capture_file"] for c in manifest["chunks"]))}
    model = joblib.load(MODEL_PATH)
    offset = 0
    started = time.perf_counter()
    for i, item in enumerate(manifest["chunks"]):
        with np.load(CACHE_DIR / item["file"], allow_pickle=False) as data:
            x = data["x"]
            y = data["y"].astype(np.uint8, copy=False)
            frame = pd.DataFrame(x, columns=features, copy=False)
            sig = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
            end = offset + len(y)
            scores[offset:end] = model.predict_proba(frame)[:, 1].astype(np.float32)
            labels[offset:end] = y
            folds[offset:end] = (sig % N_GROUP_FOLDS).astype(np.uint8)
            chunk_ids[offset:end] = i
            capture_ids[offset:end] = capture_map[item["capture_file"]]
        offset = end
        if (i + 1) % 75 == 0 or i + 1 == len(manifest["chunks"]):
            print(f"scored chunk {i + 1}/{len(manifest['chunks'])}; rows={offset}; "
                  f"elapsed={time.perf_counter() - started:.1f}s", flush=True)

    cache_started = time.perf_counter()
    tmp = SCORE_CACHE.with_suffix(".partial.npz")
    np.savez_compressed(tmp, scores=scores, labels=labels, folds=folds,
                        chunk_ids=chunk_ids, capture_ids=capture_ids)
    os.replace(tmp, SCORE_CACHE)
    meta = {
        "cache_manifest_sha256": sha256(MANIFEST_PATH),
        "model_sha256": sha256(MODEL_PATH),
        "features": features,
        "rows": rows,
        "chunks": len(manifest["chunks"]),
        "capture_files": list(capture_map),
        "group_split": "pandas.hash_pandas_object over all frozen common-77 numeric features, uint64 signature modulo 5; same feature signatures remain in one fold, with negligible but nonzero hash-collision possibility",
        "score_cache_sha256": sha256(SCORE_CACHE),
        "inference_and_hash_seconds": time.perf_counter() - started,
        "score_cache_write_seconds": time.perf_counter() - cache_started,
    }
    SCORE_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"scores": scores, "labels": labels, "folds": folds,
            "chunk_ids": chunk_ids, "capture_ids": capture_ids}, meta


def load_or_build(manifest: dict, rebuild: bool) -> tuple[dict, dict]:
    expected = {"cache_manifest_sha256": sha256(MANIFEST_PATH), "model_sha256": sha256(MODEL_PATH)}
    if not rebuild and SCORE_CACHE.exists() and SCORE_META.exists():
        meta = json.loads(SCORE_META.read_text(encoding="utf-8"))
        if all(meta.get(k) == v for k, v in expected.items()) and meta.get("score_cache_sha256") == sha256(SCORE_CACHE):
            with np.load(SCORE_CACHE, allow_pickle=False) as z:
                return {k: z[k] for k in ("scores", "labels", "folds", "chunk_ids", "capture_ids")}, meta
    return build_score_cache(manifest)


def main(rebuild: bool = False) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    arrays, score_meta = load_or_build(manifest, rebuild)
    scores, y, folds, chunks, captures = (arrays[k] for k in ("scores", "labels", "folds", "chunk_ids", "capture_ids"))
    total_chunks = len(manifest["chunks"])
    results = []
    for fraction in CHECKPOINT_FRACTIONS:
        cutoff = int(np.floor(total_chunks * fraction))
        future = chunks >= cutoff
        prefix = chunks < cutoff
        for fold in range(N_GROUP_FOLDS):
            test_future = (folds == fold) & future
            calibration_groups = folds != fold
            causal_v = calibration_groups & prefix
            fullperiod_v = calibration_groups
            fixed = score_metrics(scores[test_future], y[test_future], 0.5)
            for cap in FPR_CAPS:
                for gate_name, valmask in (("prefix_only", causal_v), ("full_period_retrospective", fullperiod_v)):
                    gate = choose_threshold(scores[valmask], y[valmask], cap)
                    if gate["threshold"] is None:
                        future_metrics = {"rows": int(test_future.sum()), "benign_rows": int(((y == 0) & test_future).sum()),
                                          "attack_rows": int(((y == 1) & test_future).sum()), "fpr": None,
                                          "attack_recall": None, "precision": None, "accuracy": None}
                    else:
                        future_metrics = score_metrics(scores[test_future], y[test_future], gate["threshold"])
                    results.append({
                        "checkpoint_fraction": fraction,
                        "cutoff_chunk_exclusive": cutoff,
                        "group_fold": fold,
                        "fpr_cap": cap,
                        "gate": gate_name,
                        "validation": gate,
                        "future_heldout_group_metrics": future_metrics,
                        "fixed_threshold_0_5_future_metrics": fixed,
                        "meets_future_operational_target": bool(
                            future_metrics.get("fpr") is not None and future_metrics.get("attack_recall") is not None
                            and future_metrics["fpr"] <= cap and future_metrics["attack_recall"] >= TARGET_RECALL
                        ),
                    })
    summary = {}
    for cap in FPR_CAPS:
        for fraction in CHECKPOINT_FRACTIONS:
            for gate in ("prefix_only", "full_period_retrospective"):
                rows = [r for r in results if r["fpr_cap"] == cap and r["checkpoint_fraction"] == fraction and r["gate"] == gate]
                val_recall = [r["validation"]["validation_attack_recall"] for r in rows if r["validation"]["validation_attack_recall"] is not None]
                fut_fpr = [r["future_heldout_group_metrics"]["fpr"] for r in rows if r["future_heldout_group_metrics"]["fpr"] is not None]
                fut_recall = [r["future_heldout_group_metrics"]["attack_recall"] for r in rows if r["future_heldout_group_metrics"]["attack_recall"] is not None]
                pooled = {key: sum(int(r["future_heldout_group_metrics"].get(key, 0) or 0) for r in rows)
                          for key in ("tn", "fp", "fn", "tp", "rows", "benign_rows", "attack_rows")}
                baseline = {key: sum(int(r["fixed_threshold_0_5_future_metrics"].get(key, 0) or 0) for r in rows)
                            for key in ("tn", "fp", "fn", "tp", "rows", "benign_rows", "attack_rows")}
                summary[f"cap={cap};checkpoint={fraction};gate={gate}"] = {
                    "folds": len(rows),
                    "gate_pass_folds": sum(r["validation"]["gate_pass"] for r in rows),
                    "future_target_pass_folds": sum(r["meets_future_operational_target"] for r in rows),
                    "mean_validation_recall": float(np.mean(val_recall)) if val_recall else None,
                    "mean_future_fpr": float(np.mean(fut_fpr)) if fut_fpr else None,
                    "future_fpr_fold_range": [float(min(fut_fpr)), float(max(fut_fpr))] if fut_fpr else None,
                    "mean_future_attack_recall": float(np.mean(fut_recall)) if fut_recall else None,
                    "future_attack_recall_fold_range": [float(min(fut_recall)), float(max(fut_recall))] if fut_recall else None,
                    "pooled_future_confusion_and_support": pooled,
                    "pooled_future_fpr": pooled["fp"] / (pooled["fp"] + pooled["tn"]) if pooled["fp"] + pooled["tn"] else None,
                    "pooled_future_attack_recall": pooled["tp"] / (pooled["tp"] + pooled["fn"]) if pooled["tp"] + pooled["fn"] else None,
                    "pooled_fixed_threshold_0_5": {
                        **baseline,
                        "fpr": baseline["fp"] / (baseline["fp"] + baseline["tn"]) if baseline["fp"] + baseline["tn"] else None,
                        "attack_recall": baseline["tp"] / (baseline["tp"] + baseline["fn"]) if baseline["tp"] + baseline["fn"] else None,
                    },
                }

    output = {
        "status": "complete",
        "experiment": "fixed IoTID20 source model; causal prefix versus retrospective full-period threshold gate on CSE",
        "date": "2026-11-03",
        "development_only": True,
        "closest_work_relation": "threshold-only gate availability sensitivity, not a reproduction of the full Jeon-Kim staged method",
        "source_model": {"path": str(MODEL_PATH.relative_to(ROOT)), "sha256": score_meta["model_sha256"],
                         "source_training": "the existing IoTID20 common-77 reference model; no CSE labels in model fitting"},
        "target": {"manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
                   "manifest_sha256": score_meta["cache_manifest_sha256"],
                   "rows": int(y.size), "chunks": total_chunks, "captures": score_meta["capture_files"],
                   "labels": "existing binary CSE target labels; each chunk's labels are treated as released only after that chunk is scored"},
        "protocol": {
            "candidate_action": "threshold-only recalibration of a fixed source model; no refitting",
            "metric_gate": "maximize past attack recall subject to 95% two-sided Wilson upper bound on past benign FPR <= cap; also report whether recall >= 0.90",
            "caps": list(FPR_CAPS), "target_recall": TARGET_RECALL,
            "checkpoints": list(CHECKPOINT_FRACTIONS),
            "future_test": "at each checkpoint, evaluate only later chunks of one hash-disjoint group fold; training/calibration groups are the other four folds",
            "prefix_only": "calibration uses labels from training-side groups in chunks strictly before the checkpoint; available under post-chunk label release",
            "full_period_retrospective": "calibration uses training-side groups from all chunks, including later chunks; deliberately noncausal oracle that exposes look-ahead optimism",
            "group_partition": score_meta["group_split"],
            "threshold_selection": "exact maximum feasible benign false-positive count under the Wilson bound; threshold is placed immediately above the next excluded benign score; it maximizes recall among thresholds satisfying that count",
        },
        "score_cache": {"path": str(SCORE_CACHE.relative_to(ROOT)), "sha256": score_meta["score_cache_sha256"],
                        "build_seconds": score_meta.get("inference_and_hash_seconds"), "write_seconds": score_meta.get("score_cache_write_seconds")},
        "fold_checkpoint_results": results,
        "summary": summary,
        "limits": [
            "The target CSE replay has already been used during development; no result here is confirmatory or independent.",
            "This isolates threshold-gate information availability; it does not implement the paper's EDG sampling, source-mixed Level 2, target-only Level 3, diversity guard, or per-attack tasks.",
            "The retrospective gate intentionally uses labels from the future relative to its decision checkpoint. It is an invalid deployment policy and only a leakage upper-bound comparator.",
            "Group separation uses a 64-bit row hash over common-77 features; collision risk is negligible but not mathematically zero. It groups feature-equivalent rows, not verified sessions.",
            "Fixed common-77 binary model and one seed limit generalization; FPR and recall are averaged across five hash folds and should be read with the per-fold values.",
            "Chunk order is the immutable v2 CSE cache order, not a claim of true network session order; equal timestamps were already kept together in cache construction.",
        ],
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    make_plot(output)
    make_report(output)
    print(json.dumps({"output": str(OUTPUT), "report": str(REPORT), "plot": str(PLOT), "rows": int(y.size),
                      "summary": summary}, indent=2), flush=True)
    return output


def make_report(result: dict) -> None:
    lines = [
        "# Prefix-only vs retrospective threshold gate on IoTID20→CSE",
        "",
        "**Status:** completed development-only sensitivity experiment; not an algorithmic novelty claim or independent confirmation.",
        "",
        "## Design",
        "",
        "A fixed LightGBM reference model trained on IoTID20 is held constant while its decision threshold is selected from labeled target validation rows. At 25%, 50%, and 75% of the immutable chronological CSE v2 stream, the causal arm uses only labels from chunks already scored; the retrospective oracle uses validation labels from the entire target period, including the future. Evaluation is restricted to later chunks in a hash-disjoint fifth of common-77 feature signatures. The two arms therefore differ in label availability, while model, action, target stream, checkpoint, and group fold remain matched.",
        "",
        "The selected threshold maximizes validation attack recall subject to a 95% Wilson upper bound on benign FPR. Both the project cap (0.1%) and the closest paper's cap (1%) are reported. A gate passes only when the FPR bound and recall ≥ 0.90 both hold. The retrospective arm is intentionally invalid for deployment and serves only as a look-ahead comparator.",
        "",
        "## Results",
        "",
        f"![Future FPR versus attack recall by gate and stream checkpoint]({PLOT.name})",
        "",
        "| FPR cap | Checkpoint | Gate | Gate passes / 5 folds | Future target passes / 5 folds | Mean validation recall | Pooled future FPR (fold range) | Pooled future attack recall (fold range) | Fixed 0.5 FPR / recall |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cap in FPR_CAPS:
        for frac in CHECKPOINT_FRACTIONS:
            for gate in ("prefix_only", "full_period_retrospective"):
                key = f"cap={cap};checkpoint={frac};gate={gate}"
                s = result["summary"][key]
                fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
                b = s["pooled_fixed_threshold_0_5"]
                baseline = f"{fmt(b['fpr'])} / {fmt(b['attack_recall'])}"
                fpr_range = s["future_fpr_fold_range"]
                rec_range = s["future_attack_recall_fold_range"]
                fpr_text = f"{fmt(s['pooled_future_fpr'])} ({fmt(fpr_range[0])}–{fmt(fpr_range[1])})" if fpr_range else "n/a"
                rec_text = f"{fmt(s['pooled_future_attack_recall'])} ({fmt(rec_range[0])}–{fmt(rec_range[1])})" if rec_range else "n/a"
                lines.append(f"| {cap:.3%} | {frac:.0%} | {gate} | {s['gate_pass_folds']}/5 | {s['future_target_pass_folds']}/5 | {fmt(s['mean_validation_recall'])} | {fpr_text} | {rec_text} | {baseline} |")
    lines += [
        "",
        "The machine-readable JSON preserves every fold/checkpoint result, including benign and attack support, threshold, Wilson bound, and future confusion counts. In this run no threshold-only gate passed the joint recall/FPR criterion at any checkpoint or cap. At threshold 0.5 the source model retained much higher attack recall but produced an approximately 99% future FPR. Raising the threshold enough to meet the historical FPR bound destroyed attack recall; the fixed model has no safe operating point under these caps on this target. The retrospective arm therefore did not falsely certify a good candidate here; it failed too. This is a negative result about threshold-only adaptation for this model/transfer pair, not evidence that retrospective gates are generally harmless.",
        "",
        "## Interpretation and limits",
        "",
        "This experiment isolates only threshold recalibration. It is not a reproduction of Jeon & Kim's full three-level system, which includes EDG budgets, source-mixed retraining, target-only retraining, and a diversity guard. CSE is already development-explored, the binary task is not the paper's seven attack-specific task, and the grouping is a 64-bit feature-signature split rather than verified flow/session disjointness. Treat the outcomes as a narrow protocol sensitivity result.",
        "",
        "The paper itself reports that its full-period sufficiency gate makes the deployment decision retrospective and the collection horizon a lower bound. This experiment operationalizes that concern for one fixed source model and two FPR caps, but it does not quantify acceptance optimism: all retrospective gates also failed the recall target. Any follow-up method would still require sequential-risk-control baselines, assumption analysis for dependent traffic, and an independent locked stream.",
        "",
        "## Reproducibility",
        "",
        f"- Script: `run_iotid20_cse_prefix_vs_fullperiod_threshold_gate.py`",
        f"- Output: `results/{OUTPUT.name}`",
        f"- Score cache: `results/{SCORE_CACHE.name}`; generated from the pinned v2 cache manifest and source-model hash.",
        f"- Target: {result['target']['rows']:,} rows, {result['target']['chunks']} chunks, {len(result['target']['captures'])} captures.",
        f"- One-pass model inference plus feature-signature hashing took {result['score_cache']['build_seconds']:.2f} s; compressed cache writing took {result['score_cache']['write_seconds']:.2f} s on this CPU environment (hardware is not normalized across studies).",
        "- The score cache can be reused only when both the source manifest and model SHA-256 values match; `--rebuild` forces regeneration.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def make_plot(result: dict) -> None:
    colors = {"prefix_only": "#1769aa", "full_period_retrospective": "#d95f02", "fixed_0_5": "#555555"}
    markers = {0.25: "o", 0.50: "s", 0.75: "^"}
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    for gate, label in (("prefix_only", "Prefix-only gate"),
                        ("full_period_retrospective", "Full-period retrospective oracle")):
        for fraction in CHECKPOINT_FRACTIONS:
            for cap, alpha in ((0.001, 1.0), (0.01, 0.55)):
                s = result["summary"][f"cap={cap};checkpoint={fraction};gate={gate}"]
                ax.scatter(s["pooled_future_fpr"], s["pooled_future_attack_recall"],
                           color=colors[gate] if cap == 0.001 else "white",
                           marker=markers[fraction], alpha=alpha,
                           s=78, edgecolor=colors[gate], linewidth=1.5, label="_nolegend_")
            b = result["summary"][f"cap=0.001;checkpoint={fraction};gate={gate}"]["pooled_fixed_threshold_0_5"]
            if gate == "prefix_only":
                ax.scatter(b["fpr"], b["attack_recall"], color=colors["fixed_0_5"],
                           marker=markers[fraction], s=64, edgecolor="white", linewidth=0.6,
                           label="_nolegend_")
    ax.axvline(0.001, color="#7b3294", linestyle="--", linewidth=1.4)
    ax.axvline(0.01, color="#008837", linestyle=":", linewidth=1.6)
    ax.axhline(TARGET_RECALL, color="#333333", linestyle="-.", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 1.25)
    ax.set_ylim(-0.025, 1.02)
    ax.set_xlabel("Future false-positive rate (held-out feature-signature groups)")
    ax.set_ylabel("Future attack recall")
    ax.set_title("Threshold-only adaptation has no safe operating point on IoTID20→CSE")
    ax.grid(True, which="both", color="#dddddd", linewidth=0.6, alpha=0.75)
    legend_items = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors["prefix_only"],
               markeredgecolor=colors["prefix_only"], label="Prefix-only gate"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors["full_period_retrospective"],
               markeredgecolor=colors["full_period_retrospective"], label="Full-period retrospective oracle"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors["fixed_0_5"],
               markeredgecolor=colors["fixed_0_5"], label="Fixed threshold 0.5"),
        Line2D([0], [0], marker=markers[0.25], color="#333333", linestyle="none", label="25% checkpoint"),
        Line2D([0], [0], marker=markers[0.50], color="#333333", linestyle="none", label="50% checkpoint"),
        Line2D([0], [0], marker=markers[0.75], color="#333333", linestyle="none", label="75% checkpoint"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#555555",
               markeredgecolor="#555555", label="Project cap 0.1% (filled)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="#555555", label="Comparator cap 1% (open)"),
        Line2D([0], [0], color="#7b3294", linestyle="--", label="Project FPR cap 0.1%"),
        Line2D([0], [0], color="#008837", linestyle=":", label="Comparator FPR cap 1%"),
        Line2D([0], [0], color="#333333", linestyle="-.", label="Recall target 90%"),
    ]
    ax.legend(handles=legend_items, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8.5, frameon=True, title="Gate / checkpoint / cap")
    fig.savefig(PLOT, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="recompute predictions/hash folds despite a valid score cache")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
