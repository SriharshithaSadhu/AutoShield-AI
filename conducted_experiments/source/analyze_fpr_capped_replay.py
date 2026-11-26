"""Summarize and audit the development FPR-capped CSE replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def attack_recall(summary: dict[str, Any]) -> float:
    counts = summary["global_confusion_counts"]
    return counts["tp"] / max(1, counts["tp"] + counts["fn"])


def main(source: Path, output: Path, fpr_cap: float = 0.001) -> dict[str, Any]:
    result = json.loads(source.read_text(encoding="utf-8"))
    audit = result["run_audit"]
    if not audit["complete_official_collection"] or audit["replayed_rows"] != 15_184_359:
        raise ValueError("Replay does not match the audited complete CSE stream")
    if audit["replayed_chunk_count"] != 383:
        raise ValueError("Expected the full 40k-target replay (383 chunks)")

    summaries = result["summaries"]
    baseline = summaries["no_adaptation"]
    capped = summaries["threshold_fpr_capped"]
    records = capped["chunk_records"]
    guard_records = [row["fpr_guard_validation"] for row in records
                     if "fpr_guard_validation" in row]
    validation_wilson = [row["validation_fpr_wilson_upper"] for row in guard_records]
    observed_chunk_fpr = [row["false_positive_rate"] for row in records]
    output_record = {
        "source_result": str(source),
        "classification": "development_only; same CSE stream used for policy development",
        "run_audit": audit,
        "fpr_cap": fpr_cap,
        "policies": {},
        "fpr_capped_recent_window_guard": {
            "windows_with_diagnostics": len(guard_records),
            "windows_meeting_wilson_upper_cap": sum(x <= fpr_cap for x in validation_wilson),
            "fraction_meeting_wilson_upper_cap": (
                sum(x <= fpr_cap for x in validation_wilson) / len(validation_wilson)
                if validation_wilson else None
            ),
            "maximum_validation_wilson_upper": max(validation_wilson, default=None),
            "mean_validation_fpr": (
                sum(row["validation_fpr"] for row in guard_records) / len(guard_records)
                if guard_records else None
            ),
            "future_replay_chunks_over_cap": sum(x > fpr_cap for x in observed_chunk_fpr),
            "future_replay_chunk_over_cap_fraction": (
                sum(x > fpr_cap for x in observed_chunk_fpr) / len(observed_chunk_fpr)
                if observed_chunk_fpr else None
            ),
            "mean_observed_chunk_fpr": (
                sum(observed_chunk_fpr) / len(observed_chunk_fpr) if observed_chunk_fpr else None
            ),
        },
        "interpretation": (
            "The Wilson restriction is imposed on the recent labeled window when choosing the next threshold. "
            "It is not a guarantee for future or pooled traffic. Compare pooled and per-chunk FPR with attack recall."
        ),
    }
    for name in ("no_adaptation", "feature_psi", "threshold_recalibration", "threshold_fpr_capped"):
        summary = summaries[name]
        output_record["policies"][name] = {
            "pooled_weighted_f1": summary["global_weighted_f1_from_confusion"],
            "pooled_macro_f1": summary["global_macro_f1_from_confusion"],
            "pooled_fpr": summary["global_false_positive_rate"],
            "attack_recall": attack_recall(summary),
            "mean_chunk_macro_f1": summary["mean_chunk_macro_f1"],
            "mean_chunk_ece": summary["mean_chunk_ece"],
            "mean_chunk_fpr": summary["mean_chunk_false_positive_rate"],
            "model_updates": summary["model_updates"],
            "threshold_recalibrations": summary["threshold_recalibrations"],
            "threshold_recalibration_seconds": summary["total_threshold_recalibration_seconds"],
            "threshold_recalibration_seconds_summary": summary["threshold_recalibration_seconds_summary"],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_record, indent=2), encoding="utf-8")
    return output_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fpr-cap", type=float, default=0.001)
    args = parser.parse_args()
    report = main(args.source, args.output, args.fpr_cap)
    for policy, metrics in report["policies"].items():
        print(policy, json.dumps(metrics, sort_keys=True))
    print("recent_window_guard", json.dumps(report["fpr_capped_recent_window_guard"], sort_keys=True))
