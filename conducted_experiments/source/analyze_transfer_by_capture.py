"""Aggregate the IoTID20-to-CSE replay by capture file for diagnostics.

This is a descriptive analysis of an already-completed development replay. It
does not refit models, alter predictions, or treat capture files as independent
confirmatory datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def summarize(rows: list[dict]) -> dict:
    total = sum(int(r["rows"]) for r in rows)
    tn = sum(int(r["tn"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    tp = sum(int(r["tp"]) for r in rows)
    benign = tn + fp
    attack = fn + tp
    precision_benign = tn / (tn + fn) if tn + fn else 0.0
    recall_benign = tn / benign if benign else 0.0
    precision_attack = tp / (tp + fp) if tp + fp else 0.0
    recall_attack = tp / attack if attack else 0.0
    f1_benign = (
        2 * precision_benign * recall_benign / (precision_benign + recall_benign)
        if precision_benign + recall_benign else 0.0
    )
    f1_attack = (
        2 * precision_attack * recall_attack / (precision_attack + recall_attack)
        if precision_attack + recall_attack else 0.0
    )
    return {
        "rows": total,
        "chunks": len(rows),
        "attack_prevalence": attack / total if total else 0.0,
        "weighted_f1": (f1_benign * benign + f1_attack * attack) / total if total else 0.0,
        "macro_f1": (f1_benign + f1_attack) / 2,
        "attack_recall": recall_attack,
        "benign_recall": recall_benign,
        "false_positive_rate": fp / benign if benign else 0.0,
        "mean_chunk_macro_f1": sum(float(r["macro_f1"]) for r in rows) / len(rows),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/iotid20_cse_transfer_replay_seed42.json"))
    parser.add_argument("--output", type=Path, default=Path("results/iotid20_cse_transfer_by_capture_seed42.csv"))
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    arms = payload["records"]
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for arm, records in arms.items():
        for record in records:
            groups[record["capture_file"]][arm].append(record)

    output_rows = []
    for capture, capture_arms in groups.items():
        for arm, records in capture_arms.items():
            row = {"capture_file": capture, "arm": arm, **summarize(records)}
            output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(output_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} capture/arm summaries to {args.output}")
    print("capture_file,arm,rows,weighted_f1,macro_f1,attack_recall,false_positive_rate,mean_chunk_macro_f1")
    for row in output_rows:
        print(",".join(str(row[k]) for k in (
            "capture_file", "arm", "rows", "weighted_f1", "macro_f1",
            "attack_recall", "false_positive_rate", "mean_chunk_macro_f1"
        )))


if __name__ == "__main__":
    main()
