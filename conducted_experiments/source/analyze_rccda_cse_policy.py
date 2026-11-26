"""Capture-level breakdown for the RCCDA-inspired CSE replay."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "cse_rccda_inspired_equal_update_baselines_seed42.json"
OUTPUT = ROOT / "results" / "cse_rccda_inspired_equal_update_baselines_seed42_by_capture.csv"


def main() -> None:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = []
    for policy, summary in data["summaries"].items():
        groups = defaultdict(list)
        for record in summary["chunk_records"]:
            groups[record["capture_file"]].append(record)
        for capture, records in groups.items():
            tn = sum(r["tn"] for r in records)
            fp = sum(r["fp"] for r in records)
            fn = sum(r["fn"] for r in records)
            tp = sum(r["tp"] for r in records)
            attack_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            benign_f1 = 2 * tn / max(1, 2 * tn + fp + fn)
            n = tn + fp + fn + tp
            attack_rows = sum(r["attack_rows"] for r in records)
            rows.append({
                "policy": policy,
                "capture_file": capture,
                "chunks": len(records),
                "rows": n,
                "attack_prevalence": attack_rows / max(1, n),
                "macro_f1": (attack_f1 + benign_f1) / 2,
                "weighted_f1": ((tp + fn) * attack_f1 + (tn + fp) * benign_f1) / max(1, n),
                "fpr": fp / max(1, fp + tn),
                "attack_recall": tp / max(1, tp + fn),
                "updates": sum(bool(r["update_requested"]) for r in records),
                "mean_chunk_fpr": sum(r["false_positive_rate"] for r in records) / max(1, len(records)),
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Capture-level summaries (policy, capture, macro-F1, FPR, attack recall, update count):")
    for r in rows:
        if r["policy"] in {"rccda_paper_estimator_port", "periodic_10pct", "random_bernoulli_10pct"}:
            print(r["policy"], r["capture_file"], f"{r['macro_f1']:.4f}",
                  f"{r['fpr']:.6f}", f"{r['attack_recall']:.4f}", r["updates"])
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
