"""Check and render the exploratory original-CSE comparison."""
import json
from pathlib import Path

from summarize_guard_cutoffs import summarize

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/corrected_cse_gate"
SOURCE = RESULTS / "original_core6_policy_comparison.json"


def main():
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    report = summarize(SOURCE)
    lines = ["# Original CSE six-field chronological comparison", "",
             "Exploratory replication across capture days using original CSE labels and a six-field projection.",
             "Replay uses estimated completion time. Some original labels and flow extraction behavior differ from the corrected release.",
             "", "| Policy | Pooled FPR | Recall | Worst chunk FPR | Chunks above cap |",
             "|---|---:|---:|---:|---:|"]
    for policy, scores in report["policies"].items():
        for key in ("benign", "attacks", "fp", "tp"):
            assert sum(row[key] for row in raw["capture_counts"][policy].values()) == scores[key]
        lines.append(f'| {policy} | {scores["fpr"]:.4%} | {scores["recall"]:.2%} | '
                     f'{scores["worst_chunk_fpr"]:.3%} | {scores["chunks_over_cap"]} |')
    lines += ["", "## Outcomes by capture", "",
              "| Capture | Policy | Benign | Attacks | FPR | Recall |",
              "|---|---|---:|---:|---:|---:|"]
    for policy, days in raw["capture_counts"].items():
        for day, counts in sorted(days.items()):
            fpr = f'{counts["fp"] / counts["benign"]:.4%}' if counts["benign"] else "Undefined"
            recall = f'{counts["tp"] / counts["attacks"]:.2%}' if counts["attacks"] else "Undefined"
            lines.append(f'| {day} | {policy} | {counts["benign"]:,} | {counts["attacks"]:,} | {fpr} | {recall} |')
    lines += ["", "## Exposure after accepted actions", "",
              "| Action | First chunk | Last chunk | Attacks | Frozen TP | Guarded TP | False alarms avoided |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["accepted_actions"]:
        lines.append(f'| {row["action"]} | {row["start"]} | {row["stop_exclusive"]-1} | {row["attacks"]:,} | '
                     f'{row["frozen_tp"]:,} | {row["guard_tp"]:,} | {row["false_alarms_avoided"]:,} |')
    if not report["accepted_actions"]:
        lines += ["", "The guard kept the incumbent at every scheduled decision."]
    (RESULTS / "ORIGINAL_CORE6_POLICY_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:13]))
    print(json.dumps(report["accepted_actions"], indent=2))


if __name__ == "__main__":
    main()
