"""Verify policy lifecycle outputs and summarize post-approval exposure."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--rollback-reference", type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding="utf-8"))
    delay = data["label_delay_chunks"]
    lines = [f"# Deployment lifecycle comparison, label delay {delay}", "",
             "| Policy | False alarms | FPR | Recall | Worst-chunk FPR | Chunks above cap |",
             "|---|---:|---:|---:|---:|---:|"]
    for policy, summary in data["summary"].items():
        for metric in ("fp", "tp", "attacks", "benign"):
            assert sum(row[metric] for row in data["chunks"][policy]) == summary[metric]
        lines.append(f'| {policy} | {summary["fp"]:,} | {summary["fpr"]:.3%} | '
                     f'{summary["recall"]:.2%} | {summary["worst_chunk_fpr"]:.3%} | '
                     f'{summary["chunks_over_cap"]} |')
    if args.rollback_reference:
        reference = json.loads(args.rollback_reference.read_text(encoding="utf-8"))
        first_restore = next(event for event in data["lifecycle"]
                             if event["policy"] == "rollback" and event["action"] == "restore")
        breach = reference["rollbacks"][0]
        assert first_restore["after_chunk"] == breach["after_chunk"]
        deployment = next(event["after_chunk"] + 1 for event in data["lifecycle"]
                          if event["policy"] == "rollback" and event["action"] == "approve")
        assert deployment <= breach["after_chunk"]
        for row, gated in zip(data["chunks"]["rollback"], data["chunks"]["gated"], strict=True):
            if row["chunk"] <= breach["after_chunk"]:
                assert row == gated
            if row["chunk"] == breach["after_chunk"]:
                assert row["fp"] == breach["observed_fp"] and row["benign"] == breach["observed_benign"]
        lines += ["", "Integrated rollback reproduces the original deployment through the first breach.",
                  "It then continues to evaluate new approvals, so full-stream counts differ from the single-approval diagnostic."]
    lines += ["", "## Traffic while an approved update was active", "",
              "| Policy | Approved intervals | Benign | Attacks | Recall | Frozen recall on same rows | Alarms from first breach through restoration |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for policy in ("rollback", "lease"):
        events = [event for event in data["lifecycle"] if event["policy"] == policy]
        starts = [event["after_chunk"] + 1 for event in events if event["action"] == "approve"]
        stops = [event["after_chunk"] + 1 for event in events if event["action"] == "restore"]
        rows = data["chunks"][policy]
        frozen = {row["chunk"]: row for row in data["chunks"]["frozen"]}
        alarms_after_breach = attacks = benign = tp = frozen_tp = 0
        for index, start in enumerate(starts):
            stop = stops[index] if index < len(stops) else rows[-1]["chunk"] + 1
            assert start < stop
            interval = [row for row in rows if start <= row["chunk"] < stop]
            breached = False
            for row in interval:
                attacks += row["attacks"]
                benign += row["benign"]
                tp += row["tp"]
                frozen_tp += frozen[row["chunk"]]["tp"]
                breached |= row["benign"] >= 3000 and 1000 * row["fp"] > row["benign"]
                if breached:
                    alarms_after_breach += row["fp"]
        recall = f"{tp / attacks:.2%}" if attacks else "Undefined"
        frozen_recall = f"{frozen_tp / attacks:.2%}" if attacks else "Undefined"
        lines.append(f"| {policy} | {len(starts)} | {benign:,} | {attacks:,} | {recall} | "
                     f"{frozen_recall} | {alarms_after_breach:,} |")
        for event in events:
            if event["action"] == "approve":
                assert event["last_released_chunk"] == event["after_chunk"] - delay
            if event["action"] == "restore":
                assert event["observed_chunk"] == event["after_chunk"] - delay
            if event["action"] in ("renew", "expire"):
                assert all(chunk <= event["after_chunk"] - delay for chunk in event["evidence_chunks"])
    fits = [event for event in data["events"] if "shared_candidate_and_gate_seconds" in event]
    shared_seconds = sum(event["shared_candidate_and_gate_seconds"] for event in fits)
    lines += ["", f"Shared candidate fits: {len(fits)}. Recorded fit, calibration and approval-only "
              f"validation time: {shared_seconds:.2f} wall seconds. "
              "Concurrent workloads can affect timing. Lifecycle-specific compute was not isolated."]
    output = args.result.with_suffix(".md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
