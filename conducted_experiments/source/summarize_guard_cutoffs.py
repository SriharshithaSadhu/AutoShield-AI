"""Summarize action exposure in corrected-CSE sensitivity runs."""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results/corrected_cse_gate"


def summarize(path):
    run = json.loads(path.read_text(encoding="utf-8"))
    actions = [event for event in run["events"] if event["guard_action"] != "keep"]
    frozen = {row["chunk"]: row for row in run["chunks"]["frozen"]}
    guarded = {row["chunk"]: row for row in run["chunks"]["recall_guard"]}
    assert frozen.keys() == guarded.keys()
    intervals = []
    for index, action in enumerate(actions):
        start = action["after_chunk"] + 1
        stop = actions[index + 1]["after_chunk"] + 1 if index + 1 < len(actions) else max(frozen) + 1
        chunks = range(start, stop)
        attacks = sum(frozen[chunk]["attacks"] for chunk in chunks)
        benign = sum(frozen[chunk]["benign"] for chunk in chunks)
        intervals.append({
            "action": action["guard_action"], "start": start, "stop_exclusive": stop,
            "attacks": attacks, "benign": benign,
            "frozen_tp": sum(frozen[chunk]["tp"] for chunk in chunks),
            "guard_tp": sum(guarded[chunk]["tp"] for chunk in chunks),
            "false_alarms_avoided": sum(frozen[chunk]["fp"] - guarded[chunk]["fp"] for chunk in chunks),
        })
    return {"evaluation_start": run["evaluation_start_chunk"], "policies": run["summary"],
            "scheduled_decisions": len(run["events"]), "accepted_actions": intervals}


def main():
    paths = [path for path in RESULTS.glob("guard_cutoff_*.json")
             if path.stem.removeprefix("guard_cutoff_").isdigit()]
    paths.append(RESULTS / "adaptation_recall_guard_dev.json")
    paths += [RESULTS / "adaptation_recall_guard_original_cutoff.json"]
    runs = sorted((summarize(path) for path in paths), key=lambda row: row["evaluation_start"])
    result = {"scope": "Overlapping cutoffs on one explored capture. Descriptive sensitivity only.",
              "runs": runs}
    (RESULTS / "guard_cutoff_sensitivity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for run in runs:
        print(run["evaluation_start"], run["scheduled_decisions"], run["accepted_actions"])


if __name__ == "__main__":
    main()
