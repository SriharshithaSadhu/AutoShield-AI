"""Compare buffered label-free routing rules on saved policy predictions."""
import json
from collections import deque
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent / "results/corrected_cse_gate"


def evaluate(data, method, limit):
    history = deque(maxlen=3)
    approvals = {e["after_chunk"] + 1 for e in data["events"] if e.get("accepted")}
    active = False
    latched = False
    totals = dict(fp=0, tp=0, benign=0, attacks=0)
    switches = []
    for frozen, gated in zip(data["chunks"]["frozen"], data["chunks"]["gated"], strict=True):
        assert all(frozen[k] == gated[k] for k in ("chunk", "benign", "attacks"))
        chunk = frozen["chunk"]
        if chunk in approvals:
            active, latched = True, False
            history.clear()
        n = frozen["benign"] + frozen["attacks"]
        rate = (gated["fp"] + gated["tp"]) / n
        incumbent = (frozen["fp"] + frozen["tp"]) / n
        excess = rate - incumbent
        signal = rate if method == "rolling_rate" else excess
        if active:
            if method == "absolute":
                latched = excess > limit
            elif len(history) == history.maxlen and signal - median(history) > limit:
                latched = True
            history.append(signal)
        chosen = frozen if latched else gated
        if latched:
            switches.append(chunk)
        for key in totals:
            totals[key] += chosen[key]
    return dict(**totals, fpr=totals["fp"]/totals["benign"], recall=totals["tp"]/totals["attacks"],
                fallback_chunks=len(switches), first_switch=switches[0] if switches else None)


def main():
    configurations = [("absolute", .01), ("rolling_rate", .10), ("rolling_excess", .10)]
    results = {}
    for delay, tag in [(0, "locked"), (1, "delay1"), (2, "delay2"), (4, "delay4")]:
        data = json.loads((ROOT / f"original_core6_lifecycle_{tag}.json").read_text())
        results[delay] = {method: evaluate(data, method, limit) for method, limit in configurations}
    report = dict(scope="Exploratory buffered routing over saved shadow predictions. Uses predicted-positive counts before routing, labels only for reporting.",
                  configurations=configurations, results=results)
    (ROOT / "rolling_alert_safeguards.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
