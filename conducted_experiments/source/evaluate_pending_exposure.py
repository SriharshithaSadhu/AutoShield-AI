"""Bound incremental false alarms on unresolved chunks using prediction marginals."""
from collections import deque
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parent / "results/corrected_cse_gate"


def replay(frozen, candidate, delay, budget):
    pending = deque()
    reserved = peak = fp = tp = accepts = 0
    for i, (base, new) in enumerate(zip(frozen, candidate, strict=True)):
        assert all(base[k] == new[k] for k in ("chunk", "benign", "attacks"))
        while pending and pending[0][0] < i-delay:
            _, charge = pending.popleft()
            reserved -= charge
        n = base["benign"] + base["attacks"]
        base_alerts = base["fp"] + base["tp"]
        new_alerts = new["fp"] + new["tp"]
        charge = min(new_alerts, n-base_alerts)
        if reserved + charge <= budget:
            chosen = new
            if charge:
                pending.append((i, charge))
            reserved += charge
            accepts += 1
        else:
            chosen = base
        peak = max(peak, reserved)
        fp += chosen["fp"]
        tp += chosen["tp"]
    assert peak <= budget
    return dict(fp=fp,tp=tp,peak_reserved=peak,candidate_chunks=accepts)


def main():
    results = {}
    for lag, tag in [(0,"locked"),(1,"delay1"),(2,"delay2"),(4,"delay4")]:
        run = json.loads((ROOT/f"original_core6_lifecycle_{tag}.json").read_text())
        base, candidate = (run["chunks"][key] for key in ("frozen","rollback"))
        results[lag] = {}
        for budget in (20,1000,10000,20000):
            counts = replay(base,candidate,lag,budget)
            counts["fpr"] = counts["fp"]/sum(r["benign"] for r in base)
            counts["recall"] = counts["tp"]/sum(r["attacks"] for r in base)
            results[lag][budget] = counts
    (ROOT/"pending_exposure_budget.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results[4],indent=2))

if __name__ == "__main__":
    main()
