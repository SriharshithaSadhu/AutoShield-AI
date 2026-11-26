"""Reconcile reviewer use and detection metrics for the early-review arms."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results/corrected_cse_gate"

def main():
    results = []
    baseline = json.loads((ROOT/"original_core6_lifecycle_delay4.json").read_text())
    for path in sorted(ROOT.glob("early_verification_*t[12].json")):
        run = json.loads(path.read_text())
        assert all(run["chunks"][name] == rows for name,rows in baseline["chunks"].items())
        by_budget = {}
        for row in run["early_verification"].values():
            by_budget.setdefault(row["budget_per_active_chunk"], set()).add(row["reviewed"])
        assert all(len(counts) == 1 for counts in by_budget.values())
        for name, review in run["early_verification"].items():
            result = run["summary"][name]
            assert review["reviewed"] <= 128 * len(run["chunks"][name])
            metric = dict(**result, missed=result["attacks"]-result["tp"],
                          precision=result["tp"]/max(1,result["tp"]+result["fp"]),
                          f1=2*result["tp"]/max(1,result["tp"]+result["fp"]+result["attacks"]))
            results.append(dict(run=path.name, policy=name, **review, **metric))
    (ROOT/"early_review_summary.json").write_text(json.dumps(results,indent=2))
    for row in results:
        print(row["run"],row["policy"],row["reviewed"],row["fp"],round(row["recall"]*100,2),
              round(row["worst_chunk_fpr"]*100,2),row["switches"])

if __name__ == "__main__":
    main()
