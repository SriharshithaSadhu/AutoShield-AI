"""Summarize allocation ablation and matched-resource results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results/corrected_cse_gate"


def main():
    summary = {}
    for path in sorted(ROOT.glob("*decision_value_t*.json")):
        data = json.loads(path.read_text())
        by_budget = {}
        for name, cost in data["early_verification"].items():
            by_budget.setdefault(cost["budget_per_active_chunk"], set()).add(cost["reviewed"])
        assert all(len(totals) == 1 for totals in by_budget.values())
        rows = {}
        for name in ["frozen", "rollback", *data["early_verification"]]:
            record = data["summary"][name]
            chunks = data["chunks"][name]
            assert all(sum(row[key] for row in chunks) == record[key]
                       for key in ("fp", "tp", "attacks", "benign"))
            worst = 0
            for start in range(len(chunks)-4):
                window = chunks[start:start+5]
                benign = sum(row["benign"] for row in window)
                if benign:
                    worst = max(worst, sum(row["fp"] for row in window)/benign)
            rows[name] = dict(**record, missed=record["attacks"]-record["tp"],
                              utility=record["tp"]-record["fp"], worst_five_chunk_fpr=worst,
                              **data["early_verification"].get(name, {}))
        summary[path.stem] = rows
        for budget in by_budget:
            val = rows[f"review_value_{budget}"]
            equal = rows[f"review_balanced_value_{budget}"]
            print(path.stem, budget, (val["tp"], val["fp"]), (equal["tp"], equal["fp"]),
                  "utility gain", val["utility"]-equal["utility"])
    (ROOT/"decision_value_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

if __name__ == "__main__":
    main()
