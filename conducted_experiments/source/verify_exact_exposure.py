"""Check exposure-budget runs against frozen baselines and realized risk."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results/corrected_cse_gate"

def main():
    results = []
    for delay, tag in ((0, "locked"), (4, "delay4")):
        run = json.loads((ROOT / f"exact_exposure_delay{delay}.json").read_text())
        baseline = json.loads((ROOT / f"original_core6_lifecycle_{tag}.json").read_text())
        assert all(run["chunks"][name] == rows for name, rows in baseline["chunks"].items())
        frozen = run["chunks"]["frozen"]
        for key, state in run["exposure"].items():
            rows = run["chunks"][key]
            increments = [max(0, row["fp"]-ref["fp"]) for row, ref in zip(rows, frozen, strict=True)]
            peak_realized = max(sum(increments[max(0, i-delay):i+1]) for i in range(len(rows)))
            assert peak_realized <= state["peak"] <= state["budget"]
            results.append(dict(delay=delay, policy=key, peak_realized_increment=peak_realized,
                                peak_reserved=state["peak"], **run["summary"][key]))
    (ROOT / "exact_exposure_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for row in results:
        print(row["delay"], row["policy"], row["fp"], round(row["recall"]*100, 2),
              row["peak_realized_increment"], row["peak_reserved"])

if __name__ == "__main__":
    main()
