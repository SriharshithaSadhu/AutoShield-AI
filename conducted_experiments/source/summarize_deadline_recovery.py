"""Verify recovery runs against matching baselines and calculate tail metrics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results/corrected_cse_gate"

def main():
    report = []
    for path in sorted(ROOT.glob("recovery_delay*_*.json")):
        data = json.loads(path.read_text())
        lag = data["label_delay_chunks"]
        base = json.loads((ROOT / f"batch_inference_delay{lag}.json").read_text())
        for policy, old_rows in base["chunks"].items():
            if policy != "deadline_shadow":
                assert data["chunks"][policy] == old_rows
        rows = data["chunks"]["deadline_shadow"]
        window_fpr = max(
            sum(row["fp"] for row in rows[start:start+5]) /
            sum(row["benign"] for row in rows[start:start+5])
            for start in range(len(rows)-4)
            if sum(row["benign"] for row in rows[start:start+5]))
        cfg = data["deadline_routing"]
        report.append(dict(run=path.name, label_delay=lag, recovery_flows=cfg["recovery_flows"],
                           fallback_events=len(cfg["switch_completion_ns"]),
                           recovery_events=len(cfg["recovery_completion_ns"]),
                           worst_five_chunk_fpr=window_fpr,
                           routing_wall_seconds=cfg["routing_wall_seconds"],
                           buffering_p95_seconds=cfg["p95_wait_seconds"],
                           suppressed_tp=cfg["source_true_positives_suppressed"],
                           **data["summary"]["deadline_shadow"]))
    (ROOT / "deadline_recovery_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
