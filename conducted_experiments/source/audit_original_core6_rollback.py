"""Reconstruct a causal rollback for a replay containing exactly one approval."""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results/corrected_cse_gate"


def main():
    data = json.loads((RESULTS / "original_core6_policy_comparison.json").read_text(encoding="utf-8"))
    approvals = [event["after_chunk"] + 1 for event in data["events"] if event.get("accepted")]
    if len(approvals) != 1:
        raise ValueError("Saved-score reconstruction requires exactly one approval.")
    frozen = data["chunks"]["frozen"]
    gated = data["chunks"]["gated"]
    active = False
    rows = []
    rollbacks = []
    for incumbent, candidate in zip(frozen, gated, strict=True):
        assert all(incumbent[key] == candidate[key] for key in ("chunk", "attacks", "benign"))
        chunk = incumbent["chunk"]
        if chunk == approvals[0]:
            active = True
        rows.append(candidate if active else incumbent)
        if active and candidate["benign"] >= 3000 and 1000 * candidate["fp"] > candidate["benign"]:
            rollbacks.append(dict(after_chunk=chunk, observed_fp=candidate["fp"],
                                  observed_benign=candidate["benign"]))
            active = False
    counts = {key: sum(row[key] for row in rows) for key in ("fp", "tp", "attacks", "benign")}
    report = dict(
        scope="Post-hoc development diagnostic with one approval and next-chunk rollback.",
        rule="Return to frozen model after the first scored chunk exceeding 0.1% FPR with 3000 benign labels.",
        counts=counts, fpr=counts["fp"] / counts["benign"], recall=counts["tp"] / counts["attacks"],
        chunks_over_cap=sum(1000 * row["fp"] > row["benign"] for row in rows),
        worst_chunk_fpr=max(row["fp"] / row["benign"] for row in rows if row["benign"]),
        rollbacks=rollbacks,
    )
    (RESULTS / "original_core6_rollback_diagnostic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
