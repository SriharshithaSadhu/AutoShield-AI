"""Pilot reserve selection on the archived AutoML replay, using released labels."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results"
output = []
for seed in (42, 43):
    stem = f"nf_ton_iot_v3_promotion_gate_comparison_seed{seed}"
    path = root / f"{stem}_shadow_previous.json"
    if not path.exists():
        continue
    replay = json.loads(path.read_text(encoding="utf-8"))
    saved = json.loads((root / f"{stem}.json").read_text(encoding="utf-8"))
    rows = replay["chunk_metrics"]
    def active(row):
        return row["arms"]["fpr_recall"]
    def score(row, name):
        return row["previous_shadow"] if name == "previous" else row["arms"]["no_update"]
    def feasible(row, name):
        reference, reserve = active(row), score(row, name)
        positives = reference["tp"] + reference["fn"]
        benign = reserve["tn"] + reserve["fp"]
        return (positives > 0 and benign > 0 and reserve["fp"] <= .001 * benign
                and reserve["tp"] >= reference["tp"] - .01 * positives)
    fields = ("tn", "fp", "fn", "tp")
    for row, reference in zip(rows, saved["chunk_metrics"], strict=True):
        for arm in ("fpr_recall", "no_update"):
            assert all(row["arms"][arm][key] == reference["arms"][arm][key] for key in fields)
    promotions = [d["after_chunk"] for d in replay["candidate_decisions"]
                  if d["decisions"]["fpr_recall"]["accept"]]
    options = ("previous", "frozen")
    counts = {name: dict(attempts=0, repairs=0, fp_delta=0, tp_delta=0,
                        recall_violations=0)
              for name in (*options, "coverage_ranked")}
    for start, end in zip(promotions, promotions[1:] + [len(rows)]):
        history = []
        breached = False
        for row in rows[start:end]:
            if breached:
                def rank(name):
                    return (sum(feasible(past, name) for past in history),
                            -sum(score(past, name)["fp"] for past in history))
                selected = max(options, key=rank)
                for policy, name in (("previous", "previous"), ("frozen", "frozen"),
                                     ("coverage_ranked", selected)):
                    if not feasible(history[-1], name):
                        continue
                    m, a, c = score(row, name), active(row), counts[policy]
                    c["attempts"] += 1
                    c["repairs"] += bool(a["false_positive_rate"] > .001 and feasible(row, name))
                    c["fp_delta"] += m["fp"] - a["fp"]
                    c["tp_delta"] += m["tp"] - a["tp"]
                    c["recall_violations"] += m["tp"] < a["tp"] - .01 * (a["tp"] + a["fn"])
            breached |= active(row)["false_positive_rate"] > .001
            history = (history + [row])[-5:]
    output.append(dict(seed=seed, chunks=len(rows), policies=counts))
(root / "nf_ton_coverage_ranked_reserve_pilot.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps(output))
