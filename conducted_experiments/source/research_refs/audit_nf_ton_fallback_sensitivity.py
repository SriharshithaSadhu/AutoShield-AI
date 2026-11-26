"""Quantify hindsight coverage and a one-chunk delayed selector on saved scores."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results"
out = []
for seed in (42, 43):
    stem = f"nf_ton_iot_v3_promotion_gate_comparison_seed{seed}"
    paths = [root / f"{stem}_shadow_previous{suffix}.json" for suffix in ("", ".partial")]
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        continue
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d.get("chunk_metrics", d.get("chunk_rows"))
    original = json.loads((root / f"{stem}.json").read_text(encoding="utf-8"))
    for a, b in zip(rows, original["chunk_metrics"], strict=len(rows)==len(original["chunk_metrics"])):
        assert a["chunk"] == b["chunk"]
        for arm in ("fpr_recall", "no_update"):
            assert all(a["arms"][arm][k] == b["arms"][arm][k] for k in ("tn","fp","fn","tp"))
    starts = [r["after_chunk"] for r in d.get("candidate_decisions", d.get("decisions"))
              if r["decisions"]["fpr_recall"]["accept"]]
    def models(row):
        return {"previous": row["previous_shadow"], "frozen": row["arms"]["no_update"]}
    for tolerance in (0, .01, .05, .10):
        eligible = covered = previous_only = attempted = successful = 0
        fp_delta = tp_delta = recall_violations = 0
        def feasible(row, model):
            active = row["arms"]["fpr_recall"]
            support = active["tp"] + active["fn"]
            benign = model["fp"] + model["tn"]
            return (support > 0 and benign > 0 and model["fp"] <= .001 * benign
                    and model["tp"] >= active["tp"] - tolerance * support)
        for start, end in zip(starts, starts[1:] + [len(rows)]):
            breach = False
            previous_row = None
            for row in rows[start:end]:
                active = row["arms"]["fpr_recall"]
                if breach:
                    choices = [name for name, m in models(previous_row).items() if feasible(previous_row, m)]
                    chosen = min(choices, key=lambda n: models(previous_row)[n]["fp"]) if choices else None
                    if chosen:
                        attempted += 1
                        selected = models(row)[chosen]
                        fp_delta += selected["fp"] - active["fp"]
                        tp_delta += selected["tp"] - active["tp"]
                        recall_violations += selected["tp"] < active["tp"] - tolerance * (active["tp"] + active["fn"])
                    if active["false_positive_rate"] > .001 and active["tp"] + active["fn"] > 0:
                        options = {name for name, m in models(row).items() if feasible(row, m)}
                        eligible += 1
                        covered += bool(options)
                        previous_only += options == {"previous"}
                        successful += chosen in options
                breach |= active["false_positive_rate"] > .001
                previous_row = row
        out.append(dict(seed=seed, chunks=len(rows), complete=path==paths[0],
                        recall_loss_tolerance=tolerance, eligible=eligible, oracle_covered=covered,
                        previous_only=previous_only, selector_attempts=attempted,
                        selector_successful_repairs=successful,
                        no_feasible_fallback=eligible-covered, missed_feasible=covered-successful,
                        selector_fp_delta=fp_delta, selector_tp_delta=tp_delta,
                        selector_recall_violations=recall_violations))
(root / "nf_ton_fallback_sensitivity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
for row in out:
    print(row)
