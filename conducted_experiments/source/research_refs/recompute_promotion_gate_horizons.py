"""Rebuild prospective risk windows from saved full-chunk replay results.

This performs no model fitting. The next-five-chunk horizon includes the chunk
on which the next proposal is evaluated. The deployed interval for an accepted
candidate continues through that next acceptance chunk, whose prediction is
still made by the incumbent candidate, or through stream end if none follows.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHODS = ("periodic", "bridge", "vbc_sg", "fpr_recall")
CAP = 0.001


def summarize(rows: list[dict]) -> dict:
    fprs = [float(row["false_positive_rate"]) for row in rows]
    recalls = [float(row["attack_recall"]) for row in rows]
    return {
        "chunks_evaluated": len(rows),
        "max_chunk_fpr": max(fprs, default=None),
        "mean_chunk_fpr": sum(fprs) / len(fprs) if fprs else None,
        "mean_chunk_recall": sum(recalls) / len(recalls) if recalls else None,
        "over_cap_chunk_count": sum(value > CAP for value in fprs),
        "any_future_chunk_over_fpr_cap": any(value > CAP for value in fprs),
    }


def main() -> None:
    for seed in (42, 43):
        path = RESULTS / f"nf_ton_iot_v3_promotion_gate_comparison_seed{seed}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        decisions = data["candidate_decisions"]
        chunks = data["chunk_metrics"]
        n_chunks = len(chunks)
        outcomes = []
        for i, decision in enumerate(decisions):
            d = int(decision["after_chunk"])
            accepted = {m: bool(decision["decisions"][m]["accept"]) for m in METHODS}
            short_end = min(d + 5, n_chunks)
            short_rows = chunks[d:short_end]  # 0-index d is chunk d+1
            row = {
                "decision_chunk": d,
                "accepted": accepted,
                "validation_candidate_fpr": decision["validation_candidate"]["false_positive_rate"],
                "validation_candidate_recall": decision["validation_candidate"]["attack_recall"],
                "future_chunks_evaluated": len(short_rows),
                "future_by_method": {},
                "deployed_interval_by_method": {},
            }
            for method in METHODS:
                row["future_by_method"][method] = summarize(
                    [chunk["arms"][method] for chunk in short_rows])
                if not accepted[method]:
                    row["deployed_interval_by_method"][method] = None
                    continue
                next_accept = next((int(later["after_chunk"])
                                    for later in decisions[i + 1:]
                                    if later["decisions"][method]["accept"]), n_chunks)
                lo = d + 1
                hi = min(next_accept, n_chunks)
                interval = [chunks[idx - 1]["arms"][method]
                            for idx in range(lo, hi + 1)]
                row["deployed_interval_by_method"][method] = {
                    "first_chunk": lo if interval else None,
                    "last_chunk": hi if interval else None,
                    **summarize(interval),
                }
            outcomes.append(row)
        data["prospective_update_outcomes"] = outcomes
        data["prospective_outcome_horizon_note"] = (
            "next-five includes the incumbent-scored chunk on the next decision; "
            "deployed_interval follows an accepted model until the next accepted "
            "candidate's decision chunk (inclusive) or stream end."
        )
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

        summary_path = RESULTS / f"nf_ton_iot_v3_promotion_gate_comparison_seed{seed}.md"
        lines = [f"# NF-ToN-IoT-v3 causal promotion-gate pilot (seed {seed})", "",
                 "Development-only one-seed comparison. Candidates use only the three labeled chunks before the validation chunk; the current validation chunk is held out and scored before its labels are used for acceptance. Next-five outcomes include the chunk on which the next candidate is evaluated. Deployed-interval risk follows an accepted model until the next accepted candidate's decision chunk (inclusive), or stream end.", "",
                 "## Pooled results", "", "| Method | FPR | Attack recall | Macro F1 |", "|---|---:|---:|---:|"]
        for method, metrics in data["pooled_metrics"].items():
            lines.append(f"| {method} | {metrics['false_positive_rate']:.4%} | {metrics['attack_recall']:.2%} | {metrics['macro_f1']:.4f} |")
        lines += ["", "## Promotion decisions and deployed-interval harms", "",
                  "An interval is counted as harmful here if any future scored chunk during that accepted model's actual deployment interval exceeded the 0.1% FPR cap.", "",
                  "| Method | Accepted / proposed | Accepted intervals with any over-cap future chunk | Labels examined for gate | Gate time (s) |", "|---|---:|---:|---:|---:|"]
        for method in ("periodic", "bridge", "vbc_sg", "fpr_recall"):
            accepted_count = sum(o["accepted"][method] for o in outcomes)
            harm_count = sum(bool(o["deployed_interval_by_method"][method]["any_future_chunk_over_fpr_cap"])
                             for o in outcomes if o["accepted"][method])
            lines.append(f"| {method} | {accepted_count}/{len(outcomes)} | {harm_count} | {data['gate_label_use_total'].get(method, 0):,} | {data['gate_runtime_seconds'].get(method, 0.0):.3f} |")
        lines += ["", "## Limits", "",
                  "The pilot uses one inspected stream and one seed. BRIDGE's multinomial bootstrap and exact-binomial/McNemar bounds are shown as published decision baselines, not as valid guarantees for dependent flows. VBC-SG uses a later-time validation chunk here; the source paper's temporal runner uses older probe windows, so this is a causal protocol adaptation, not a bit-for-bit reproduction of its replay experiment. The FPR-cap gate's nominal confidence calculations do not account for serial dependence or arbitrary future drift.", "",
                  f"Machine-readable detail: `results/{path.name}`. Protocol runner: `research_refs/run_nf_ton_iot_v3_promotion_gate_pilot.py`."]
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"seed={seed}: refreshed {path.name} and {summary_path.name}")


if __name__ == "__main__":
    main()
