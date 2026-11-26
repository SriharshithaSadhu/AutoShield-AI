"""Measure frozen fallback trade-offs after a breach becomes observable."""
import json
from pathlib import Path

results = Path(__file__).resolve().parents[1] / "results"
cap = 0.001
output = {}
for seed in (42, 43):
    path = results / f"nf_ton_iot_v3_promotion_gate_comparison_seed{seed}.json"
    replay = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for decision in replay["prospective_update_outcomes"]:
        interval = decision["deployed_interval_by_method"]["fpr_recall"]
        if not interval or not interval["chunks_evaluated"]:
            continue
        chunks = replay["chunk_metrics"][interval["first_chunk"] - 1:interval["last_chunk"]]
        first = next((i for i, chunk in enumerate(chunks)
                      if chunk["arms"]["fpr_recall"]["false_positive_rate"] > cap), None)
        if first is None:
            continue
        later = [chunk["arms"] for chunk in chunks[first + 1:]]
        rows.append({
            "promotion_chunk": decision["decision_chunk"],
            "first_breach_chunk": interval["first_chunk"] + first,
            "remaining_chunks": len(later),
            "over_cap_chunks_avoided": sum(int(a["fpr_recall"]["false_positive_rate"] > cap)
                                           - int(a["no_update"]["false_positive_rate"] > cap) for a in later),
            "false_positives_avoided": sum(a["fpr_recall"]["fp"] - a["no_update"]["fp"] for a in later),
            "attack_detections_lost": sum(a["fpr_recall"]["tp"] - a["no_update"]["tp"] for a in later),
        })
    output[seed] = rows
(results / "nf_ton_frozen_fallback_diagnostic.json").write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(json.dumps(output))
