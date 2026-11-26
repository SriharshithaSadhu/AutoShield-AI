import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results"
output = []
for seed in (42, 43):
    path = root / f"nf_ton_iot_v3_promotion_gate_comparison_seed{seed}.json"
    replay = json.loads(path.read_text(encoding="utf-8"))
    for method in ("periodic", "bridge", "vbc_sg", "fpr_recall"):
        tenures = []
        for decision in replay["prospective_update_outcomes"]:
            interval = decision["deployed_interval_by_method"][method]
            if not interval or not interval["chunks_evaluated"]:
                continue
            chunks = replay["chunk_metrics"][interval["first_chunk"] - 1:interval["last_chunk"]]
            breaches = [i for i, chunk in enumerate(chunks, 1)
                        if chunk["arms"][method]["false_positive_rate"] > 0.001]
            tenures.append((len(chunks), breaches))
        horizons = {}
        for horizon in (1, 5, 10, 20):
            eligible = [breaches for length, breaches in tenures if length >= horizon]
            horizons[horizon] = {
                "complete": len(eligible),
                "breach": sum(any(i <= horizon for i in b) for b in eligible),
            }
        output.append(dict(seed=seed, method=method, horizons=horizons,
                           tenures=len(tenures),
                           breach_after_five=sum(bool(b) and b[0] > 5 for _, b in tenures),
                           exposed_chunks=sum(n for n, _ in tenures),
                           over_cap_chunks=sum(len(b) for _, b in tenures)))
(root / "nf_ton_promotion_exposure_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
for row in output:
    print(row)
