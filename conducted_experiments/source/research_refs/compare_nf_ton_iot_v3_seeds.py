"""Create a fixed, descriptive comparison of the locked NF-v3 seed-42/43 replays."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "nf_ton_iot_v3_seed42_seed43_comparison.md"


def load(seed: int) -> dict:
    path = ROOT / "results" / f"nf_ton_iot_v3_fixed_model_replay_seed{seed}.json"
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def pct(value: float) -> str:
    return f"{100 * value:.4f}%"


def main() -> None:
    data = {seed: load(seed) for seed in (42, 43)}
    lines = [
        "# NF-ToN-IoT-v3 seed-42/43 replay comparison",
        "",
        "This is a descriptive sensitivity comparison on the same inspected data stream, split, feature schema, thresholds, chunking, update policy, and 21,677,227 replay rows. Only the sampling/model seed differs. It is not an independent dataset replication and does not support a population-level confidence interval.",
        "",
        "## Pooled primary operating point",
        "",
        "| Seed | Arm | FPR | Attack recall | Macro F1 | FP | TP | Updates | Fit CPU seconds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in (42, 43):
        d = data[seed]
        for arm in ("frozen", "periodic_refit"):
            m = d["pooled_metrics"][f"{arm}:fpr_0p1_percent"]
            updates = d["updates"] if arm == "periodic_refit" else []
            fit_s = sum(item.get("fit_seconds", 0.0) for item in updates)
            lines.append(
                f"| {seed} | {arm} | {pct(m['false_positive_rate'])} | "
                f"{100*m['attack_recall']:.3f}% | {m['macro_f1']:.4f} | "
                f"{m['fp']:,} | {m['tp']:,} | {len(updates)} | {fit_s:.2f} |"
            )

    lines += ["", "## Daily false-positive range for periodic refitting", "", "| Seed | Minimum | Median | Maximum |", "|---:|---:|---:|---:|"]
    for seed in (42, 43):
        vals = [
            item["false_positive_rate"]
            for item in data[seed]["per_day_metrics"]
            if item["arm"] == "periodic_refit" and item["threshold"] == "fpr_0p1_percent"
        ]
        lines.append(f"| {seed} | {pct(min(vals))} | {pct(median(vals))} | {pct(max(vals))} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "Across both seeds, periodic refitting raises pooled attack recall from about 1.4–1.6% to about 75.4% while raising pooled FPR from about 0.016–0.018% to about 1.394–1.396%. The highest daily FPR remains above 22% in both runs. This makes the measured detection trade-off insensitive to the two tested initialization/sampling seeds on this stream; it does not establish generality beyond the same inspected ToN-IoT-family collection.",
        "",
        "Refit CPU time is less stable: the summed measured fit time is 404.22 seconds for seed 42 and 774.95 seconds for seed 43, despite the same 43-update schedule. Treat these timings as observed run measurements, not a stable cost estimate; investigate the long seed-43 refit(s) before using runtime as a comparative claim.",
        "",
        "The result supports testing an update-acceptance policy as a hypothesis, but it does not show that a gate would have rejected harmful updates. No candidate acceptance gate was run here. The streams' inspected attack schedule, shared data source, and only two seeds limit the evidence.",
        "",
        "Source protocols: `NF_TON_IOT_V3_REPLAY_PROTOCOL_2026-11-05.md` and `NF_TON_IOT_V3_SEED43_REPLICATION_PROTOCOL_2026-11-05.md`. Seed-specific outputs and analyses are in the adjacent `nf_ton_iot_v3_fixed_model_replay_seed{42,43}*` files.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
