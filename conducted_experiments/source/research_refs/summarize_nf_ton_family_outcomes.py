"""Generate paired-seed family-level recall summary for NF-ToN-IoT-v3."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
POLICIES = {
    "Frozen fixed threshold": ("threshold", "fixed_threshold"),
    "Threshold only (3 chunks)": ("threshold", "threshold_only"),
    "Periodic refit": ("gate", "periodic"),
    "BRIDGE gate": ("gate", "bridge"),
    "VBC-SG gate": ("gate", "vbc_sg"),
    "FPR + recall gate": ("gate", "fpr_recall"),
}


def load(seed: int, kind: str) -> dict:
    if kind == "gate":
        path = RESULTS / f"nf_ton_iot_v3_promotion_gate_comparison_seed{seed}.json"
    else:
        path = RESULTS / f"nf_ton_iot_v3_threshold_only_window3_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    data = {seed: {name: load(seed, kind) for name, (kind, _) in POLICIES.items()}
            for seed in (42, 43)}
    reference = data[42]["Periodic refit"]["per_family_metrics"]
    families = sorted({row["family"] for row in reference if row["family"] != "Benign"})
    family_n = {row["family"]: row["fn"] + row["tp"]
                for row in reference if row["family"] != "Benign"}
    lines = [
        "# NF-ToN-IoT-v3 attack-family outcomes — two seeds", "",
        "Attack recall is shown by the dataset's `Attack` family label. Percentages are the mean of seeds 42 and 43, with both seed values in parentheses. Both seeds replay the same stream, so these ranges measure model-seed sensitivity only.", "",
        "| Attack family | Attack flows | Frozen, fixed threshold | Threshold only | Periodic refit | BRIDGE | VBC-SG | FPR + recall gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in families:
        cells = [family, f"{family_n[family]:,}"]
        for policy, (_, method) in POLICIES.items():
            recalls = []
            for seed in (42, 43):
                rows = data[seed][policy]["per_family_metrics"]
                row = next(r for r in rows if r["family"] == family and r["method"] == method)
                recalls.append(100 * row["attack_recall"])
            cells.append(f"{sum(recalls)/2:.1f}% ({recalls[0]:.1f}, {recalls[1]:.1f})")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "", "## Benign false-positive rates", "",
        "The dataset's `Attack` column assigns all benign rows to the single `Benign` group. It does not expose benign capture/host subgroups in these replay artifacts, so a benign subgroup-FPR analysis is not possible from this table. The available day-level benign FPR remains the tail-risk measure.", "",
        "| Policy | Seed 42 benign FPR | Seed 43 benign FPR |",
        "|---|---:|---:|",
    ]
    for policy, (_, method) in POLICIES.items():
        vals = []
        for seed in (42, 43):
            rows = data[seed][policy]["per_family_metrics"]
            row = next(r for r in rows if r["family"] == "Benign" and r["method"] == method)
            vals.append(100 * row["false_positive_rate"])
        lines.append(f"| {policy} | {vals[0]:.4f}% | {vals[1]:.4f}% |")

    lines += [
        "", "## Interpretation", "",
        "The overall recall/FPR trade-off hides strong attack-family asymmetry. The FPR-gated model is effective on Backdoor and DDoS in this stream, but recall is very low on DoS, injection, MITM, password, and ransomware families; scanning and XSS lie between. The periodic and BRIDGE policies detect more password/XSS flows, while threshold-only recalibration remains close to the frozen model for most families. These family outcomes are descriptive and depend on the campaign ordering and attack-family mix; they do not establish general attack-family performance.", "",
        "The benign label is a single aggregated category, and the stream contains six day groups. To test whether the reported tail is driven by specific benign populations, an independent source with capture/host provenance or safely retained benign identifiers is needed. No benign subgroup claim can be made from the current artifacts.", "",
        "Rebuild with `research_refs/summarize_nf_ton_family_outcomes.py`. Inputs: the two gate JSONs and two rolling-threshold JSONs."
    ]
    out = RESULTS / "NF_TON_PER_FAMILY_RECALL_COMPARISON.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
