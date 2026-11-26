"""Create compact, reproducible tables from the locked NF-v3 replay result."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "fpr_0p1_percent"


def pct(value: float) -> float:
    return round(100.0 * value, 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    seed = parser.parse_args().seed
    source = ROOT / "results" / f"nf_ton_iot_v3_fixed_model_replay_seed{seed}.json"
    out_json = ROOT / "results" / f"nf_ton_iot_v3_fixed_model_replay_seed{seed}_analysis.json"
    out_md = ROOT / "results" / f"nf_ton_iot_v3_fixed_model_replay_seed{seed}_analysis.md"
    data = json.loads(source.read_text(encoding="utf-8"))
    days = [r for r in data["per_day_metrics"] if r["threshold"] == PRIMARY]
    families = [r for r in data["per_family_metrics"] if r["threshold"] == PRIMARY]
    pooled = {
        arm: data["pooled_metrics"][f"{arm}:{PRIMARY}"]
        for arm in ("frozen", "periodic_refit")
    }
    update_times = [float(r["fit_seconds"]) for r in data["updates"]]
    ordered_update = sorted(update_times)
    latency = {}
    for arm, values in data["prediction_latency_ms_per_row_by_chunk"].items():
        vals = sorted(float(v) for v in values)
        latency[arm] = {
            "chunks": len(vals),
            "median_ms_per_row": statistics.median(vals),
            "p95_ms_per_row": vals[int(0.95 * (len(vals) - 1))],
            "max_ms_per_row": vals[-1],
        }

    day_rows = []
    for r in days:
        day_rows.append({
            "date_utc": r["date_utc"], "arm": r["arm"],
            "benign_rows": r["tn"] + r["fp"],
            "attack_rows": r["fn"] + r["tp"],
            "fp": r["fp"], "fn": r["fn"], "tp": r["tp"],
            "fpr_percent": pct(r["false_positive_rate"]),
            "attack_recall_percent": pct(r["attack_recall"]),
            "macro_f1": r["macro_f1"],
        })
    family_rows = []
    for r in families:
        name = r["attack_or_benign_family"]
        row = {"family": name, "arm": r["arm"], "support": r["support"],
               "tp": r["tp"], "fn": r["fn"]}
        if name.lower() == "benign":
            row.update({"fp": r["fp"], "fpr_percent": pct(r["false_positive_rate"])})
        else:
            row["attack_recall_percent"] = pct(r["attack_recall"])
        family_rows.append(row)

    diagnostics = {}
    for arm in ("frozen", "periodic_refit"):
        arm_days = [r for r in day_rows if r["arm"] == arm]
        arm_families = [r for r in family_rows if r["arm"] == arm and r["family"].lower() != "benign"]
        fprs = [r["fpr_percent"] for r in arm_days]
        recalls = [r["attack_recall_percent"] for r in arm_days]
        diagnostics[arm] = {
            "active_days": len(arm_days),
            "daily_fpr_percent_min_median_max": [min(fprs), statistics.median(fprs), max(fprs)],
            "daily_recall_percent_min_median_max": [min(recalls), statistics.median(recalls), max(recalls)],
            "attack_families": len(arm_families),
            "family_recall_percent_min_median_max": [
                min(r["attack_recall_percent"] for r in arm_families),
                statistics.median(r["attack_recall_percent"] for r in arm_families),
                max(r["attack_recall_percent"] for r in arm_families),
            ],
            "lowest_recall_families": sorted(
                [{"family": r["family"], "recall_percent": r["attack_recall_percent"], "support": r["support"]}
                 for r in arm_families], key=lambda r: r["recall_percent"]
            )[:5],
            "highest_recall_families": sorted(
                [{"family": r["family"], "recall_percent": r["attack_recall_percent"], "support": r["support"]}
                 for r in arm_families], key=lambda r: r["recall_percent"], reverse=True
            )[:5],
        }

    result = {
        "source_result": str(source.relative_to(ROOT)),
        "protocol": "NF_TON_IOT_V3_REPLAY_PROTOCOL_2026-11-05.md",
        "threshold": PRIMARY,
        "development_only": True,
        "rows": data["rows"], "chunks": data["chunk_count"],
        "pooled": pooled,
        "daily_diagnostics": diagnostics,
        "daily_rows": day_rows,
        "attack_family_rows": family_rows,
        "updates": {
            "count": len(update_times), "total_fit_seconds": sum(update_times),
            "median_fit_seconds": statistics.median(update_times),
            "p95_fit_seconds": ordered_update[int(0.95 * (len(ordered_update) - 1))],
            "max_fit_seconds": max(update_times),
        },
        "inference_latency": latency,
        "model": data["model"],
        "calibration": data["calibration"],
        "wall_seconds": data["wall_seconds_total_including_initial_sample_fit_and_replay"],
        "interpretation": [
            "Periodic refitting improves pooled detection metrics on this inspected development schedule.",
            "The fixed initial calibration threshold becomes unsafe after refitting: pooled and daily FPR rise substantially.",
            "Campaign-family recall is heterogeneous; pooled gains do not establish generalization or novelty.",
        ],
    }
    lines = [
        f"# NF-ToN-IoT-v3 fixed-model replay analysis (seed {seed})", "",
        "This report is derived from the locked replay JSON. It is development-only: the campaign chronology was inspected before the fit/calibration/replay boundary was set. The dataset belongs to the ToN-IoT testbed family and does not provide independent-network confirmation.", "",
        f"Primary operating point: threshold calibrated on benign calibration rows for a 0.1% empirical FPR cap. Replay: {data['rows']:,} rows / {data['chunk_count']} tied-time-preserving chunks.", "",
        "## Pooled outcomes", "",
        "| Arm | Benign FPR | Attack recall | Macro F1 | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, r in pooled.items():
        lines.append(f"| {arm} | {pct(r['false_positive_rate']):.4f}% | {pct(r['attack_recall']):.2f}% | {r['macro_f1']:.4f} | {r['fp']:,} | {r['fn']:,} | {r['tp']:,} |")
    lines.extend(["", "## Day-level risk range", "", "| Arm | Active days | Daily FPR min / median / max | Daily attack recall min / median / max |", "|---|---:|---:|---:|"])
    for arm, d in diagnostics.items():
        f = " / ".join(f"{x:.4f}%" for x in d["daily_fpr_percent_min_median_max"])
        a = " / ".join(f"{x:.2f}%" for x in d["daily_recall_percent_min_median_max"])
        lines.append(f"| {arm} | {d['active_days']} | {f} | {a} |")
    lines.extend(["", "## Attack-family recall", "", "| Family | Support | Frozen recall | Periodic-refit recall |", "|---|---:|---:|---:|"])
    by_family = {}
    for r in family_rows:
        if r["family"].lower() != "benign":
            by_family.setdefault(r["family"], {})[r["arm"]] = r
    for name, arms in sorted(by_family.items()):
        fr = arms.get("frozen", {}).get("attack_recall_percent", 0.0)
        pr = arms.get("periodic_refit", {}).get("attack_recall_percent", 0.0)
        support = arms.get("frozen", arms.get("periodic_refit", {})).get("support", 0)
        lines.append(f"| {name} | {support:,} | {fr:.2f}% | {pr:.2f}% |")
    u = result["updates"]
    lines.extend(["", "## Compute", "", f"Periodic refit count: {u['count']}; total refit CPU time: {u['total_fit_seconds']:.2f}s; median: {u['median_fit_seconds']:.2f}s; p95: {u['p95_fit_seconds']:.2f}s; max: {u['max_fit_seconds']:.2f}s.", "", "Serialized initial/final model sizes: {0:.1f} KB / {1:.1f} KB.".format(data["model"]["model_file_sizes"]["initial_frozen_bytes"] / 1024, data["model"]["model_file_sizes"]["final_periodic_refit_bytes"] / 1024), "", "## Interpretation", "", "Refitting substantially raises pooled recall but also increases false alarms beyond the calibration target; day and family tables show that pooled results conceal severe unevenness. Threshold recalibration is an established IDS response and is treated here as a comparator, not a novelty claim. Any new comparison requires its own locked protocol.", "", f"Source result: `results/nf_ton_iot_v3_fixed_model_replay_seed{seed}.json`. Protocol: `NF_TON_IOT_V3_REPLAY_PROTOCOL_2026-11-05.md`.", ""])
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(out_md.relative_to(ROOT))
    print(out_json.relative_to(ROOT))


if __name__ == "__main__":
    main()
