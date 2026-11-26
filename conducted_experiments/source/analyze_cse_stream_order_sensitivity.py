"""Paired descriptive analysis of fixed-chunk CSE replay schedules."""
from __future__ import annotations

import json
import hashlib
import statistics
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
POLICIES = ("static", "rccda_paper_estimator_port", "periodic_10pct")
METRICS = (
    "pooled_macro_f1", "pooled_weighted_f1", "pooled_attack_recall",
    "pooled_benign_fpr", "mean_chunk_macro_f1", "mean_chunk_fpr",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def timestamp_iso(value: int) -> str:
    return pd.to_datetime(value, unit="ns").isoformat()


def describe(values: list[float]) -> dict:
    return {
        "n_values": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def capture_metrics(records: list[dict]) -> dict[str, dict[str, float | int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in records:
        agg = counts.setdefault(row["capture_file"], {k: 0 for k in ("tn", "fp", "fn", "tp")})
        for key in agg:
            agg[key] += int(row[key])
    out = {}
    for name, c in counts.items():
        tn, fp, fn, tp = (c[k] for k in ("tn", "fp", "fn", "tp"))
        attack_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        benign_f1 = 2 * tn / max(1, 2 * tn + fp + fn)
        out[name] = {
            **c,
            "rows": tn + fp + fn + tp,
            "attack_recall": tp / max(1, tp + fn),
            "benign_fpr": fp / max(1, fp + tn),
            "macro_f1": (attack_f1 + benign_f1) / 2,
        }
    return out


def main() -> None:
    reference_path = RESULTS / "cse_stream_order_sensitivity_chronological.json"
    reference = load(reference_path)
    manifest = load(ROOT / "data" / "cse_replay_chunk_cache_v2" / "manifest.json")
    saved_manifest_hash = manifest["manifest_sha256"]
    raw_manifest = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    actual_manifest_hash = hashlib.sha256(
        json.dumps(raw_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual_manifest_hash != saved_manifest_hash:
        raise ValueError("v2 chunk manifest self-check failed")
    candidates = sorted(RESULTS.glob("cse_stream_order_sensitivity_within_day_seed*.json"))
    if not candidates:
        raise FileNotFoundError("No within-day permutation result files")
    permuted = [load(path) for path in candidates]
    base_manifest = reference["chunk_manifest_sha256"]
    base_sources = reference["source_file_sha256"]
    base_rows = reference["stream"]["n_rows"]
    base_chunks = reference["stream"]["n_chunks"]
    for item in permuted:
        if item["chunk_manifest_sha256"] != base_manifest:
            raise ValueError("Compared schedules use different immutable chunk manifests")
        if item["source_file_sha256"] != base_sources:
            raise ValueError("Compared schedules use different source files")
        if (item["stream"]["n_rows"], item["stream"]["n_chunks"]) != (base_rows, base_chunks):
            raise ValueError("Compared schedules contain different row/chunk totals")
        if item["checks"]["static_model_confusion"] != reference["checks"]["static_model_confusion"]:
            raise ValueError("Static-model confusion changed across schedules")
        if not item["checks"]["all_labels_released_after_chunk_scoring"]:
            raise ValueError("A schedule violates delayed-label sequencing")

    for item in [reference, *permuted]:
        if item["chunk_manifest_sha256"] != saved_manifest_hash:
            raise ValueError("Result refers to a different cache manifest")
        if item["chunk_cache"].replace("\\", "/") != "data/cse_replay_chunk_cache_v2":
            raise ValueError("Result does not use the normalized v2 cache")
        for arm in item["policies"].values():
            for row in arm["chunk_records"]:
                cache_row = manifest["chunks"][row["chunk_id"]]
                if (row["capture_file"], row["rows"]) != (cache_row["capture_file"], cache_row["rows"]):
                    raise ValueError("Result chunk ID does not match cached chunk identity")
                if (row["start_timestamp"], row["end_timestamp"]) != (
                    timestamp_iso(cache_row["timestamp_min_ns"]),
                    timestamp_iso(cache_row["timestamp_max_ns"]),
                ):
                    raise ValueError("Saved timestamp metadata differs from v2 cache")

    result = {
        "question": "Does within-capture intact-chunk arrival order change adaptive IDS outcomes?",
        "status": "pilot" if len(permuted) < 5 else "five-schedule descriptive analysis",
        "chronological_result": str(reference_path.relative_to(ROOT)),
        "permutation_results": [str(p.relative_to(ROOT)) for p in candidates],
        "n_permutation_schedules": len(permuted),
        "chunk_manifest_sha256": base_manifest,
        "source_file_sha256": base_sources,
        "fixed_rows": base_rows,
        "fixed_chunks": base_chunks,
        "static_control_invariant": True,
        "same_chunk_manifest_and_sources": True,
        "paired_deltas": {},
        "policy_rankings": {},
        "updates": {},
        "capture_level_comparisons": {
            "captures_per_schedule": None,
            "permutation_seeds": [item["order_seed"] for item in permuted],
            "paired_capture_seed_observations_per_policy": None,
            "aggregation_note": "descriptive capture-by-seed deltas; captures and seeds are not independent dataset replicates",
        },
        "capture_level_deltas": {},
        "limitations": [
            "Counterfactual within-day schedules are not natural chronology or deployment estimates.",
            "Seeds are randomized order schedules conditional on one dataset and one initial model, not independent dataset replicates.",
            "Chunk and flow counts are not independent replicates; results are descriptive.",
            "RCCDA is a scheduler port to LightGBM full refits; its published SGD guarantee does not transfer.",
        ],
    }

    for arm in POLICIES:
        deltas: dict[str, list[float]] = {metric: [] for metric in METRICS}
        updates = []
        ranks = []
        base_summary = reference["policies"][arm]["summary"]
        base_capture = capture_metrics(reference["policies"][arm]["chunk_records"])
        if result["capture_level_comparisons"]["captures_per_schedule"] is None:
            result["capture_level_comparisons"]["captures_per_schedule"] = len(base_capture)
            result["capture_level_comparisons"]["paired_capture_seed_observations_per_policy"] = len(base_capture) * len(permuted)
        capture_delta = {metric: [] for metric in ("attack_recall", "benign_fpr", "macro_f1")}
        for item in permuted:
            summary = item["policies"][arm]["summary"]
            for metric in METRICS:
                deltas[metric].append(summary[metric] - base_summary[metric])
            fit_times = [x["fit_seconds_reported_by_refit"] for x in item["policies"][arm]["updates"]]
            updates.append({
                "seed": item["order_seed"],
                "count": len(fit_times),
                "reported_fit_seconds_sum": sum(fit_times),
            })
            current_capture = capture_metrics(item["policies"][arm]["chunk_records"])
            for name, b in base_capture.items():
                for metric in capture_delta:
                    capture_delta[metric].append(current_capture[name][metric] - b[metric])
        result["paired_deltas"][arm] = {metric: describe(values) for metric, values in deltas.items()}
        result["updates"][arm] = {
            "chronological_count": len(reference["policies"][arm]["updates"]),
            "chronological_reported_fit_seconds_sum": sum(
                x["fit_seconds_reported_by_refit"]
                for x in reference["policies"][arm]["updates"]
            ),
            "permutation_schedules": updates,
        }
        result["capture_level_deltas"][arm] = {
            metric: describe(values) for metric, values in capture_delta.items()
        }

    for item in [reference, *permuted]:
        ranked = sorted(
            ((item["policies"][arm]["summary"]["pooled_macro_f1"], arm) for arm in POLICIES),
            reverse=True,
        )
        result["policy_rankings"][item["order"] + (f"_seed{item['order_seed']}" if item["order_seed"] is not None else "")] = [
            arm for _, arm in ranked
        ]

    out_json = RESULTS / "cse_stream_order_sensitivity_paired_analysis.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fixed-chunk stream-order sensitivity: paired results",
        "",
        f"**Status:** {result['status']}; {len(permuted)} within-day permutation schedule(s).",
        "",
        f"Both treatments use {base_rows:,} identical rows and {base_chunks} immutable chunks. "
        f"Manifest SHA-256: `{base_manifest}`. Static confusion is invariant and all labels are "
        "released after scoring each chunk.",
        "",
        "## Pooled paired changes",
        "",
        "Deltas are permutation minus chronological. These are descriptive counterfactual "
        "schedule effects, not estimates of deployment performance.",
        "",
        "| Policy | Pooled macro-F1 Δ | Attack recall Δ | Benign FPR Δ | Mean chunk macro-F1 Δ | Updates (chrono → perm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("rccda_paper_estimator_port", "periodic_10pct"):
        d = result["paired_deltas"][arm]
        upd = result["updates"][arm]
        perm_counts = ", ".join(str(x["count"]) for x in upd["permutation_schedules"])
        lines.append(
            f"| {arm} | {d['pooled_macro_f1']['mean']:+.4f} | "
            f"{d['pooled_attack_recall']['mean']:+.4f} | {d['pooled_benign_fpr']['mean']:+.4f} | "
            f"{d['mean_chunk_macro_f1']['mean']:+.4f} | {upd['chronological_count']} → {perm_counts} |"
        )
    lines += [
        "",
        "## Per-seed paired deltas",
        "",
        "The following values prevent the mean from hiding schedule-to-schedule variation.",
        "",
        "| Policy | Outcome | " + " | ".join(f"Seed {item['order_seed']}" for item in permuted) + " |",
        "|---|---|" + "---:|" * len(permuted),
    ]
    for arm in ("rccda_paper_estimator_port", "periodic_10pct"):
        for metric in ("pooled_macro_f1", "pooled_attack_recall", "pooled_benign_fpr", "mean_chunk_macro_f1"):
            values = result["paired_deltas"][arm][metric]["values"]
            lines.append(
                f"| {arm} | {metric} | " + " | ".join(f"{v:+.4f}" for v in values) + " |"
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "A seed randomizes intact chunk order within each capture while preserving capture order, "
        "chunk contents, chunk labels, and release timing. Seeds are alternative schedules of the "
        "same data, not independent datasets. Do not use flow-level standard errors or claim a "
        "general drift result from this analysis. Refit wall time is hardware/load-sensitive and "
        "should be treated as descriptive; compare update counts and report measured runtime with "
        "the machine/run conditions.",
        "",
        f"Machine-readable output: `{out_json.relative_to(ROOT)}`.",
    ]
    out_md = RESULTS / "cse_stream_order_sensitivity_paired_analysis.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out_json), "markdown": str(out_md),
                      "n_permutation_schedules": len(permuted),
                      "policy_rankings": result["policy_rankings"]}, indent=2))


if __name__ == "__main__":
    main()
