"""Summarize selected metrics from the AISec'24 released RF/primary/SCS artifact.

This reads released per-trial results; it does not retrain models or claim a
reproduction of the paper's full pipeline.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "research_refs/aisec24/code/results/rf_primary_scs.txt"
OUTPUT = ROOT / "results/aisec24_rf_primary_scs_artifact_summary.json"


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n_trials": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
    }


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    selected = ["fb_future", "d_fb_future", "ens_future", "d_ens_future"]
    output: dict[str, object] = {
        "source_artifact": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "interpretation": {
            "benign": "accuracy on benign-only examples; interpreted as specificity",
            "malicious_uniform": "mean of malware-specific accuracies; interpreted as macro attack recall",
            "d_prefix": "the paper's defensive-model variant; not an adaptive IDS policy",
            "limitation": "summary of released 50-trial result arrays, not an independent rerun",
        },
        "metrics": {},
    }
    metrics = output["metrics"]
    assert isinstance(metrics, dict)
    for cohort in ("benign", "malicious_uniform"):
        metrics[cohort] = {
            name: summarize(data[cohort][name]) for name in selected
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for cohort, rows in metrics.items():
        print(cohort)
        for name, values in rows.items():
            print(f"  {name}: mean={values['mean']:.6f}, sd={values['sample_sd']:.6f}, n={values['n_trials']}")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
