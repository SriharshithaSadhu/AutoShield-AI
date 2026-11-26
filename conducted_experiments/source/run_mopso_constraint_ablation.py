"""Run constraint-blind MOPSO with the matched seed-42 development protocol.

The existing seed-42 feasibility-first MOPSO records are the constrained
counterparts. Run this only when no other CPU-heavy project experiment is
active, so candidate latency comparisons are not distorted by contention.
"""

from pathlib import Path
import argparse
import json

from deployment_evaluation import DeploymentConstraints
from research_runner import file_sha256, search_lightgbm_mopso


ROOT = Path(__file__).parent
# This profile was part of the completed resource sweep and imposes meaningful
# latency and size limits on the measured CPU candidates.
CONSTRAINTS = DeploymentConstraints(
    min_weighted_f1=0.95,
    max_p95_latency_ms_per_row=0.05,
    max_serialized_size_mb=4.0,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

for dataset, target, name in (
    (ROOT / "data" / "CICIDS2017_sample_km.csv", "Label", "cicids2017"),
    (ROOT / "data" / "IoTID20_sample_0.05.csv", "Label", "iotid20"),
):
    for mode, feasibility_first in (("feasibility_first", True), ("constraint_blind", False)):
        output = ROOT / "results" / (
            f"{name}_lightgbm_mopso_{mode}_dev_seed{args.seed}.json"
        )
        resume_from = None
        if output.exists():
            checkpoint = json.loads(output.read_text(encoding="utf-8"))
            candidate_count = len(checkpoint.get("candidate_records", []))
            completed_iterations = len(checkpoint.get("iteration_log", []))
            is_complete = (
                candidate_count == 24
                and completed_iterations == 4
                and checkpoint.get("search_mode") == mode
                and checkpoint.get("constraints") == CONSTRAINTS.__dict__
                and checkpoint.get("dataset_sha256") == file_sha256(dataset)
                and checkpoint.get("target_column") == target
            )
            if is_complete:
                print(name, mode, "already complete; skipping", flush=True)
                continue
            if candidate_count and (
                checkpoint.get("checkpoint_version") == 1
                or completed_iterations == 1
            ):
                resume_from = output
                print(name, mode, "resuming saved checkpoint at iteration",
                      completed_iterations, "with", candidate_count,
                      "candidates", flush=True)
            elif candidate_count:
                raise ValueError(
                    f"Cannot safely resume {output}: {candidate_count} candidates "
                    f"across {completed_iterations} saved iterations"
                )
        result = search_lightgbm_mopso(
            dataset,
            target_column=target,
            constraints=CONSTRAINTS,
            output_path=output,
            seed=args.seed,
            particles=6,
            iterations=4,
            n_jobs=1,
            feasibility_first=feasibility_first,
            resume_from=resume_from,
        )
        actual_feasible = sum(
            bool(item["feasibility"]["feasible"])
            for item in result["candidate_records"]
        )
        print(
            name,
            mode,
            "unique candidates", len(result["candidate_records"]),
            "feasible candidates", actual_feasible,
            "archive size", len(result["pareto_archive"]),
            "saved", output,
            flush=True,
        )
