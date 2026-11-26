"""Seeded feasibility-first MOPSO with an explicit nondominated archive.

This optimizer is model-agnostic: ``evaluate(config)`` must return a record
containing ``metrics`` and ``feasibility`` mappings. It does not inspect test
data; callers should evaluate candidates on shared development folds only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from .evaluation import pareto_front


@dataclass(frozen=True)
class Dimension:
    name: str
    lower: float
    upper: float
    kind: str = "float"  # float, int, or log_float

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ValueError(f"Invalid bounds for {self.name}")
        if self.kind not in {"float", "int", "log_float"}:
            raise ValueError(f"Unsupported dimension kind: {self.kind}")
        if self.kind == "log_float" and self.lower <= 0:
            raise ValueError("log_float bounds must be positive")

    def decode(self, position: float) -> float | int:
        position = float(np.clip(position, 0.0, 1.0))
        if self.kind == "log_float":
            value = np.exp(
                np.log(self.lower)
                + position * (np.log(self.upper) - np.log(self.lower))
            )
            return float(value)
        value = self.lower + position * (self.upper - self.lower)
        if self.kind == "int":
            return int(np.clip(np.rint(value), self.lower, self.upper))
        return float(value)


def _objective_values(
    record: dict[str, Any],
    feasibility_first: bool = True,
) -> tuple[float, ...]:
    metrics = record["metrics"]
    quality_costs = (
        float(metrics["weighted_f1"]),
        float(metrics["macro_f1"]),
        -float(metrics["top_label_ece"]),
        -float(metrics["p95_latency_ms_per_row"]),
        -float(metrics["serialized_size_mb"]),
        -float(metrics.get("mean_fit_seconds", 0.0)),
    )
    if not feasibility_first or record["feasibility"]["feasible"]:
        return quality_costs
    limits = record["feasibility"]["constraints"]
    violations = (
        max(0.0, float(limits["min_weighted_f1"]) - float(metrics["weighted_f1"]))
        / max(abs(float(limits["min_weighted_f1"])), 1e-12),
        max(
            0.0,
            float(metrics["p95_latency_ms_per_row"])
            - float(limits["max_p95_latency_ms_per_row"]),
        )
        / max(abs(float(limits["max_p95_latency_ms_per_row"])), 1e-12),
        max(
            0.0,
            float(metrics["serialized_size_mb"])
            - float(limits["max_serialized_size_mb"]),
        )
        / max(abs(float(limits["max_serialized_size_mb"])), 1e-12),
    )
    return tuple(-value for value in violations) + quality_costs


def _dominates(
    left: dict[str, Any],
    right: dict[str, Any],
    feasibility_first: bool = True,
) -> bool:
    lf = bool(left["feasibility"]["feasible"])
    rf = bool(right["feasibility"]["feasible"])
    if feasibility_first and lf != rf:
        return lf
    lv = _objective_values(left, feasibility_first)
    rv = _objective_values(right, feasibility_first)
    return all(a >= b for a, b in zip(lv, rv)) and any(a > b for a, b in zip(lv, rv))


def _crowding_distance(
    records: list[dict[str, Any]],
    feasibility_first: bool = True,
) -> list[float]:
    if len(records) <= 2:
        return [float("inf")] * len(records)
    values = np.asarray(
        [_objective_values(record, feasibility_first) for record in records],
        dtype=float,
    )
    distances = np.zeros(len(records), dtype=float)
    for objective in range(values.shape[1]):
        order = np.argsort(values[:, objective], kind="stable")
        distances[order[0]] = distances[order[-1]] = float("inf")
        span = values[order[-1], objective] - values[order[0], objective]
        if span <= 0:
            continue
        for index in range(1, len(order) - 1):
            if np.isfinite(distances[order[index]]):
                distances[order[index]] += (
                    values[order[index + 1], objective]
                    - values[order[index - 1], objective]
                ) / span
    return distances.tolist()


def _trim_archive(
    records: list[dict[str, Any]],
    maximum: int,
    feasibility_first: bool = True,
) -> list[dict[str, Any]]:
    unique = {}
    for record in records:
        key = json.dumps(record["config"], sort_keys=True, separators=(",", ":"))
        unique.setdefault(key, record)
    candidates = list(unique.values())
    if feasibility_first:
        archive = pareto_front(candidates)
    else:
        archive = []
        for candidate in candidates:
            if not any(
                _dominates(other, candidate, False)
                for other in candidates
                if other is not candidate
            ):
                archive.append(candidate)
    while len(archive) > maximum:
        distances = _crowding_distance(archive, feasibility_first)
        remove_at = int(np.argmin(np.asarray(distances)))
        archive.pop(remove_at)
    return archive


def run_pareto_mopso(
    evaluate: Callable[[dict[str, Any]], dict[str, Any]],
    dimensions: list[Dimension],
    *,
    fixed_config: dict[str, Any] | None = None,
    particles: int = 8,
    iterations: int = 5,
    archive_size: int = 32,
    seed: int = 42,
    output_path: str | Path | None = None,
    feasibility_first: bool = True,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Search a bounded configuration space and log every unique evaluation.

    In feasibility-first mode, feasible candidates dominate infeasible ones.
    Otherwise feasibility is measured but ignored during the search. In either
    mode, the archive maximizes weighted/macro F1 and minimizes ECE, p95
    latency, model size, and fit time. With feasibility-first mode and no
    feasible candidate, normalized constraint violations join the objectives.
    """
    if not dimensions or particles < 2 or iterations < 1 or archive_size < 1:
        raise ValueError(
            "Need dimensions, at least two particles, one iteration, and a positive archive size"
        )
    fixed_config = dict(fixed_config or {})
    rng = np.random.default_rng(seed)
    dimension_count = len(dimensions)
    positions = rng.random((particles, dimension_count))
    velocities = rng.uniform(-0.1, 0.1, size=(particles, dimension_count))
    personal_positions: list[np.ndarray | None] = [None] * particles
    personal_records: list[dict[str, Any] | None] = [None] * particles
    evaluated: dict[str, dict[str, Any]] = {}
    archive: list[dict[str, Any]] = []
    iteration_log = []
    start_iteration = 0
    inertia_start, inertia_end = 0.9, 0.4
    cognitive, social = 1.5, 1.5

    def decode(position: np.ndarray) -> dict[str, Any]:
        config = dict(fixed_config)
        config.update(
            {
                dim.name: dim.decode(position[index])
                for index, dim in enumerate(dimensions)
            }
        )
        return config

    if resume_from is not None:
        checkpoint = json.loads(Path(resume_from).read_text(encoding="utf-8"))
        expected_dimensions = [asdict(dim) for dim in dimensions]
        checks = {
            "seed": seed,
            "feasibility_first": feasibility_first,
            "dimensions": expected_dimensions,
            "fixed_config": fixed_config,
            "particles": particles,
            "iterations": iterations,
        }
        for field, expected in checks.items():
            if checkpoint.get(field) != expected:
                raise ValueError(
                    f"MOPSO checkpoint {field} does not match this run: "
                    f"{checkpoint.get(field)!r} != {expected!r}"
                )
        evaluated = {
            json.dumps(row["config"], sort_keys=True, separators=(",", ":")): row
            for row in checkpoint.get("candidate_records", [])
            if "config" in row
        }
        if not evaluated:
            raise ValueError("MOPSO checkpoint has no cached candidate evaluations")
        if checkpoint.get("checkpoint_version") == 1:
            positions = np.asarray(checkpoint["positions"], dtype=float)
            velocities = np.asarray(checkpoint["velocities"], dtype=float)
            personal_positions = [
                None if row is None else np.asarray(row, dtype=float)
                for row in checkpoint["personal_positions"]
            ]
            personal_records = checkpoint["personal_records"]
            archive = checkpoint["pareto_archive"]
            iteration_log = checkpoint["iteration_log"]
            rng.bit_generator.state = checkpoint["rng_state"]
            start_iteration = len(iteration_log)
        elif len(checkpoint.get("iteration_log", [])) != 1:
            raise ValueError(
                "Legacy MOPSO checkpoints can only be resumed from iteration one"
            )

    def evaluate_position(position: np.ndarray) -> dict[str, Any] | None:
        config = decode(position)
        key = json.dumps(config, sort_keys=True, separators=(",", ":"))
        if key not in evaluated:
            started = perf_counter()
            try:
                result = evaluate(config)
                if "metrics" not in result or "feasibility" not in result:
                    raise ValueError(
                        "evaluate(config) must return metrics and feasibility"
                    )
                evaluated[key] = {
                    "config": config,
                    "metrics": result["metrics"],
                    "feasibility": result["feasibility"],
                    "evaluation_seconds": float(perf_counter() - started),
                    "evaluation_metadata": result.get("metadata", {}),
                }
            except Exception as exc:
                evaluated[key] = {
                    "config": config,
                    "error": f"{type(exc).__name__}: {exc}",
                    "evaluation_seconds": float(perf_counter() - started),
                }
        record = evaluated[key]
        return record if "metrics" in record else None

    for iteration in range(start_iteration, iterations):
        iteration_start = perf_counter()
        valid_this_iteration = 0
        for particle_index in range(particles):
            record = evaluate_position(positions[particle_index])
            if record is None:
                continue
            valid_this_iteration += 1
            archive = _trim_archive(archive + [record], archive_size, feasibility_first)
            incumbent = personal_records[particle_index]
            if incumbent is None or _dominates(record, incumbent, feasibility_first):
                personal_records[particle_index] = record
                personal_positions[particle_index] = positions[particle_index].copy()
            elif (
                not _dominates(incumbent, record, feasibility_first)
                and rng.random() < 0.5
            ):
                personal_records[particle_index] = record
                personal_positions[particle_index] = positions[particle_index].copy()

        if not archive:
            raise RuntimeError(
                "Every MOPSO candidate evaluation failed; see candidate error log"
            )
        iteration_log.append(
            {
                "iteration": iteration + 1,
                "valid_particle_evaluations": valid_this_iteration,
                "unique_candidates_so_far": len(evaluated),
                "archive_size": len(archive),
                "feasible_archive_size": sum(
                    bool(item["feasibility"]["feasible"]) for item in archive
                ),
                "wall_seconds": float(perf_counter() - iteration_start),
            }
        )

        if iteration + 1 < iterations:
            crowding = np.asarray(
                _crowding_distance(archive, feasibility_first), dtype=float
            )
            finite = np.isfinite(crowding)
            weights = np.ones(len(archive), dtype=float)
            if np.any(finite) and float(np.max(crowding[finite])) > 0:
                weights[finite] += crowding[finite]
                weights[~finite] = float(np.max(weights[finite])) * 2
            weights /= weights.sum()
            for particle_index in range(particles):
                leader = archive[int(rng.choice(len(archive), p=weights))]
                leader_position = np.asarray(
                    [
                        (
                            np.clip(
                                (float(leader["config"][dim.name]) - dim.lower)
                                / (dim.upper - dim.lower),
                                0.0,
                                1.0,
                            )
                            if dim.kind != "log_float"
                            else np.clip(
                                (
                                    np.log(float(leader["config"][dim.name]))
                                    - np.log(dim.lower)
                                )
                                / (np.log(dim.upper) - np.log(dim.lower)),
                                0.0,
                                1.0,
                            )
                        )
                        for dim in dimensions
                    ]
                )
                pbest = personal_positions[particle_index]
                if pbest is None:
                    pbest = positions[particle_index]
                inertia = inertia_start - (inertia_start - inertia_end) * (
                    iteration / max(1, iterations - 1)
                )
                velocities[particle_index] = (
                    inertia * velocities[particle_index]
                    + cognitive
                    * rng.random(dimension_count)
                    * (pbest - positions[particle_index])
                    + social
                    * rng.random(dimension_count)
                    * (leader_position - positions[particle_index])
                )
                velocities[particle_index] = np.clip(
                    velocities[particle_index], -0.2, 0.2
                )
                positions[particle_index] = np.clip(
                    positions[particle_index] + velocities[particle_index], 0.0, 1.0
                )

        if output_path is not None:
            save_path = Path(output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            partial = save_path.with_suffix(save_path.suffix + ".part")
            partial.write_text(
                json.dumps(
                    {
                        "algorithm": (
                            "feasibility-first MOPSO with external nondominated archive"
                            if feasibility_first
                            else "constraint-blind MOPSO with external nondominated archive"
                        ),
                        "seed": seed,
                        "feasibility_first": feasibility_first,
                        "dimensions": [asdict(dim) for dim in dimensions],
                        "fixed_config": fixed_config,
                        "particles": particles,
                        "iterations": iterations,
                        "iteration_log": iteration_log,
                        "candidate_records": list(evaluated.values()),
                        "pareto_archive": archive,
                        "positions": positions.tolist(),
                        "velocities": velocities.tolist(),
                        "checkpoint_version": 1,
                        "personal_positions": [
                            None if row is None else row.tolist()
                            for row in personal_positions
                        ],
                        "personal_records": personal_records,
                        "rng_state": rng.bit_generator.state,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            partial.replace(save_path)

    return {
        "algorithm": (
            "feasibility-first MOPSO with external nondominated archive"
            if feasibility_first
            else "constraint-blind MOPSO with external nondominated archive"
        ),
        "seed": seed,
        "feasibility_first": feasibility_first,
        "dimensions": [asdict(dim) for dim in dimensions],
        "fixed_config": fixed_config,
        "particles": particles,
        "iterations": iterations,
        "iteration_log": iteration_log,
        "candidate_records": list(evaluated.values()),
        "pareto_archive": archive,
    }
