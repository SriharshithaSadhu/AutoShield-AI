"""Check algebraic CICFlowMeter invariants in the IoTID20/CSE transfer inputs.

This tests a few identities (rates = counts / duration) rather than assuming
that matching feature names proves full semantic equivalence. It is a
read-only, deterministic audit and does not fit or alter a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "iotid20_binary_common77_dev.csv"
CACHE = ROOT / "data" / "cse_replay_chunk_cache_v2"
OUT = ROOT / "results" / "iotid20_cse_feature_semantics_audit_20261102.json"
SEED = 20261102
FRACTION_PER_CHUNK = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(frame: pd.DataFrame) -> dict[str, object]:
    duration = frame["Flow Duration"].to_numpy(dtype=np.float64)
    packets = frame["Tot Fwd Pkts"].to_numpy(dtype=np.float64) + frame["Tot Bwd Pkts"].to_numpy(dtype=np.float64)
    directional = {
        "Flow Pkts/s": packets,
        "Fwd Pkts/s": frame["Tot Fwd Pkts"].to_numpy(dtype=np.float64),
        "Bwd Pkts/s": frame["Tot Bwd Pkts"].to_numpy(dtype=np.float64),
    }
    valid = np.isfinite(duration) & (duration > 0)
    rows: dict[str, object] = {"rows": int(len(frame)), "valid_positive_duration_rows": int(valid.sum())}
    for rate_name, count in directional.items():
        rate = frame[rate_name].to_numpy(dtype=np.float64)
        mask = valid & np.isfinite(rate) & np.isfinite(count)
        predicted_count = rate[mask] * duration[mask] / 1_000_000.0
        abs_error = np.abs(predicted_count - count[mask])
        tolerance = np.maximum(1e-6, np.abs(count[mask]) * 1e-6)
        rows[rate_name] = {
            "n": int(mask.sum()),
            "duration_microsecond_identity_fraction_within_1e-6_relative": float(np.mean(abs_error <= tolerance)),
            "absolute_error_quantiles": [float(v) for v in np.quantile(abs_error, [0, 0.5, 0.95, 1])],
            "duration_seconds_identity_fraction_within_1e-6_relative": float(
                np.mean(np.abs(rate[mask] * duration[mask] - count[mask]) <= tolerance)
            ),
        }
    return rows


def main() -> None:
    source = pd.read_csv(SOURCE)
    features = [name for name in source.columns if name != "is_attack"]
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    if manifest["features"] != features:
        raise ValueError("Source feature order differs from the CSE cache manifest")

    rng = np.random.default_rng(SEED)
    cse_parts: list[np.ndarray] = []
    sampled_rows = 0
    for chunk in manifest["chunks"]:
        with np.load(CACHE / chunk["file"], allow_pickle=False) as record:
            n = len(record["y"])
            take = min(n, max(1, int(round(n * FRACTION_PER_CHUNK))))
            indices = np.sort(rng.choice(n, size=take, replace=False))
            cse_parts.append(np.asarray(record["x"][indices], dtype=np.float64))
            sampled_rows += take

    cse = pd.DataFrame(np.concatenate(cse_parts), columns=features)
    result = {
        "status": "complete",
        "audit": "Algebraic packet-rate/duration invariants for IoTID20 and timestamped CSE transfer features",
        "date": "2026-11-02",
        "seed": SEED,
        "target_sample_fraction_per_chunk": FRACTION_PER_CHUNK,
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": sha256(SOURCE),
            "rows": int(len(source)),
            "identity_checks": evaluate(source),
        },
        "target": {
            "cache": str(CACHE.relative_to(ROOT)),
            "manifest_sha256": manifest["manifest_sha256"],
            "chunks": len(manifest["chunks"]),
            "population_rows": int(sum(chunk["rows"] for chunk in manifest["chunks"])),
            "sampled_rows": sampled_rows,
            "identity_checks": evaluate(cse),
        },
        "interpretation_limits": [
            "These identities provide strong evidence that sampled duration and packet-rate fields use a consistent microsecond-to-second rate relationship across the two prepared inputs.",
            "They do not prove full feature-semantic equivalence, exporter/version identity, packet-direction equivalence, timeout settings, labeling validity, or transfer generality.",
            "CSE checks use a fixed-seed 1% sample from each existing replay chunk and inherit that replay cache's preprocessing and inclusion rules.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "source_rows": result["source"]["rows"],
        "target_sampled_rows": sampled_rows,
        "source_microsecond_identity": {
            key: value["duration_microsecond_identity_fraction_within_1e-6_relative"]
            for key, value in result["source"]["identity_checks"].items() if isinstance(value, dict)
        },
        "target_microsecond_identity": {
            key: value["duration_microsecond_identity_fraction_within_1e-6_relative"]
            for key, value in result["target"]["identity_checks"].items() if isinstance(value, dict)
        },
    }, indent=2))


if __name__ == "__main__":
    main()
