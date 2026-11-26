"""Normalize cached CSE replay timestamps to nanoseconds without changing X/y."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "cse_replay_chunk_cache_v1"
TARGET = ROOT / "data" / "cse_replay_chunk_cache_v2"
TEMP = ROOT / "data" / "cse_replay_chunk_cache_v2.building"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_sha256(value: dict) -> str:
    payload = {k: v for k, v in value.items() if k != "manifest_sha256"}
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if TARGET.exists() or TEMP.exists():
        raise FileExistsError(f"Refusing to overwrite {TARGET} or {TEMP}")
    manifest_path = SOURCE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_version") != 1:
        raise ValueError("Expected the version-1 cache with mixed ns/us timestamp integers")
    TEMP.mkdir(parents=True)

    previous_max: dict[str, int] = {}
    normalized = 0
    for item in manifest["chunks"]:
        source_path = SOURCE / item["file"]
        if file_sha256(source_path) != item["sha256"]:
            raise ValueError(f"Source chunk checksum mismatch: {source_path}")
        with np.load(source_path, allow_pickle=False) as arrays:
            x = arrays["x"]
            y = arrays["y"]
            raw = arrays["timestamps_ns"].astype(np.int64, copy=False)
            if len(raw) != item["rows"] or len(x) != len(y) or len(y) != len(raw):
                raise ValueError(f"Array length mismatch in {source_path}")
            # DuckDB-backed 20-February output uses datetime64[us]; pandas CSV
            # captures use datetime64[ns]. Infer and validate each chunk by date.
            unit = "us" if abs(int(raw.min())) < 10**17 else "ns"
            dates = pd.to_datetime(raw, unit=unit, errors="raise")
            expected_date = item["capture_date"]
            if not (dates.date == pd.Timestamp(expected_date).date()).all():
                raise ValueError(f"Timestamp date mismatch for {item['file']} using {unit}")
            ns = dates.astype("datetime64[ns]").astype(np.int64).to_numpy()
            if ns.min() < previous_max.get(item["capture_file"], np.iinfo(np.int64).min):
                raise ValueError(f"Chunks are not chronologically sorted: {item['file']}")
            previous_max[item["capture_file"]] = int(ns.max())
            if unit != "ns":
                normalized += 1
            out_path = TEMP / item["file"]
            np.savez(out_path, x=x, y=y, timestamps_ns=ns)
        item["sha256"] = file_sha256(out_path)
        item["timestamp_min_ns"] = int(ns.min())
        item["timestamp_max_ns"] = int(ns.max())

    manifest["cache_version"] = 2
    manifest["purpose"] = "immutable fixed chunks with normalized nanosecond timestamps"
    manifest["timestamp_storage_unit"] = "ns"
    manifest["migrated_from"] = "cse_replay_chunk_cache_v1; X and y arrays retained unchanged"
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    (TEMP / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    TEMP.rename(TARGET)
    print(json.dumps({
        "source_cache": str(SOURCE.relative_to(ROOT)),
        "target_cache": str(TARGET.relative_to(ROOT)),
        "chunks": len(manifest["chunks"]),
        "chunks_unit_normalized": normalized,
        "rows": sum(int(x["rows"]) for x in manifest["chunks"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "timestamp_storage_unit": "ns",
        "chronology_validated_per_capture": True,
    }, indent=2))


if __name__ == "__main__":
    main()
