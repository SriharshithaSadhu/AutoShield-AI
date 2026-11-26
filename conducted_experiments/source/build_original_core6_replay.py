"""Project the original CSE replay cache onto the six-field comparison schema."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data/cse_replay_chunk_cache_v2"
DESTINATION = ROOT / "data/cse_original_replay_core6"
FEATURES = ("Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
            "TotLen Fwd Pkts", "TotLen Bwd Pkts")


def main():
    manifest = json.loads((CACHE / "manifest.json").read_text())
    columns = [manifest["features"].index(name) for name in FEATURES]
    rows = sum(chunk["rows"] for chunk in manifest["chunks"])
    features = np.empty((rows, len(columns)), dtype=np.float64)
    labels = np.empty(rows, dtype=np.uint8)
    completion = np.empty(rows, dtype=np.int64)
    capture_days = np.empty(rows, dtype=np.int32)
    offset = 0
    for index, chunk in enumerate(manifest["chunks"]):
        with np.load(CACHE / chunk["file"]) as record:
            values = record["x"][:, columns]
            duration = values[:, 1]
            assert np.isfinite(duration).all() and (duration >= 0).all()
            assert np.equal(duration, duration.astype(np.int64)).all()
            stop = offset + len(values)
            features[offset:stop] = values
            labels[offset:stop] = record["y"]
            completion[offset:stop] = record["timestamps_ns"] + duration.astype(np.int64) * 1000
            capture_days[offset:stop] = int(chunk["capture_date"].replace("-", ""))
            offset = stop
        if index % 100 == 0:
            print(f"Projected {offset:,} rows", flush=True)
    assert offset == rows and labels.sum() == sum(chunk["attack"] for chunk in manifest["chunks"])
    order = np.argsort(completion, kind="stable")
    times = completion[order]
    offsets = np.unique(np.r_[0, np.searchsorted(times, times[19999::20000], side="right"), rows])
    assert (times[offsets[1:-1] - 1] < times[offsets[1:-1]]).all()
    DESTINATION.mkdir(exist_ok=True)
    np.savez(DESTINATION / "stream.npz", features=features[order], labels=labels[order],
             completion_ns=times, source_row=order, chunk_offsets=offsets, capture_day=capture_days[order])
    report = {"rows": rows, "attacks": int(labels.sum()), "features": FEATURES, "chunks": len(offsets)-1,
              "source_manifest_sha256": manifest["manifest_sha256"],
              "scope": "Original CSE labels and exporter. Completion order is derived from recorded timestamps."}
    (DESTINATION / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
