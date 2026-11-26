"""Verify the replay cache and measure support for fixed chronological partitions."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cse_corrected/replay_core6")
    args = parser.parse_args()
    cache = args.cache
    manifest = json.loads((cache / "manifest.json").read_text())
    with np.load(cache / "stream.npz") as archive:
        times = archive["completion_ns"]
        labels = archive["labels"]
        offsets = archive["chunk_offsets"]
        source_rows = archive["source_row"]
    rows = manifest["rows"]
    assert len(times) == len(labels) == len(source_rows) == rows
    assert np.array_equal(np.sort(source_rows), np.arange(rows))
    assert np.all(times[1:] >= times[:-1])
    tied = times[1:] == times[:-1]
    assert np.all(source_rows[1:][tied] > source_rows[:-1][tied])
    assert offsets[0] == 0 and offsets[-1] == rows and np.all(np.diff(offsets) > 0)
    assert np.all(times[offsets[1:-1]-1] < times[offsets[1:-1]])
    assert np.isin(labels, [0, 1]).all() and int(labels.sum()) == manifest["attacks"]
    attacks = np.add.reduceat(labels.astype(np.int64), offsets[:-1])
    benign = np.diff(offsets) - attacks
    first, second = len(attacks) // 3, 2 * len(attacks) // 3
    partitions = {}
    for name, part in [("training", slice(0, first)), ("calibration", slice(first, second)),
                       ("evaluation", slice(second, None))]:
        partitions[name] = {"attacks": int(attacks[part].sum()), "benign": int(benign[part].sum()),
                            "mixed_chunks": int(((attacks[part] > 0) & (benign[part] > 0)).sum())}
    evaluation_start = second
    windows = []
    for end in range(evaluation_start + 10, len(attacks) - 9, 10):
        slices = {"fit": slice(end - 10, end - 4), "calibration": slice(end - 4, end - 2),
                  "validation": slice(end - 2, end), "followup": slice(end, end + 10)}
        support = {name: {"attacks": int(attacks[window].sum()), "benign": int(benign[window].sum())}
                   for name, window in slices.items()}
        eligible = all(support[name]["attacks"] >= 100 for name in ("fit", "validation", "followup"))
        eligible &= all(support[name]["benign"] >= 3000 for name in slices)
        windows.append({"deploy_chunk": end, "support": support, "eligible": bool(eligible)})
    report = {"verified_rows": rows, "verified_chunks": len(attacks), "equal_completion_pairs": int(tied.sum()),
              "attack_bearing_chunks": int((attacks > 0).sum()),
              "fixed_chunk_count_thirds": partitions,
              "eligible_update_windows": sum(window["eligible"] for window in windows),
              "update_windows": windows,
              "scope": "Support diagnostic only, no model training or threshold tuning."}
    name = "replay_cache_audit.json" if cache.name == "replay_core6" else f"{cache.name}_audit.json"
    dest = ROOT / "results/corrected_cse_gate" / name
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
