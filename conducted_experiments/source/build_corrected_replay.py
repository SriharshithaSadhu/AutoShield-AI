"""Build a same-exporter development stream using six directly recorded fields."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MEMBER = "Wednesday-28-02-2018.csv"
FEATURES = ["Protocol", "Flow Duration", "Total Fwd Packet", "Total Bwd packets",
            "Total Length of Fwd Packet", "Total Length of Bwd Packet"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", default=MEMBER)
    args = parser.parse_args()
    member = Path(args.member).name
    if member != args.member or not member.endswith(".csv"):
        parser.error("provide a capture CSV basename")
    source = ROOT / "data/cse_corrected" / member
    audit = json.loads((ROOT / "results/corrected_cse_gate" /
                        (member + ".full_audit.json")).read_text())
    assert source.stat().st_size == audit["provenance"]["bytes"]
    rows = audit["counts"]["rows"]
    features = np.empty((rows, len(FEATURES)), dtype=np.int64)
    labels = np.empty(rows, dtype=np.uint8)
    times = np.empty(rows, dtype=np.int64)
    offset = 0
    for frame in pd.read_csv(source, usecols=FEATURES + ["Timestamp", "Label", "Attempted Category"],
                             chunksize=200_000, dtype={name: "int64" for name in FEATURES}):
        stop = offset + len(frame)
        values = frame[FEATURES].to_numpy()
        assert (values >= 0).all() and frame["Label"].notna().all()
        assert frame["Attempted Category"].notna().all()
        features[offset:stop] = values
        labels[offset:stop] = (frame["Label"].str.strip().str.casefold().ne("benign")
                               & frame["Attempted Category"].eq(-1))
        start = pd.to_datetime(frame["Timestamp"], format="mixed")
        assert start.notna().all()
        times[offset:stop] = (start + pd.to_timedelta(frame["Flow Duration"], unit="us")).to_numpy(
            dtype="datetime64[ns]").astype("int64")
        offset = stop
        if offset % 1_000_000 == 0:
            print("Parsed", offset, flush=True)
    assert offset == rows and int(labels.sum()) == audit["counts"]["primary_attack"]
    order = times.argsort(kind="stable")
    sorted_times = times[order]
    assert sorted_times[0] == audit["completion_min_ns"]
    assert sorted_times[-1] == audit["completion_max_ns"]
    assert np.all(sorted_times[1:] >= sorted_times[:-1])
    cuts = np.unique(np.r_[0, np.searchsorted(sorted_times, sorted_times[19999::20000],
                                             side="right"), rows])
    assert np.all(sorted_times[cuts[1:-1]-1] < sorted_times[cuts[1:-1]])
    # Exact equality on the projected integer fields, with both labels retained.
    table = pd.DataFrame(features, columns=FEATURES)
    table["attack"] = labels
    groups = table.groupby(FEATURES, sort=False)["attack"].agg(["size", "sum"])
    conflicts = groups["sum"].gt(0) & groups["sum"].lt(groups["size"])
    result = {
        "source": member, "source_sha256_from_prior_audit": audit["provenance"]["sha256"],
        "features": FEATURES, "rows": rows, "attacks": int(labels.sum()),
        "chunks": len(cuts)-1, "largest_chunk": int(np.diff(cuts).max()),
        "unique_feature_vectors": len(groups),
        "rows_in_repeated_vectors": int(groups.loc[groups["size"].gt(1), "size"].sum()),
        "conflicting_vectors": int(conflicts.sum()),
        "rows_in_conflicting_vectors": int(groups.loc[conflicts, "size"].sum()),
        "scope": "Same-capture development, six-field projection. Completion time is a replay surrogate."
    }
    folder = "replay_core6" if member == MEMBER else f"replay_core6_{Path(member).stem}"
    output = ROOT / "data/cse_corrected" / folder
    output.mkdir(exist_ok=True)
    np.savez(output / "stream.npz", features=features[order], labels=labels[order],
             completion_ns=sorted_times, source_row=order, chunk_offsets=cuts)
    (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
