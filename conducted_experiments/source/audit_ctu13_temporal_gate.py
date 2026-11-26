"""Audit one official CTU-13 labeled bidirectional flow file for causal replay."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "ctu13" / "scenario1" / "capture20110810.binetflow"
OUTPUT = ROOT / "results" / "ctu13_scenario1_temporal_gate.json"
SOURCE_URL = (
    "https://mcfp.felk.cvut.cz/publicDatasets/"
    "CTU-Malware-Capture-Botnet-42/detailed-bidirectional-flow-labels/"
    "capture20110810.binetflow"
)
CHUNK_ROWS = 500_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    labels: Counter[str] = Counter()
    rows = invalid_times = invalid_durations = negative_durations = 0
    start_min = start_max = end_min = end_max = None
    previous_start_max = None
    start_order_regressions = 0
    confirmed_attack = confirmed_normal = unresolved = 0
    label_time_ranges = {"attack": [None, None], "normal": [None, None]}
    hourly_label_counts: Counter[str] = Counter()
    src_ip_unique: set[str] = set()
    dst_ip_unique: set[str] = set()

    for frame in pd.read_csv(SOURCE, chunksize=CHUNK_ROWS, low_memory=False):
        rows += len(frame)
        label = frame["Label"].fillna("").astype(str).str.strip()
        labels.update(label.value_counts(dropna=False).to_dict())
        normalized = label.str.casefold()
        attack = normalized.str.startswith("flow=from-botnet")
        normal = normalized.str.startswith("flow=from-normal")
        confirmed_attack += int(attack.sum())
        confirmed_normal += int(normal.sum())
        unresolved += int((~(attack | normal)).sum())

        start = pd.to_datetime(frame["StartTime"], errors="coerce", format="mixed")
        duration = pd.to_numeric(frame["Dur"], errors="coerce")
        invalid_times += int(start.isna().sum())
        invalid_durations += int(duration.isna().sum())
        negative_durations += int((duration.dropna() < 0).sum())
        valid = start.notna()
        if valid.any():
            chunk_min, chunk_max = start[valid].min(), start[valid].max()
            start_min = chunk_min if start_min is None else min(start_min, chunk_min)
            start_max = chunk_max if start_max is None else max(start_max, chunk_max)
            if previous_start_max is not None:
                start_order_regressions += int((start[valid] < previous_start_max).sum())
            previous_start_max = chunk_max
            for group, mask in (("attack", attack & valid), ("normal", normal & valid)):
                if mask.any():
                    group_min, group_max = start[mask].min(), start[mask].max()
                    old_min, old_max = label_time_ranges[group]
                    label_time_ranges[group] = [
                        str(group_min if old_min is None else min(pd.Timestamp(old_min), group_min)),
                        str(group_max if old_max is None else max(pd.Timestamp(old_max), group_max)),
                    ]
            timed = pd.DataFrame({
                "hour": start[valid].dt.floor("h"),
                "attack": attack[valid].astype("int8"),
                "normal": normal[valid].astype("int8"),
            })
            for hour, group in timed.groupby("hour", sort=False):
                hourly_label_counts[f"{hour.isoformat()}|attack"] += int(group["attack"].sum())
                hourly_label_counts[f"{hour.isoformat()}|normal"] += int(group["normal"].sum())
            valid_end = valid & duration.notna() & duration.ge(0)
            if valid_end.any():
                ends = start[valid_end] + pd.to_timedelta(duration[valid_end], unit="s")
                chunk_end_min, chunk_end_max = ends.min(), ends.max()
                end_min = chunk_end_min if end_min is None else min(end_min, chunk_end_min)
                end_max = chunk_end_max if end_max is None else max(end_max, chunk_end_max)

        src_ip_unique.update(frame["SrcAddr"].dropna().astype(str).unique().tolist())
        dst_ip_unique.update(frame["DstAddr"].dropna().astype(str).unique().tolist())

    manifest = {
        "dataset": "CTU-13 Scenario 1 / Botnet-42 (Neris)",
        "source_url": SOURCE_URL,
        "retrieved_file": str(SOURCE.relative_to(ROOT)),
        "sha256": sha256(SOURCE),
        "bytes": SOURCE.stat().st_size,
        "rows": rows,
        "columns": [
            "StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir",
            "DstAddr", "Dport", "State", "sTos", "dTos", "TotPkts",
            "TotBytes", "SrcBytes", "Label",
        ],
        "timestamp_order_in_source": {
            "start_min": str(start_min), "start_max": str(start_max),
            "start_time_regressions_across_chunks": start_order_regressions,
            "completion_min": str(end_min), "completion_max": str(end_max),
        },
        "quality": {
            "invalid_start_times": invalid_times,
            "invalid_durations": invalid_durations,
            "negative_durations": negative_durations,
            "unique_source_addresses": len(src_ip_unique),
            "unique_destination_addresses": len(dst_ip_unique),
        },
        "label_counts": dict(sorted(labels.items())),
        "conservative_label_groups": {
            "attack_from_verified_botnet_host": confirmed_attack,
            "benign_from_verified_normal_host": confirmed_normal,
            "unresolved_background_or_to_host_or_other": unresolved,
            "mapping": (
                "Only From-Botnet is positive and From-Normal is negative. "
                "Background and To-* labels remain unresolved; do not score them as benign."
            ),
        },
        "conservative_label_time_ranges": label_time_ranges,
        "hourly_confirmed_label_counts": dict(sorted(hourly_label_counts.items())),
        "causal_replay_rule": (
            "Order completed flow records by StartTime + Dur because aggregate "
            "flow features are only available once the flow has ended."
        ),
        "limitations": [
            "One scenario from a single university network and a single Neris campaign.",
            "Historical controlled malware capture; not natural prevalence or contemporary traffic.",
            "Background and To-* flows are not ground-truth benign under the conservative label map.",
            "Do not reuse SrcAddr or DstAddr as model features; remove flow identifiers and labels.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "dataset", "sha256", "bytes", "rows", "timestamp_order_in_source",
        "quality", "label_counts", "conservative_label_groups",
    )}, indent=2))
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
