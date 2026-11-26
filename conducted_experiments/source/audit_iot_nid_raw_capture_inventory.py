"""Verify the creator-provided IoT-NID raw PCAP archive against its workbook."""
from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "iot_nid_raw"
PACKETS = DATA / "extracted" / "Packets"
WORKBOOK = DATA / "extracted" / "dataset_description.xlsx"
SOURCE_CSV = ROOT / "data" / "provenance" / "IoTID20_official_original.csv"
OUT = ROOT / "results" / "iot_nid_raw_capture_inventory_20261102.json"
MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1_000_000),
    b"\xa1\xb2\xc3\xd4": (">", 1_000_000),
    b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000),
    b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_pcap(path: Path) -> dict[str, object]:
    count = 0
    first = last = None
    with path.open("rb") as stream:
        header = stream.read(24)
        if len(header) != 24 or header[:4] not in MAGIC:
            raise ValueError(f"Unsupported or invalid classic PCAP header: {path}")
        endian, resolution = MAGIC[header[:4]]
        _, major, minor, _zone, _sigfigs, snaplen, linktype = struct.unpack(endian + "IHHIIII", header)
        while True:
            packet_header = stream.read(16)
            if not packet_header:
                break
            if len(packet_header) != 16:
                raise ValueError(f"Truncated packet header in {path}")
            seconds, fraction, included, _original = struct.unpack(endian + "IIII", packet_header)
            timestamp = seconds + fraction / resolution
            first = timestamp if first is None else first
            last = timestamp
            count += 1
            stream.seek(included, 1)
    to_iso = lambda stamp: datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat() if stamp is not None else None
    return {
        "packets": count,
        "first_packet_utc": to_iso(first),
        "last_packet_utc": to_iso(last),
        "first_packet_epoch": first,
        "last_packet_epoch": last,
        "pcap_version": f"{major}.{minor}",
        "linktype": linktype,
        "snaplen": snaplen,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    workbook = pd.read_excel(WORKBOOK, sheet_name=0, header=2)
    workbook = workbook.dropna(subset=["File Name"])
    inventory = {path.name: inspect_pcap(path) for path in sorted(PACKETS.glob("*.pcap"))}
    expected = {str(row["File Name"]): int(row["# Total Packets"]) for _, row in workbook.iterrows()}
    if set(inventory) != set(expected):
        raise ValueError(f"Workbook/capture filename mismatch; missing={sorted(set(expected)-set(inventory))}; extra={sorted(set(inventory)-set(expected))}")
    matches = {name: inventory[name]["packets"] == expected[name] for name in inventory}
    source = pd.read_csv(SOURCE_CSV, usecols=["Timestamp", "Cat", "Label"], low_memory=False)
    timestamps = pd.to_datetime(source["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    valid = timestamps.notna()
    dates = {datetime.fromtimestamp(info["first_packet_epoch"], tz=timezone.utc).date().isoformat() for info in inventory.values()}
    # CSV timestamps are displayed to whole seconds and are timezone-naive.
    # Search a fixed offset rather than assuming that the CSV uses UTC.
    source_epoch = timestamps.astype("int64").to_numpy(dtype=np.float64) / 1e9
    interval_rows = [
        (float(inventory[name]["first_packet_epoch"]), float(inventory[name]["last_packet_epoch"]), str(row["Category"]))
        for _, row in workbook.iterrows() for name in [str(row["File Name"])]
    ]
    offsets = []
    for offset_minutes in range(-12 * 60, 14 * 60 + 1, 15):
        shifted = source_epoch + offset_minutes * 60
        matched = np.zeros(len(source), dtype=bool)
        for start, end, _category in interval_rows:
            matched |= (shifted >= start - 1.0) & (shifted <= end + 1.0)
        offsets.append({"offset_minutes": offset_minutes, "fraction_in_any_capture_interval": float(matched[valid.to_numpy()].mean())})
    best = max(offsets, key=lambda item: item["fraction_in_any_capture_interval"])
    best_epoch = source_epoch + best["offset_minutes"] * 60
    candidate_count = np.zeros(len(source), dtype=np.uint8)
    candidate_categories: list[set[str]] = [set() for _ in range(len(source))]
    for start, end, category in interval_rows:
        inside = (best_epoch >= start - 1.0) & (best_epoch <= end + 1.0) & valid.to_numpy()
        candidate_count += inside.astype(np.uint8)
        for idx in np.flatnonzero(inside):
            candidate_categories[int(idx)].add(category)
    category_map = {
        "normal": "normal",
        "dos": "denial of service (dos)",
        "mirai": "mirai botnet",
        "scan": "scanning",
        "mitm arp spoofing": "man in the middle (mitm)",
    }
    csv_category = source["Cat"].astype(str).str.strip().str.casefold().map(category_map)
    category_compatible = np.array([
        bool(cats) and all(cat.casefold() == csv_category.iloc[idx] for cat in cats)
        for idx, cats in enumerate(candidate_categories)
    ])
    interval_match = candidate_count > 0
    result = {
        "status": "complete",
        "date": "2026-11-02",
        "source": "HCRL creator-linked IoT Network Intrusion Dataset PCAP archive",
        "archive_sha256": json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))["archive_sha256"],
        "capture_count": len(inventory),
        "total_pcap_packets": int(sum(x["packets"] for x in inventory.values())),
        "workbook_packet_count_match": {"matched": int(sum(matches.values())), "total": len(matches)},
        "pcap_capture_dates_utc": sorted(dates),
        "timestamp_alignment": {
            "best_csv_to_pcap_offset_minutes": int(best["offset_minutes"]),
            "interval_tolerance_seconds": 1.0,
            "rows_inside_any_capture_interval": int(interval_match.sum()),
            "valid_timestamp_rows": int(valid.sum()),
            "fraction_inside_any_capture_interval": float(interval_match[valid.to_numpy()].mean()),
            "rows_with_overlapping_capture_intervals": int((candidate_count > 1).sum()),
            "rows_with_conflicting_capture_categories": int(sum(len(cats) > 1 for cats in candidate_categories)),
            "rows_whose_csv_category_matches_all_time_candidate_capture_categories": int(category_compatible.sum()),
            "category_match_fraction_of_valid_rows": float(category_compatible[valid.to_numpy()].mean()),
            "best_offset_search_grid_minutes": 15,
            "best_offset_and_next_best": sorted(offsets, key=lambda item: -item["fraction_in_any_capture_interval"])[:5],
        },
        "iotid20_csv": {
            "rows": int(len(source)),
            "invalid_timestamps": int((~valid).sum()),
            "category_counts": {str(k): int(v) for k, v in source["Cat"].value_counts(dropna=False).items()},
            "binary_counts": {str(k): int(v) for k, v in source["Label"].value_counts(dropna=False).items()},
        },
        "captures": [
            {"file": name, "workbook_packet_count": expected[name], "workbook_count_matches": matches[name], **inventory[name]}
            for name in sorted(inventory)
        ],
        "limits": [
            "A matching packet count validates archive/workbook integrity but does not verify IoTID20 flow boundaries or feature values.",
            "Timestamp-interval and category agreement support source lineage but do not prove the complete packet-to-flow label aggregation procedure.",
            "The best fixed offset is an observed alignment, not a verified timezone or exporter configuration setting.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "captures": result["capture_count"],
        "packets": result["total_pcap_packets"],
        "workbook_count_matches": result["workbook_packet_count_match"],
        "iotid20_rows": result["iotid20_csv"]["rows"],
        "timestamp_alignment": result["timestamp_alignment"],
        "capture_dates": result["pcap_capture_dates_utc"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
