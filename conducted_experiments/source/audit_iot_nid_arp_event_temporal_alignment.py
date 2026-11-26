"""Compare creator-selected ARP event intervals with published IoTID20 flow labels."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import pandas as pd

ROOT = Path(__file__).resolve().parent
FILTER_AUDIT = ROOT / "results" / "iot_nid_attack_filter_audit_20261103.json"
RAW_INVENTORY = ROOT / "results" / "iot_nid_raw_capture_inventory_20261102.json"
OFFICIAL = ROOT / "data" / "provenance" / "IoTID20_official_original.csv"
OUTPUT = ROOT / "results" / "iot_nid_arp_event_temporal_alignment_20261103.json"
WSL_ROOT = (
    PurePosixPath("/mnt") / ROOT.drive.rstrip(":").lower()
    / PurePosixPath(*ROOT.relative_to(Path(ROOT.anchor)).parts)
)
WSL_PCAPS = WSL_ROOT / "data/iot_nid_raw/extracted/Packets"
CSV_TO_PCAP = timedelta(hours=4)  # Observed alignment from the separate inventory audit.


def selected_arp_times(event: dict) -> list[float]:
    run = subprocess.run(
        ["wsl", "-e", "tshark", "-r", str(WSL_PCAPS / event["file"]),
         "-Y", f"({event['filter']}) && arp", "-T", "fields", "-e", "frame.time_epoch"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    if run.returncode:
        raise RuntimeError(f"tshark failed for {event['file']}: {run.stderr[-2000:]}")
    return [float(line) for line in run.stdout.splitlines() if line.strip()]


def main() -> None:
    filter_audit = json.loads(FILTER_AUDIT.read_text(encoding="utf-8"))
    inventory = json.loads(RAW_INVENTORY.read_text(encoding="utf-8"))
    capture_by_file = {c["file"]: c for c in inventory["captures"]}
    arp_events = [e for e in filter_audit["filter_events"] if "arp." in e["filter"].lower()]
    epochs = {}
    for event in arp_events:
        times = selected_arp_times(event)
        if not times:
            raise ValueError(f"No ARP packets selected for {event['file']} / {event['subcategory']}")
        epochs[event["file"]] = {
            "event": event,
            "first": min(times),
            "last": max(times),
            "packet_count": len(times),
        }
        print(f"ARP event {event['file']}: {len(times)} frames", flush=True)

    keys = ["Timestamp", "Flow_Duration", "Label", "Cat", "Sub_Cat"]
    chunks = []
    for chunk in pd.read_csv(OFFICIAL, usecols=keys, chunksize=50_000, low_memory=False):
        parsed = pd.to_datetime(chunk["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
        chunk = chunk.assign(_time=parsed)
        chunks.append(chunk.loc[chunk["_time"].notna(), keys + ["_time"]])
    flows = pd.concat(chunks, ignore_index=True)
    flows["_end"] = flows["_time"] + pd.to_timedelta(
        pd.to_numeric(flows["Flow_Duration"], errors="coerce").fillna(0).clip(lower=0) / 1_000_000,
        unit="s",
    )

    results = []
    for file, row in epochs.items():
        event = row["event"]
        # The inventory found PCAP time = CSV time + four hours; convert the
        # selected PCAP event interval into the CSV's local timestamp scale.
        start = datetime.fromtimestamp(row["first"], tz=timezone.utc).replace(tzinfo=None) - CSV_TO_PCAP
        end = datetime.fromtimestamp(row["last"], tz=timezone.utc).replace(tzinfo=None) - CSV_TO_PCAP
        cap = capture_by_file[file]
        cap_start = datetime.fromisoformat(cap["first_packet_utc"]).replace(tzinfo=None) - CSV_TO_PCAP
        cap_end = datetime.fromisoformat(cap["last_packet_utc"]).replace(tzinfo=None) - CSV_TO_PCAP
        cap_mask = flows["_time"].between(cap_start, cap_end, inclusive="both")
        event_start_mask = flows["_time"].between(start, end, inclusive="both")
        overlap_mask = (flows["_time"] <= end) & (flows["_end"] >= start)
        results.append({
            "file": file,
            "category": event["category"],
            "subcategory": event["subcategory"],
            "selected_arp_packet_count": row["packet_count"],
            "arp_event_start_csv_scale": start.isoformat(sep=" "),
            "arp_event_end_csv_scale": end.isoformat(sep=" "),
            "capture_flow_rows": int(cap_mask.sum()),
            "capture_flow_label_counts": {str(k): int(v) for k, v in flows.loc[cap_mask, "Sub_Cat"].value_counts(dropna=False).items()},
            "flow_start_inside_arp_event_rows": int(event_start_mask.sum()),
            "flow_start_inside_arp_event_subcategory_counts": {str(k): int(v) for k, v in flows.loc[event_start_mask, "Sub_Cat"].value_counts(dropna=False).items()},
            "flow_duration_overlaps_arp_event_rows": int(overlap_mask.sum()),
            "flow_duration_overlap_subcategory_counts": {str(k): int(v) for k, v in flows.loc[overlap_mask, "Sub_Cat"].value_counts(dropna=False).items()},
        })

    output = {
        "status": "complete",
        "date": "2026-11-03",
        "method": "Selected packet timestamps are taken directly from validated creator Wireshark filters; published flow start timestamps and duration are compared after applying the previously observed four-hour CSV-to-PCAP alignment.",
        "csv_to_pcap_offset_minutes_observed": 240,
        "event_count": len(results),
        "events": results,
        "limitations": [
            "The four-hour offset is empirical and is not a verified timezone configuration.",
            "Flows are counted by timestamp containment or duration overlap only; this is not proof of packet-to-flow identity.",
            "No labels are modified or assigned.",
        ],
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": output["status"],
        "event_count": len(results),
        "rows_overlapping_arp_events": sum(x["flow_duration_overlaps_arp_event_rows"] for x in results),
        "rows_starting_inside_arp_events": sum(x["flow_start_inside_arp_event_rows"] for x in results),
    }, indent=2))


if __name__ == "__main__":
    main()
