"""Check creator workbook attack filters against the raw IoT-NID PCAPs.

This audit only verifies the published packet filters and counts. It does not
assign packet labels to flows or claim to reproduce IoTID20's label-lifting
procedure.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path, PurePosixPath

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "data" / "iot_nid_raw" / "extracted" / "dataset_description.xlsx"
PCAP_ROOT = ROOT / "data" / "iot_nid_raw" / "extracted" / "Packets"
INVENTORY = ROOT / "results" / "iot_nid_raw_capture_inventory_20261102.json"
OUTPUT = ROOT / "results" / "iot_nid_attack_filter_audit_20261103.json"
WSL_ROOT = (
    PurePosixPath("/mnt") / ROOT.drive.rstrip(":").lower()
    / PurePosixPath(*ROOT.relative_to(Path(ROOT.anchor)).parts)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_scalar(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def read_events() -> tuple[list[dict], dict[str, dict]]:
    book = load_workbook(WORKBOOK, read_only=True, data_only=True)
    sheet = book["Files & description"]
    previous: dict[int, object] = {}
    events: list[dict] = []
    captures: dict[str, dict] = {}
    for row_number, row in enumerate(sheet.iter_rows(min_row=3, values_only=True), start=3):
        # Columns A:J: number, filename, date, size, target, category, subtype,
        # attack display filter, packet count, filtered attack packet count.
        values = list(row[:10])
        for index in (1, 2, 3, 4, 5, 6, 8):
            if values[index] is None and index in previous:
                values[index] = previous[index]
            elif values[index] is not None:
                previous[index] = values[index]
        (number, filename, creation_date, file_kb, target, category, subtype,
         rule, packet_count, attack_count) = values
        if not filename or not isinstance(filename, str) or not filename.lower().endswith(".pcap"):
            continue
        capture = captures.setdefault(filename, {
            "file": filename,
            "creation_date": json_scalar(creation_date),
            "file_size_kb": file_kb,
            "target_device": target,
            "category": category,
            "packet_count": int(packet_count) if packet_count is not None else None,
            "events": [],
        })
        filter_text = str(rule).strip() if rule is not None else ""
        if filter_text and filter_text != "-":
            if attack_count is None:
                raise ValueError(f"Missing expected packet count at workbook row {row_number}")
            events.append({
                "workbook_row": row_number,
                "file": filename,
                "category": category,
                "subcategory": subtype,
                "target_device": target,
                "filter": filter_text,
                "expected_attack_packets": int(attack_count),
            })
            capture["events"].append(len(events) - 1)
    book.close()
    return events, captures


def main() -> None:
    events, captures = read_events()
    with INVENTORY.open(encoding="utf-8") as handle:
        inventory = json.load(handle)
    inventory_by_file = {row["file"]: row for row in inventory["captures"]}
    pcap_root_linux = WSL_ROOT / "data/iot_nid_raw/extracted/Packets"

    results = []
    for index, event in enumerate(events, start=1):
        pcap = PCAP_ROOT / event["file"]
        if not pcap.is_file():
            raise FileNotFoundError(pcap)
        cmd = [
            "wsl", "-e", "tshark", "-r", str(pcap_root_linux / event["file"]),
            "-Y", event["filter"], "-T", "fields", "-e", "frame.number",
        ]
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
        if run.returncode != 0:
            raise RuntimeError(
                f"tshark failed for {event['file']} row {event['workbook_row']} "
                f"(exit {run.returncode}): {run.stderr[-2000:]}"
            )
        observed = sum(1 for line in run.stdout.splitlines() if line.strip())
        results.append({
            **event,
            "observed_attack_packets": observed,
            "count_matches_workbook": observed == event["expected_attack_packets"],
        })
        print(f"[{index:02d}/{len(events)}] {event['file']} / {event['subcategory']}: "
              f"{observed} vs {event['expected_attack_packets']}", flush=True)

    capture_summaries = []
    for filename, capture in captures.items():
        raw = inventory_by_file.get(filename)
        if raw is None:
            raise ValueError(f"Workbook capture absent from raw inventory: {filename}")
        capture_summaries.append({
            "file": filename,
            "category": capture["category"],
            "packet_count_workbook": capture["packet_count"],
            "packet_count_pcap": raw["packets"],
            "packet_count_matches": capture["packet_count"] == raw["packets"],
            "pcap_sha256": raw["sha256"],
            "event_rule_count": len(capture["events"]),
            "event_categories": sorted({results[i]["subcategory"] for i in capture["events"]}),
        })

    result = {
        "status": "complete",
        "date": "2026-11-03",
        "purpose": "Validate creator-provided packet filters and workbook packet counts; no flow-label assignment is performed.",
        "source_workbook": str(WORKBOOK.relative_to(ROOT)),
        "source_workbook_sha256": sha256(WORKBOOK),
        "tshark_version": subprocess.run(
            ["wsl", "-e", "tshark", "--version"], capture_output=True, text=True,
            timeout=30, check=True,
        ).stdout.splitlines()[0],
        "capture_count": len(captures),
        "filter_event_count": len(events),
        "capture_packet_counts_match_raw_inventory": sum(x["packet_count_matches"] for x in capture_summaries),
        "filter_packet_counts_match_workbook": sum(x["count_matches_workbook"] for x in results),
        "filter_packet_counts_total": len(results),
        "captures": capture_summaries,
        "filter_events": results,
        "limitations": [
            "A matching packet-filter count validates the workbook rule against the distributed PCAP, not its flow-label aggregation policy.",
            "Per-event packet counts are not summed as unique malicious packets because filters within a capture may overlap.",
            "No labels are assigned to flows in this audit.",
        ],
    }
    result["status"] = "complete" if (
        result["capture_packet_counts_match_raw_inventory"] == result["capture_count"]
        and result["filter_packet_counts_match_workbook"] == result["filter_packet_counts_total"]
    ) else "mismatch"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "status", "capture_count", "filter_event_count",
        "capture_packet_counts_match_raw_inventory", "filter_packet_counts_match_workbook",
        "filter_packet_counts_total",
    )}, indent=2))


if __name__ == "__main__":
    main()
