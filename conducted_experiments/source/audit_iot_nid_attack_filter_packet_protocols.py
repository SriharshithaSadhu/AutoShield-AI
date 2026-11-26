"""Count packet protocols selected by creator rules containing ARP clauses."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "iot_nid_attack_filter_audit_20261103.json"
OUTPUT = ROOT / "results" / "iot_nid_attack_filter_protocol_audit_20261103.json"
WSL_ROOT = (
    PurePosixPath("/mnt") / ROOT.drive.rstrip(":").lower()
    / PurePosixPath(*ROOT.relative_to(Path(ROOT.anchor)).parts)
)
WSL_PCAPS = WSL_ROOT / "data/iot_nid_raw/extracted/Packets"


def count(pcap: str, filt: str) -> int:
    run = subprocess.run(
        ["wsl", "-e", "tshark", "-r", str(WSL_PCAPS / pcap), "-Y", filt,
         "-T", "fields", "-e", "frame.number"],
        capture_output=True, text=True, timeout=300, check=False,
    )
    if run.returncode:
        raise RuntimeError(f"tshark failed for {pcap}: {run.stderr[-2000:]}")
    return sum(bool(line.strip()) for line in run.stdout.splitlines())


def main() -> None:
    audit = json.loads(SOURCE.read_text(encoding="utf-8"))
    selected = [e for e in audit["filter_events"] if "arp." in e["filter"].lower()]
    rows = []
    for i, event in enumerate(selected, start=1):
        base = f"({event['filter']})"
        counts = {proto: count(event["file"], f"{base} && {proto}")
                  for proto in ("arp", "ip", "ipv6")}
        rows.append({
            "file": event["file"],
            "subcategory": event["subcategory"],
            "expected_filter_packets": event["expected_attack_packets"],
            "arp_packets": counts["arp"],
            "ipv4_packets": counts["ip"],
            "ipv6_packets": counts["ipv6"],
            "protocol_subcounts_sum": sum(counts.values()),
        })
        print(f"[{i}/{len(selected)}] {event['file']} / {event['subcategory']}: "
              f"ARP={counts['arp']} IPv4={counts['ip']} IPv6={counts['ipv6']}", flush=True)
    output = {
        "status": "complete",
        "date": "2026-11-03",
        "method": "For validated workbook filters containing ARP fields, count selected ARP, IPv4, and IPv6 packets independently with TShark 4.2.2.",
        "rule_count": len(rows),
        "events": rows,
        "limitations": [
            "Protocol counts are not assumed mutually exclusive; packet decapsulation may expose multiple protocol layers.",
            "This does not map packets to flows or assign labels.",
        ],
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": output["status"], "rule_count": len(rows),
        "total_selected_arp_packets": sum(r["arp_packets"] for r in rows),
        "arp_rules_with_selected_packets": sum(r["arp_packets"] > 0 for r in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
