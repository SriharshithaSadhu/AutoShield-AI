"""Measure validated creator-filter packet coverage in Argus v5 flow intervals.

This is a conservative positive-evidence audit: tuple/time overlap marks a packet
as represented by one or more Argus flow intervals. It does not prove packet-to-
flow membership, and nonmatching flows are never labeled benign.
"""
from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
FILTERS = ROOT / "results" / "iot_nid_attack_filter_audit_20261103.json"
ARGUS_DIR = ROOT / "results" / "flow_exporter_pilot" / "argus_v5_42" / "csv"
PCAP_DIR = (PurePosixPath("/mnt") / ROOT.drive.rstrip(":").lower()
            / PurePosixPath(*ROOT.relative_to(Path(ROOT.anchor)).parts)
            / "data/iot_nid_raw/extracted/Packets")
OUT = ROOT / "results" / "flow_exporter_pilot" / "argus_v5_attack_packet_coverage.json"
TOLERANCE_SECONDS = 0.01


def endpoint(addr: str, port: str) -> tuple[str, int]:
    return (addr.strip().lower(), int(float(port)) if port.strip() else 0)


def tuple_key(src: str, sport: str, dst: str, dport: str, proto: int):
    return (*sorted((endpoint(src, sport), endpoint(dst, dport))), proto)


def protocol_number(name: str, ipproto: str, tcp: str, udp: str,
                    icmp: str, icmpv6: str) -> int | None:
    if tcp:
        return 6
    if udp:
        return 17
    if icmp:
        return 1
    if icmpv6:
        return 58
    if ipproto.strip():
        try:
            return int(ipproto)
        except ValueError:
            pass
    return {"tcp": 6, "udp": 17, "icmp": 1, "icmpv6": 58}.get(name.strip().lower())


def read_argus(path: Path) -> dict[tuple, list[tuple[float, float]]]:
    index: dict[tuple, list[tuple[int, float, float]]] = defaultdict(list)
    row_index = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_index += 1
            try:
                proto_name = row["Proto"].strip().lower()
                proto = {"tcp": 6, "udp": 17, "icmp": 1, "icmpv6": 58}.get(proto_name)
                if proto is None:
                    continue
                start = float(row["StartTime"])
                end = float(row["LastTime"])
                key = tuple_key(row["SrcAddr"], row["Sport"], row["DstAddr"], row["Dport"], proto)
            except (ValueError, TypeError, KeyError):
                continue
            index[key].append((row_index, start, end))
    for intervals in index.values():
        intervals.sort(key=lambda item: (item[1], item[2], item[0]))
    return index


def extract_event_packets(pcap: Path, display_filter: str) -> tuple[int, list[dict], int]:
    fields = ["frame.time_epoch", "ip.src", "ip.dst", "ipv6.src", "ipv6.dst",
              "ip.proto", "ipv6.nxt", "tcp.srcport", "tcp.dstport", "udp.srcport",
              "udp.dstport", "icmp.type", "icmpv6.type"]
    cmd = ["wsl", "-e", "tshark", "-r", str(pcap), "-Y", display_filter,
           "-T", "fields", "-E", "separator=\t", "-E", "occurrence=f"]
    for field in fields:
        cmd.extend(["-e", field])
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if run.returncode:
        raise RuntimeError(f"TShark failed for {pcap.name}: {run.stderr[-1500:]}")
    packets = []
    timestamp_parse_failures = 0
    total = 0
    for vals in csv.reader(run.stdout.splitlines(), delimiter="\t"):
        vals += [""] * (len(fields) - len(vals))
        total += 1
        (time_s, ip_src, ip_dst, ip6_src, ip6_dst, ipproto, ip6proto,
         tcp_src, tcp_dst, udp_src, udp_dst, icmp, icmpv6) = vals[:len(fields)]
        src, dst = (ip_src, ip_dst) if ip_src and ip_dst else (ip6_src, ip6_dst)
        if not time_s or not src or not dst:
            continue
        proto = protocol_number("", ipproto or ip6proto, tcp_src, udp_src, icmp, icmpv6)
        if proto is None:
            continue
        sport, dport = ((tcp_src, tcp_dst) if tcp_src else
                        (udp_src, udp_dst) if udp_src else ("", ""))
        try:
            packet_time = float(time_s)
        except ValueError:
            packet_time = None
            timestamp_parse_failures += 1
        packets.append({"time": packet_time,
                        "key": tuple_key(src, sport, dst, dport, proto)})
    return total, packets, timestamp_parse_failures


def main() -> None:
    audit = json.loads(FILTERS.read_text(encoding="utf-8"))
    results = []
    for i, event in enumerate(audit["filter_events"], 1):
        pcap_name = event["file"]
        pcap = PCAP_DIR / pcap_name
        flow_csv = ARGUS_DIR / (Path(pcap_name).stem + ".60s.csv")
        pcap_exists = subprocess.run(["wsl", "-e", "test", "-f", str(pcap)],
                                     capture_output=True, check=False).returncode == 0
        if not pcap_exists or not flow_csv.exists():
            raise FileNotFoundError(f"Missing paired input for {pcap_name}")
        total, packets, timestamp_failures = extract_event_packets(pcap, event["filter"])
        index = read_argus(flow_csv)
        linked = ambiguous = no_candidate = 0
        candidate_but_outside = 0
        touched_flow_rows = set()
        for packet in packets:
            if packet["time"] is None:
                continue
            intervals = index.get(packet["key"], [])
            matches = [iv for iv in intervals
                       if iv[1] - TOLERANCE_SECONDS <= packet["time"] <= iv[2] + TOLERANCE_SECONDS]
            if matches:
                linked += 1
                ambiguous += len(matches) > 1
                touched_flow_rows.update(iv[0] for iv in matches)
            elif intervals:
                candidate_but_outside += 1
            else:
                no_candidate += 1
        row = {
            "capture": pcap_name,
            "category": event["category"],
            "subcategory": event["subcategory"],
            "creator_expected_packet_count": int(event["expected_attack_packets"]),
            "tshark_filter_packet_count": total,
            "filter_count_matches_creator_audit": total == int(event["expected_attack_packets"]),
            "filter_packets_with_ip_tuple_and_supported_protocol": len(packets),
            "filter_packets_with_unparseable_timestamps": timestamp_failures,
            "tuple_and_time_parseable_packet_count": len(packets) - timestamp_failures,
            "packet_tuple_time_matches_argus_flow_interval": linked,
            "coverage_of_parseable_tuple_packets": linked / (len(packets) - timestamp_failures) if len(packets) > timestamp_failures else None,
            "argus_flow_rows_with_creator_filter_packet_overlap": len(touched_flow_rows),
            "ambiguous_packet_matches_multiple_argus_flows": ambiguous,
            "same_tuple_flows_but_packet_outside_all_intervals": candidate_but_outside,
            "no_same_tuple_argus_flow": no_candidate,
            "argus_flow_count": sum(map(len, index.values())),
        }
        results.append(row)
        print(f"[{i:02d}/{len(audit['filter_events'])}] {pcap_name} / {event['subcategory']}: "
              f"{linked}/{len(packets)} tuple-representable packets covered; filter={total}", flush=True)

    summary = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "argus_version": "5.0.4",
        "argus_flow_options": "HERA-style -S 60 -A -Z -J projection",
        "creator_filter_audit": FILTERS.relative_to(ROOT).as_posix(),
        "flow_interval_match_tolerance_seconds": TOLERANCE_SECONDS,
        "event_count": len(results),
        "events_reproducing_creator_filter_count": sum(r["filter_count_matches_creator_audit"] for r in results),
        "total_creator_filter_packets": sum(r["tshark_filter_packet_count"] for r in results),
        "total_tuple_representable_filter_packets": sum(r["filter_packets_with_ip_tuple_and_supported_protocol"] for r in results),
        "total_filter_packets_with_unparseable_timestamps": sum(r["filter_packets_with_unparseable_timestamps"] for r in results),
        "total_tuple_and_time_parseable_packets": sum(r["tuple_and_time_parseable_packet_count"] for r in results),
        "total_packets_matching_argus_flow_interval": sum(r["packet_tuple_time_matches_argus_flow_interval"] for r in results),
        "event_results": results,
        "limitations": [
            "Tuple plus interval overlap is a representation/coverage proxy, not proof of packet membership in the exporter record.",
            "Creator event filters establish positive packet evidence only; unmatched flows in attack captures are unknown, not benign.",
            "ARP and other packets without supported IP tuple fields are reported as not tuple-representable.",
            "Multiple creator rules may select the same packet; event totals are not unique-packet totals.",
            "No common-feature audit or classifier evaluation is performed."
        ]
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "event_results"}, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
