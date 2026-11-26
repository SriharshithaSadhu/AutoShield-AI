"""Match validated DoS-SYN packet-rule hits to CFM-V3 and official IoTID20 flows."""
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import pandas as pd

ROOT = Path(__file__).resolve().parent
FILTER_AUDIT = ROOT / "results" / "iot_nid_attack_filter_audit_20261103.json"
INVENTORY = ROOT / "results" / "iot_nid_raw_capture_inventory_20261102.json"
OFFICIAL_CSV = ROOT / "data" / "provenance" / "IoTID20_official_original.csv"
CFM_CSV = ROOT / "results" / "iot_nid_cfm_v3_pilot" / "dos-synflooding-5-dec.pcap_flowTimeout_100000.csv"
OUTPUT = ROOT / "results" / "iot_nid_dos_syn_packet_flow_correspondence_20261103.json"
PCAPS = (PurePosixPath("/mnt") / ROOT.drive.rstrip(":").lower()
         / PurePosixPath(*ROOT.relative_to(Path(ROOT.anchor)).parts)
         / "data/iot_nid_raw/extracted/Packets")
FILE = "dos-synflooding-5-dec.pcap"
OFFSET_TO_PCAP = timedelta(hours=4)  # Previously observed: CSV time + 4h aligns to PCAP time.
TOLERANCE = timedelta(seconds=1)


def tuple_key(src: str, sport: object, dst: str, dport: object, proto: object) -> tuple:
    a = (str(src), int(float(sport or 0)))
    b = (str(dst), int(float(dport or 0)))
    return (*sorted((a, b)), int(float(proto or 0)))


def parse_cfm_time(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%d/%m/%Y %H:%M:%S").replace(tzinfo=timezone.utc)


def parse_official_time(value: str) -> datetime:
    parsed = datetime.strptime(value.strip(), "%d/%m/%Y %I:%M:%S %p")
    return (parsed.replace(tzinfo=timezone.utc) + OFFSET_TO_PCAP)


def indexed_flows(frame: pd.DataFrame, src: str, sport: str, dst: str, dport: str,
                  proto: str, timestamp: str, duration: str, official: bool) -> dict[tuple, list[dict]]:
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for row_index, row in frame.iterrows():
        try:
            start = parse_official_time(row[timestamp]) if official else parse_cfm_time(row[timestamp])
            dur = max(float(row[duration] or 0), 0.0) / 1_000_000.0
            key = tuple_key(row[src], row[sport], row[dst], row[dport], row[proto])
        except (ValueError, TypeError, OverflowError):
            continue
        by_key[key].append({"index": int(row_index), "start": start,
                            "end": start + timedelta(seconds=dur),
                            "category": str(row.get("Cat", "")),
                            "subcategory": str(row.get("Sub_Cat", "")),
                            "label": str(row.get("Label", ""))})
    return by_key


def links(packets: list[dict], flow_index: dict[tuple, list[dict]]) -> tuple[set[int], int, int, list[dict]]:
    linked_rows: set[int] = set()
    linked_packets = 0
    linked_pairs = 0
    unmatched = []
    for packet in packets:
        candidates = flow_index.get(packet["key"], [])
        hits = [flow for flow in candidates
                if flow["start"] - TOLERANCE <= packet["time"] <= flow["end"] + TOLERANCE]
        if hits:
            linked_packets += 1
            linked_rows.update(flow["index"] for flow in hits)
            linked_pairs += len(hits)
        else:
            nearest_gap = min((
                min(abs((packet["time"] - flow["start"]).total_seconds()),
                    abs((packet["time"] - flow["end"]).total_seconds()))
                for flow in candidates
            ), default=None)
            unmatched.append({**packet,
                              "same_tuple_candidate_flows": len(candidates),
                              "nearest_flow_boundary_gap_seconds": nearest_gap})
    return linked_rows, linked_packets, linked_pairs, unmatched


def summarize_unmatched(rows: list[dict]) -> dict:
    by_key = Counter(row["key"] for row in rows)
    signatures = Counter(tuple(row.get(name, "") for name in (
        "ip_flags_mf", "ip_fragment_offset", "tcp_header_length", "tcp_payload_length",
        "ip_checksum_status", "tcp_checksum_status", "captured_length", "wire_length",
    )) for row in rows)
    return {
        "packet_count": len(rows),
        "first_packet_time_utc": min((row["time"] for row in rows), default=None).isoformat() if rows else None,
        "last_packet_time_utc": max((row["time"] for row in rows), default=None).isoformat() if rows else None,
        "packets_with_same_tuple_candidate_but_outside_flow_interval": sum(
            row["same_tuple_candidate_flows"] > 0 for row in rows
        ),
        "packets_with_no_same_tuple_candidate": sum(
            row["same_tuple_candidate_flows"] == 0 for row in rows
        ),
        "nearest_boundary_gap_seconds_min": min((
            row["nearest_flow_boundary_gap_seconds"] for row in rows
            if row["nearest_flow_boundary_gap_seconds"] is not None
        ), default=None),
        "top_bidirectional_tuples": [
            {"key": [*key], "unmatched_packet_count": n}
            for key, n in by_key.most_common(10)
        ],
        "unmatched_packet_structure_counts": [
            {"ip_flags_mf": key[0], "ip_fragment_offset": key[1],
             "tcp_header_length": key[2], "tcp_payload_length": key[3],
             "ip_checksum_status": key[4], "tcp_checksum_status": key[5],
             "captured_length": key[6], "wire_length": key[7],
             "packet_count": n}
            for key, n in signatures.most_common(10)
        ],
    }


def count_pcap_packets_for_keys(rows: list[dict]) -> dict:
    keys = sorted({row["key"] for row in rows})
    if not keys:
        return {"unique_tuple_count": 0, "total_packets_for_tuples": 0, "packet_count_distribution": {}}
    clauses = []
    for (ip_a, port_a), (ip_b, port_b), proto in keys:
        ab = f"(ip.src == {ip_a} && tcp.srcport == {port_a} && ip.dst == {ip_b} && tcp.dstport == {port_b})"
        ba = f"(ip.src == {ip_b} && tcp.srcport == {port_b} && ip.dst == {ip_a} && tcp.dstport == {port_a})"
        clauses.append(f"(ip.proto == {proto} && ({ab} || {ba}))")
    run = subprocess.run([
        "wsl", "-e", "tshark", "-r", str(PCAPS / FILE), "-Y", "tcp && (" + " || ".join(clauses) + ")",
        "-T", "fields", "-E", "separator=\t", "-e", "ip.src", "-e", "tcp.srcport",
        "-e", "ip.dst", "-e", "tcp.dstport", "-e", "ip.proto", "-e", "frame.number",
        "-e", "frame.time_epoch", "-e", "tcp.flags", "-e", "ip.len",
    ], capture_output=True, text=True, timeout=300, check=False)
    if run.returncode:
        raise RuntimeError(f"tshark tuple recount failed: {run.stderr[-2000:]}")
    counts = Counter()
    packet_details = defaultdict(list)
    for packet in csv.reader(run.stdout.splitlines(), delimiter="\t"):
        if len(packet) >= 5 and packet[0] and packet[2]:
            key = tuple_key(packet[0], packet[1], packet[2], packet[3], packet[4])
            counts[key] += 1
            packet_details[key].append({
                "frame_number": packet[5] if len(packet) > 5 else "",
                "time_epoch": packet[6] if len(packet) > 6 else "",
                "tcp_flags": packet[7] if len(packet) > 7 else "",
                "ip_length": packet[8] if len(packet) > 8 else "",
            })
    return {
        "unique_tuple_count": len(keys),
        "total_packets_for_tuples": sum(counts.values()),
        "packet_count_distribution": {str(n): int(v) for n, v in sorted(Counter(counts.values()).items())},
        "tuples_with_no_other_packet_in_capture": sum(counts[key] == 1 for key in keys),
        "multi_packet_tuple_details": [
            {"key": [*key], "packet_count": counts[key], "packets": packet_details[key]}
            for key in keys if counts[key] > 1
        ],
        "tshark_filter_scope": "all TCP packets in the capture matching the 42 bidirectional 5-tuples; not limited to the creator attack filter",
    }


def summarize_unlinked_flows(frame: pd.DataFrame, official: bool) -> dict:
    if official:
        flow_id, fwd, bwd = "Flow_ID", "Tot_Fwd_Pkts", "Tot_Bwd_Pkts"
        category, subcategory = "Cat", "Sub_Cat"
    else:
        flow_id, fwd, bwd = "Flow ID", "Tot Fwd Pkts", "Tot Bwd Pkts"
        category, subcategory = None, None
    records = []
    for _, row in frame.head(12).iterrows():
        records.append({
            "flow_id": str(row.get(flow_id, "")),
            "protocol": int(float(row.get("Protocol", 0) or 0)),
            "forward_packets": float(row.get(fwd, 0) or 0),
            "backward_packets": float(row.get(bwd, 0) or 0),
            "category": str(row.get(category, "")) if category else None,
            "subcategory": str(row.get(subcategory, "")) if subcategory else None,
        })
    return {
        "flow_count": int(len(frame)),
        "protocol_counts": {str(k): int(v) for k, v in frame["Protocol"].value_counts(dropna=False).items()},
        "category_counts": {str(k): int(v) for k, v in frame[category].value_counts(dropna=False).items()} if category else {},
        "subcategory_counts": {str(k): int(v) for k, v in frame[subcategory].value_counts(dropna=False).items()} if subcategory else {},
        "sample_flows": records,
    }


def main() -> None:
    audit = json.loads(FILTER_AUDIT.read_text(encoding="utf-8"))
    event = next(row for row in audit["filter_events"]
                 if row["file"] == FILE and row["subcategory"] == "SYN Flooding")
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    capture = next(row for row in inventory["captures"] if row["file"] == FILE)

    run = subprocess.run([
        "wsl", "-e", "tshark", "-r", str(PCAPS / FILE), "-Y", event["filter"],
        "-T", "fields", "-E", "separator=\t", "-e", "frame.time_epoch",
        "-e", "ip.src", "-e", "tcp.srcport", "-e", "ip.dst", "-e", "tcp.dstport", "-e", "ip.proto",
        "-e", "frame.number", "-e", "ip.flags.mf", "-e", "ip.frag_offset", "-e", "tcp.hdr_len",
        "-e", "tcp.len", "-e", "ip.checksum.status", "-e", "tcp.checksum.status", "-e", "frame.cap_len", "-e", "frame.len",
    ], capture_output=True, text=True, timeout=300, check=False)
    if run.returncode:
        raise RuntimeError(f"tshark failed: {run.stderr[-2000:]}")
    packets = []
    for row in csv.reader(run.stdout.splitlines(), delimiter="\t"):
        if len(row) < 6 or not row[0] or not row[1] or not row[3]:
            continue
        packets.append({
            "time": datetime.fromtimestamp(float(row[0]), tz=timezone.utc),
            "key": tuple_key(row[1], row[2], row[3], row[4], row[5]),
            "frame_number": row[6] if len(row) > 6 else "",
            "ip_flags_mf": row[7] if len(row) > 7 else "",
            "ip_fragment_offset": row[8] if len(row) > 8 else "",
            "tcp_header_length": row[9] if len(row) > 9 else "",
            "tcp_payload_length": row[10] if len(row) > 10 else "",
            "ip_checksum_status": row[11] if len(row) > 11 else "",
            "tcp_checksum_status": row[12] if len(row) > 12 else "",
            "captured_length": row[13] if len(row) > 13 else "",
            "wire_length": row[14] if len(row) > 14 else "",
        })

    cfm = pd.read_csv(CFM_CSV, low_memory=False)
    cfm_index = indexed_flows(cfm, "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol",
                              "Timestamp", "Flow Duration", official=False)
    cfm_rows, cfm_packets, cfm_pairs, cfm_unmatched = links(packets, cfm_index)
    unmatched_tuple_packet_recount = count_pcap_packets_for_keys(cfm_unmatched)

    cols = ["Flow_ID", "Timestamp", "Flow_Duration", "Src_IP", "Src_Port", "Dst_IP", "Dst_Port",
            "Protocol", "Tot_Fwd_Pkts", "Tot_Bwd_Pkts", "Label", "Cat", "Sub_Cat"]
    official = pd.read_csv(OFFICIAL_CSV, usecols=cols, low_memory=False)
    parsed = pd.to_datetime(official["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce", utc=True)
    start = datetime.fromisoformat(capture["first_packet_utc"].replace("Z", "+00:00")) - TOLERANCE
    end = datetime.fromisoformat(capture["last_packet_utc"].replace("Z", "+00:00")) + TOLERANCE
    in_capture = parsed.between(start - OFFSET_TO_PCAP, end - OFFSET_TO_PCAP, inclusive="both")
    official = official.loc[in_capture].copy().reset_index(drop=True)
    official_index = indexed_flows(official, "Src_IP", "Src_Port", "Dst_IP", "Dst_Port", "Protocol",
                                   "Timestamp", "Flow_Duration", official=True)
    official_rows, official_packets, official_pairs, official_unmatched = links(packets, official_index)

    official_linked = official.iloc[sorted(official_rows)] if official_rows else official.iloc[0:0]
    cfm_linked = cfm.iloc[sorted(cfm_rows)] if cfm_rows else cfm.iloc[0:0]
    official_unlinked = official.loc[~official.index.isin(official_rows)]
    cfm_unlinked = cfm.loc[~cfm.index.isin(cfm_rows)]
    cfm_unlinked_keys = cfm_unlinked["Flow ID"].astype(str) + "|" + pd.to_datetime(
        cfm_unlinked["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce", utc=True
    ).astype(str)
    official_unlinked_times = pd.to_datetime(
        official_unlinked["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce", utc=True
    ) + OFFSET_TO_PCAP
    official_unlinked_keys = official_unlinked["Flow_ID"].astype(str) + "|" + official_unlinked_times.astype(str)
    cfm_key_counts = Counter(cfm_unlinked_keys)
    official_key_counts = Counter(official_unlinked_keys)
    common_unlinked = set(cfm_key_counts) & set(official_key_counts)
    one_to_one_unlinked = [key for key in common_unlinked
                           if cfm_key_counts[key] == 1 and official_key_counts[key] == 1]
    result = {
        "status": "complete",
        "date": "2026-11-03",
        "capture": FILE,
        "capture_sha256": capture["sha256"],
        "creator_filter_subcategory": event["subcategory"],
        "creator_filter_expected_packets": event["expected_attack_packets"],
        "observed_packets_exported_for_match": len(packets),
        "matching_rule": "bidirectional canonical IP/port/protocol 5-tuple plus packet time inside flow start-to-end interval (one-second tolerance for second-resolution flow timestamps)",
        "cfm": {
            "file": str(CFM_CSV.relative_to(ROOT)),
            "flow_timeout_micros": 100000,
            "flows_in_output": int(len(cfm)),
            "attack_packets_linked": cfm_packets,
            "attack_packet_link_fraction": cfm_packets / len(packets) if packets else None,
            "matched_packet_flow_pairs": cfm_pairs,
            "unique_flows_linked": len(cfm_rows),
            "unmatched_packet_summary": summarize_unmatched(cfm_unmatched),
            "unmatched_tuple_packet_recount": unmatched_tuple_packet_recount,
            "flows_without_attack_packet_tuple_time_match": summarize_unlinked_flows(cfm_unlinked, official=False),
        },
        "official_iotid20": {
            "rows_in_capture_interval": int(len(official)),
            "attack_packets_linked": official_packets,
            "attack_packet_link_fraction": official_packets / len(packets) if packets else None,
            "matched_packet_flow_pairs": official_pairs,
            "unique_flows_linked": len(official_rows),
            "linked_category_counts": {str(k): int(v) for k, v in official_linked["Cat"].value_counts(dropna=False).items()},
            "linked_subcategory_counts": {str(k): int(v) for k, v in official_linked["Sub_Cat"].value_counts(dropna=False).items()},
            "linked_binary_label_counts": {str(k): int(v) for k, v in official_linked["Label"].value_counts(dropna=False).items()},
            "unmatched_packet_summary": summarize_unmatched(official_unmatched),
            "flows_without_attack_packet_tuple_time_match": summarize_unlinked_flows(official_unlinked, official=True),
        },
        "unlinked_flow_identity_comparison": {
            "rule": "Flow_ID plus second-resolution start timestamp after the observed +4-hour CSV-to-PCAP alignment",
            "cfm_unlinked_flow_count": int(len(cfm_unlinked)),
            "official_unlinked_flow_count": int(len(official_unlinked)),
            "common_flow_time_keys": int(len(common_unlinked)),
            "one_to_one_unlinked_flow_pairs": int(len(one_to_one_unlinked)),
            "one_to_one_official_label_counts": {
                str(k): int(v) for k, v in official_unlinked.loc[official_unlinked_keys.isin(one_to_one_unlinked), "Sub_Cat"].value_counts(dropna=False).items()
            },
        },
        "feature_agreement": "See the timeout-pilot audit for same-second flow-ID/feature comparisons; this tuple-time audit does not claim that matching tuples imply feature equivalence.",
        "limitations": [
            "The 100 ms CICFlowMeter timeout is a pre-existing pilot setting, not selected to optimize these correspondence counts.",
            "A tuple and interval match is not packet-to-flow provenance proof when tuple reuse creates multiple candidate flows.",
            "The official CSV-to-PCAP four-hour offset is empirical, not a verified timezone configuration.",
            "This is one capture and one event; it does not establish the publisher's label policy or population-level performance.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "capture": FILE,
        "packet_hits": len(packets),
        "cfm_linked_packets": cfm_packets,
        "cfm_linked_flows": len(cfm_rows),
        "official_linked_packets": official_packets,
        "official_linked_flows": len(official_rows),
        "official_subcategories": result["official_iotid20"]["linked_subcategory_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
