"""Validate packet-to-Zeek-flow assignment against Zeek interval counters.

This is a data-quality gate, not a detection experiment. Because CTU Echo has
label-dependent missing durations, this audit only checks rows with finite
duration and available Zeek counters, and reports the excluded rows by label.
"""
from __future__ import annotations

import json
import argparse
import socket
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import dpkt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ctu_sme11"
PCAP_DIR = DATA / "echo_pcaps"
FLOW_DIR = DATA / "extracted"
MANIFEST = DATA / "echo_pcap_extraction_manifest.json"
TIME_TOLERANCE_SECONDS = 0.000010


def read_zeek(path: Path) -> pd.DataFrame:
    fields = None
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if line.startswith("#fields\t"):
                fields = line.rstrip("\r\n").split("\t")[1:]
                break
    if not fields:
        raise ValueError(f"Missing #fields in {path}")
    return pd.read_csv(path, sep="\t", comment="#", names=fields,
                       na_values=["-", "(empty)"], low_memory=False)


def endpoint_key(address: str, port: int) -> tuple[str, int]:
    return (str(address), int(port))


def canonical_key(orig: tuple[str, int], resp: tuple[str, int], proto: str):
    if orig <= resp:
        return (proto, orig, resp), 0
    return (proto, resp, orig), 1


def packet_fields(ts: float, raw: bytes):
    try:
        frame = dpkt.ethernet.Ethernet(raw)
        ip = frame.data
        if isinstance(ip, dpkt.ip.IP):
            family, proto_num = socket.AF_INET, ip.p
        elif isinstance(ip, dpkt.ip6.IP6):
            family, proto_num = socket.AF_INET6, ip.nxt
        else:
            return None
        if proto_num == dpkt.ip.IP_PROTO_TCP:
            transport, proto = ip.data, "tcp"
        elif proto_num == dpkt.ip.IP_PROTO_UDP:
            transport, proto = ip.data, "udp"
        else:
            return None
        src = endpoint_key(socket.inet_ntop(family, ip.src), transport.sport)
        dst = endpoint_key(socket.inet_ntop(family, ip.dst), transport.dport)
        key, reverse = canonical_key(src, dst, proto)
        ip_bytes = int(getattr(ip, "len", len(bytes(ip))))
        return key, reverse, float(ts), ip_bytes
    except (dpkt.dpkt.UnpackError, AttributeError, ValueError, OSError,
            TypeError, OverflowError):
        return None


def load_flows(path: Path, day: str, end_grace_seconds: float):
    frame = read_zeek(path)
    required = ["ts", "duration", "id.orig_h", "id.orig_p", "id.resp_h",
                "id.resp_p", "proto", "label", "orig_pkts", "resp_pkts",
                "orig_ip_bytes", "resp_ip_bytes"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing fields: {missing}")
    by_key: dict[tuple, list[dict]] = {}
    label_counts = Counter()
    usable = 0
    for index, row in frame.iterrows():
        label = str(row["label"])
        label_counts[(label, "all")] += 1
        if row["proto"] not in ("tcp", "udp"):
            label_counts[(label, "non_tcp_udp")] += 1
            continue
        vals = [row[c] for c in required if c not in ("label", "proto")]
        if any(pd.isna(value) for value in vals) or pd.isna(row["duration"]):
            label_counts[(label, "missing_duration_or_counter")] += 1
            continue
        try:
            ts = float(row["ts"])
            duration = float(row["duration"])
            orig = endpoint_key(row["id.orig_h"], int(row["id.orig_p"]))
            resp = endpoint_key(row["id.resp_h"], int(row["id.resp_p"]))
            proto = str(row["proto"])
            key, reverse = canonical_key(orig, resp, proto)
            counts = {
                "day": day,
                "label": label,
                "proto": proto,
                "service": str(row.get("service", "-")),
                "conn_state": str(row.get("conn_state", "-")),
                "duration": duration,
                "start": ts,
                "end": ts + duration + end_grace_seconds,
                "orig_is_canonical_first": reverse == 0,
                "expected_orig_pkts": int(row["orig_pkts"]),
                "expected_resp_pkts": int(row["resp_pkts"]),
                "expected_orig_bytes": int(row["orig_ip_bytes"]),
                "expected_resp_bytes": int(row["resp_ip_bytes"]),
                "got_orig_pkts": 0,
                "got_resp_pkts": 0,
                "got_orig_bytes": 0,
                "got_resp_bytes": 0,
                "matched_start": False,
                "ambiguous_packets": 0,
            }
            by_key.setdefault(key, []).append(counts)
            label_counts[(label, "duration_counter_usable")] += 1
            usable += 1
        except (TypeError, ValueError, OverflowError):
            label_counts[(label, "invalid_duration_or_counter")] += 1
    for rows in by_key.values():
        rows.sort(key=lambda item: (item["start"], item["end"]))
    indexes = {}
    for key, rows in by_key.items():
        starts = [item["start"] for item in rows]
        prefix_max_end = []
        maximum = float("-inf")
        for item in rows:
            maximum = max(maximum, item["end"])
            prefix_max_end.append(maximum)
        indexes[key] = (starts, prefix_max_end)
    return by_key, indexes, label_counts, usable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-grace-seconds", type=float, default=0.1,
                        help="Sensitivity window added after Zeek duration for trailing TCP packets")
    args = parser.parse_args()
    if args.end_grace_seconds < 0:
        raise ValueError("end-grace-seconds must be nonnegative")
    grace_tag = f"{args.end_grace_seconds:g}".replace(".", "p")
    out_path = DATA / f"echo_pcap_flow_assignment_{grace_tag}s_grace_audit.json"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    flow_paths = {path.parts[-3]: path for path in FLOW_DIR.glob("*/zeek/conn.log.labeled")}
    pcap_paths = {Path(item["output"]).parts[-2]: ROOT / item["output"]
                  for item in manifest["files"]}
    if set(flow_paths) != set(pcap_paths):
        raise ValueError("PCAP and Zeek flow day sets differ")

    state = {}
    per_label_input = Counter()
    total_packets = total_matched_packets = total_ambiguous_packets = 0
    timestamp_regressions = 0
    for day in sorted(flow_paths):
        by_key, indexes, label_counts, usable = load_flows(flow_paths[day], day,
                                                            args.end_grace_seconds)
        for key, value in label_counts.items():
            per_label_input[key] += value
        per_file_packets = 0
        last_ts = float("-inf")
        print(f"Assigning packets for {day}: {pcap_paths[day].name}", flush=True)
        with pcap_paths[day].open("rb") as src:
            reader = dpkt.pcap.Reader(src)
            for packet_ts, raw in reader:
                fields = packet_fields(packet_ts, raw)
                if fields is None:
                    continue
                key, reverse, packet_ts, ip_bytes = fields
                per_file_packets += 1
                if packet_ts < last_ts:
                    timestamp_regressions += 1
                last_ts = packet_ts
                rows = by_key.get(key)
                if rows is None:
                    continue
                starts, prefix_max_end = indexes[key]
                hi = bisect_right(starts, packet_ts + TIME_TOLERANCE_SECONDS)
                candidates = []
                idx = hi - 1
                while idx >= 0 and prefix_max_end[idx] >= packet_ts - TIME_TOLERANCE_SECONDS:
                    item = rows[idx]
                    if item["start"] - TIME_TOLERANCE_SECONDS <= packet_ts <= item["end"] + TIME_TOLERANCE_SECONDS:
                        candidates.append(item)
                    idx -= 1
                if len(candidates) > 1:
                    total_ambiguous_packets += 1
                    for item in candidates:
                        item["ambiguous_packets"] += 1
                    continue
                if not candidates:
                    continue
                item = candidates[0]
                total_matched_packets += 1
                if (reverse == 0) == item["orig_is_canonical_first"]:
                    item["got_orig_pkts"] += 1
                    item["got_orig_bytes"] += ip_bytes
                else:
                    item["got_resp_pkts"] += 1
                    item["got_resp_bytes"] += ip_bytes
                if abs(packet_ts - item["start"]) <= TIME_TOLERANCE_SECONDS:
                    item["matched_start"] = True
        total_packets += per_file_packets
        state[day] = [row for rows in by_key.values() for row in rows]
        print(f"  TCP/UDP packets={per_file_packets:,}; interval-assigned={total_matched_packets:,}", flush=True)

    records = [row for day_rows in state.values() for row in day_rows]
    labels = sorted({row["label"] for row in records})

    def summarize(subset):
        exact = [row for row in subset
                 if row["got_orig_pkts"] == row["expected_orig_pkts"]
                 and row["got_resp_pkts"] == row["expected_resp_pkts"]
                 and row["got_orig_bytes"] == row["expected_orig_bytes"]
                 and row["got_resp_bytes"] == row["expected_resp_bytes"]]
        return {
            "flow_rows": len(subset),
            "exact_all_directional_pkt_and_byte_counts": len(exact),
            "exact_fraction": len(exact) / max(1, len(subset)),
            "first_packet_matched": sum(row["matched_start"] for row in subset),
            "expected_packets": sum(row["expected_orig_pkts"] + row["expected_resp_pkts"] for row in subset),
            "assigned_packets": sum(row["got_orig_pkts"] + row["got_resp_pkts"] for row in subset),
            "expected_ip_bytes": sum(row["expected_orig_bytes"] + row["expected_resp_bytes"] for row in subset),
            "assigned_ip_bytes": sum(row["got_orig_bytes"] + row["got_resp_bytes"] for row in subset),
            "rows_overassigned_packets": sum(
                row["got_orig_pkts"] + row["got_resp_pkts"] > row["expected_orig_pkts"] + row["expected_resp_pkts"]
                for row in subset),
            "rows_underassigned_packets": sum(
                row["got_orig_pkts"] + row["got_resp_pkts"] < row["expected_orig_pkts"] + row["expected_resp_pkts"]
                for row in subset),
            "duration_median_seconds": float(pd.Series([row["duration"] for row in subset]).median()) if subset else None,
        }

    label_result = {}
    for label in labels:
        subset = [row for row in records if row["label"] == label]
        packet_exact = [row for row in subset
                        if row["got_orig_pkts"] == row["expected_orig_pkts"]
                        and row["got_resp_pkts"] == row["expected_resp_pkts"]]
        byte_exact = [row for row in subset
                      if row["got_orig_bytes"] == row["expected_orig_bytes"]
                      and row["got_resp_bytes"] == row["expected_resp_bytes"]]
        both_exact = [row for row in subset
                      if row["got_orig_pkts"] == row["expected_orig_pkts"]
                      and row["got_resp_pkts"] == row["expected_resp_pkts"]
                      and row["got_orig_bytes"] == row["expected_orig_bytes"]
                      and row["got_resp_bytes"] == row["expected_resp_bytes"]]
        label_result[label] = {
            "usable_duration_and_counters": len(subset),
            "all_packet_counts_exact": len(packet_exact),
            "packet_count_exact_fraction": len(packet_exact) / max(1, len(subset)),
            "all_directional_ip_bytes_exact": len(byte_exact),
            "ip_bytes_exact_fraction": len(byte_exact) / max(1, len(subset)),
            "both_packet_and_byte_counters_exact": len(both_exact),
            "both_exact_fraction": len(both_exact) / max(1, len(subset)),
            "matched_first_packet": sum(row["matched_start"] for row in subset),
            "ambiguous_flow_rows": sum(row["ambiguous_packets"] > 0 for row in subset),
            "missing_duration_or_counter_or_non_tcp_udp": int(
                sum(per_label_input[(label, key)] for key in
                    ("missing_duration_or_counter", "invalid_duration_or_counter", "non_tcp_udp"))),
            "all_rows_in_source_logs": int(per_label_input[(label, "all")]),
        }
    per_day_label = {}
    for day in sorted(state):
        per_day_label[day] = {
            label: summarize([row for row in state[day] if row["label"] == label])
            for label in labels
        }
    per_proto_label = {}
    for proto in sorted({row["proto"] for row in records}):
        per_proto_label[proto] = {
            label: summarize([row for row in records
                              if row["proto"] == proto and row["label"] == label])
            for label in labels
        }
    per_conn_state_label = {}
    for conn_state in sorted({row["conn_state"] for row in records}):
        per_conn_state_label[conn_state] = {
            label: summarize([row for row in records
                              if row["conn_state"] == conn_state and row["label"] == label])
            for label in labels
        }
    mismatches = [row for row in records
                  if row["got_orig_pkts"] != row["expected_orig_pkts"]
                  or row["got_resp_pkts"] != row["expected_resp_pkts"]
                  or row["got_orig_bytes"] != row["expected_orig_bytes"]
                  or row["got_resp_bytes"] != row["expected_resp_bytes"]]
    result = {
        "purpose": "Data-quality gate; validate whether PCAP packets assigned by exact bidirectional tuple and Zeek [ts, ts+duration+grace] intervals reproduce Zeek directional packet and IP-byte counters. Grace is a sensitivity parameter, not a claimed protocol end rule.",
        "source_archive_md5": manifest["archive_md5_verified"],
        "time_tolerance_seconds": TIME_TOLERANCE_SECONDS,
        "end_grace_seconds_after_zeek_duration": args.end_grace_seconds,
        "pcap_tcp_udp_packets_scanned": total_packets,
        "packet_timestamp_regressions": timestamp_regressions,
        "interval_assigned_unambiguous_packets": total_matched_packets,
        "ambiguous_packets_not_assigned": total_ambiguous_packets,
        "label_conditioned_complete_duration_results": label_result,
        "per_day_and_label_interval_assignment": per_day_label,
        "per_protocol_and_label_interval_assignment": per_proto_label,
        "per_conn_state_and_label_interval_assignment": per_conn_state_label,
        "mismatch_examples": [{key: row[key] for key in (
            "day", "label", "proto", "service", "conn_state", "duration",
            "expected_orig_pkts", "got_orig_pkts", "expected_resp_pkts", "got_resp_pkts",
            "expected_orig_bytes", "got_orig_bytes", "expected_resp_bytes", "got_resp_bytes",
            "matched_start") } for row in mismatches[:50]],
        "limitations": [
            "Only flows with finite duration and all required counters are eligible; missingness is label-associated and excludes a biased part of Echo malicious traffic.",
            "Zeek interval counters validate complete-flow membership, not packet-prefix model validity or the exact per-packet availability of labeled decisions.",
            "Any flow with an overlapping same-tuple interval is marked ambiguous at packet level and is not assigned those packets.",
        ],
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
