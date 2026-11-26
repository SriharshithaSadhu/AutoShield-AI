"""Audit assigning PCAP packets to labeled Zeek flows by tuple/start segments.

For each bidirectional TCP/UDP 5-tuple, sort Zeek records by first-packet time
and assign a PCAP packet to the latest preceding logged start. This avoids
using Zeek duration as a flow-end time: Zeek documents that duration excludes
trailing nonproductive TCP teardown packets. Matching Zeek directional packet
and IP-byte counters is the acceptance diagnostic. This is an offline data
association audit, not a deployed flow tracker or IDS result.
"""
from __future__ import annotations

import json
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import dpkt
import pandas as pd

from audit_ctu_echo_pcap_alignment import packet_key, read_zeek
from audit_ctu_echo_pcap_flow_assignment import canonical_key, endpoint_key


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ctu_sme11"
PCAP_DIR = DATA / "echo_pcaps"
FLOW_DIR = DATA / "extracted"
MANIFEST = DATA / "echo_pcap_extraction_manifest.json"
OUT = DATA / "echo_pcap_tuple_segment_assignment_audit.json"
TIME_TOLERANCE_SECONDS = 0.000010


def get_packet(raw: bytes, ts: float):
    key = packet_key(ts, raw)
    if key is None:
        return None
    source, source_port, destination, destination_port, proto, _start_us = key
    try:
        _, reverse = canonical_key(endpoint_key(source, source_port),
                                   endpoint_key(destination, destination_port), proto)
        frame = dpkt.ethernet.Ethernet(raw)
        ip = frame.data
        ip_bytes = int(getattr(ip, "len", len(bytes(ip))))
        return (proto,
                *canonical_key(endpoint_key(source, source_port),
                               endpoint_key(destination, destination_port), proto)[0][1:],
                float(ts), reverse, ip_bytes)
    except (dpkt.dpkt.UnpackError, AttributeError, ValueError, OSError,
            TypeError, OverflowError):
        return None


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    flow_paths = {p.parts[-3]: p for p in FLOW_DIR.glob("*/zeek/conn.log.labeled")}
    pcap_paths = {Path(item["output"]).parts[-2]: ROOT / item["output"]
                  for item in manifest["files"]}
    if set(flow_paths) != set(pcap_paths):
        raise ValueError("PCAP and Zeek flow day sets differ")

    all_rows = []
    by_day_key = {}
    input_counts = Counter()
    for day, path in sorted(flow_paths.items()):
        frame = read_zeek(path)
        needed = ["ts", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
                  "proto", "label", "duration", "orig_pkts", "resp_pkts",
                  "orig_ip_bytes", "resp_ip_bytes"]
        missing = [col for col in needed if col not in frame.columns]
        if missing:
            raise ValueError(f"{path} missing columns {missing}")
        for _, row in frame.iterrows():
            label = str(row["label"])
            input_counts[(label, "all_rows")] += 1
            if row["proto"] not in ("tcp", "udp"):
                input_counts[(label, "non_tcp_udp")] += 1
                continue
            try:
                start = float(row["ts"])
                orig = endpoint_key(row["id.orig_h"], int(row["id.orig_p"]))
                resp = endpoint_key(row["id.resp_h"], int(row["id.resp_p"]))
                proto = str(row["proto"])
                fields = [row[c] for c in ("orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")]
                if any(pd.isna(x) for x in fields):
                    input_counts[(label, "missing_counters")]+=1
                    continue
                key, reverse = canonical_key(orig, resp, proto)
                history = str(row.get("history", "-"))
                record = {
                    "day": day,
                    "key": key,
                    "label": label,
                    "proto": proto,
                    "start": start,
                    "duration_missing": bool(pd.isna(row["duration"])),
                    "conn_state": str(row.get("conn_state", "-")),
                    "history": history,
                    "orig_endpoint": f"{orig[0]}:{orig[1]}",
                    "resp_endpoint": f"{resp[0]}:{resp[1]}",
                    # The endpoints in conn.log already define the final
                    # originator/responder directions used by its counters.
                    # History '^' records an internal heuristic flip; do not
                    # apply a second flip to the logged endpoint fields.
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
                    "ambiguous": False,
                }
                if record["duration_missing"]:
                    input_counts[(label, "duration_missing")] += 1
                ids = by_day_key.setdefault((day, key), [])
                ids.append(record)
                all_rows.append(record)
                input_counts[(label, "rows_with_counters")]+=1
            except (ValueError, TypeError, OverflowError):
                input_counts[(label, "invalid_tuple_or_counter")]+=1

    indexes = {}
    for day_key, rows in by_day_key.items():
        rows.sort(key=lambda r: r["start"])
        starts = [r["start"] for r in rows]
        duplicate_starts = {s for s, n in Counter(starts).items() if n > 1}
        for row in rows:
            row["ambiguous"] = row["start"] in duplicate_starts
        indexes[day_key] = starts

    total_pcap_packets = total_assigned = total_ambiguous = start_hits = 0
    per_day_packets = {}
    for day, pcap in sorted(pcap_paths.items()):
        print(f"Tuple-segment scan {day}: {pcap.name}", flush=True)
        day_packets = day_assigned = 0
        last_ts = float("-inf")
        with pcap.open("rb") as stream:
            for ts, raw in dpkt.pcap.Reader(stream):
                packet = get_packet(raw, ts)
                if packet is None:
                    continue
                proto, endpoint_a, endpoint_b, packet_ts, reverse, ip_bytes = packet
                day_packets += 1
                if packet_ts < last_ts:
                    pass  # PCAP event timestamps are not strictly monotonic.
                last_ts = packet_ts
                day_key = (day, (proto, endpoint_a, endpoint_b))
                rows = by_day_key.get(day_key)
                if rows is None:
                    continue
                starts = indexes[day_key]
                idx = bisect_right(starts, packet_ts + TIME_TOLERANCE_SECONDS) - 1
                if idx < 0:
                    continue
                row = rows[idx]
                # Multiple Zeek records with the same tuple and start cannot
                # be separated by this association key.
                if row["ambiguous"]:
                    row["_ambiguous_packets"] = row.get("_ambiguous_packets", 0) + 1
                    total_ambiguous += 1
                    continue
                day_assigned += 1
                total_assigned += 1
                if abs(packet_ts - row["start"]) <= TIME_TOLERANCE_SECONDS:
                    row["matched_start"] = True
                    start_hits += 1
                if (reverse == 0) == row["orig_is_canonical_first"]:
                    row["got_orig_pkts"] += 1
                    row["got_orig_bytes"] += ip_bytes
                else:
                    row["got_resp_pkts"] += 1
                    row["got_resp_bytes"] += ip_bytes
        per_day_packets[day] = {"tcp_udp_packets": day_packets,
                                "assigned_to_labeled_tuple_segments": day_assigned}
        total_pcap_packets += day_packets

    def summary(rows):
        exact = [r for r in rows if
                 r["got_orig_pkts"] == r["expected_orig_pkts"] and
                 r["got_resp_pkts"] == r["expected_resp_pkts"] and
                 r["got_orig_bytes"] == r["expected_orig_bytes"] and
                 r["got_resp_bytes"] == r["expected_resp_bytes"]]
        return {
            "rows": len(rows),
            "duration_missing_rows": sum(r["duration_missing"] for r in rows),
            "exact_four_directional_counters": len(exact),
            "exact_counter_fraction": len(exact) / max(1, len(rows)),
            "first_packet_match_rows": sum(r["matched_start"] for r in rows),
            "expected_packets": sum(r["expected_orig_pkts"] + r["expected_resp_pkts"] for r in rows),
            "assigned_packets": sum(r["got_orig_pkts"] + r["got_resp_pkts"] for r in rows),
            "expected_ip_bytes": sum(r["expected_orig_bytes"] + r["expected_resp_bytes"] for r in rows),
            "assigned_ip_bytes": sum(r["got_orig_bytes"] + r["got_resp_bytes"] for r in rows),
            "ambiguous_rows": sum(r["ambiguous"] for r in rows),
            "rows_with_ambiguous_packets": sum(r.get("_ambiguous_packets", 0) > 0 for r in rows),
        }

    labels = sorted({r["label"] for r in all_rows})
    by_label_duration_status = {
        label: {
            str(missing): summary([r for r in all_rows
                                   if r["label"] == label and r["duration_missing"] == missing])
            for missing in (False, True)
        }
        for label in labels
    }
    by_label = {label: summary([r for r in all_rows if r["label"] == label])
                for label in labels}
    by_proto_label = {
        proto: {label: summary([r for r in all_rows if r["proto"] == proto and r["label"] == label])
                for label in labels}
        for proto in sorted({r["proto"] for r in all_rows})
    }
    by_day_label = {
        day: {label: summary([r for r in all_rows if r["day"] == day and r["label"] == label])
              for label in labels}
        for day in sorted(flow_paths)
    }
    mismatches = [r for r in all_rows if
                  r["got_orig_pkts"] != r["expected_orig_pkts"] or
                  r["got_resp_pkts"] != r["expected_resp_pkts"] or
                  r["got_orig_bytes"] != r["expected_orig_bytes"] or
                  r["got_resp_bytes"] != r["expected_resp_bytes"]]
    start_mismatches = [r for r in all_rows if not r["matched_start"]]
    result = {
        "method": "For every bidirectional 5-tuple within a capture day, associate each packet with the latest preceding Zeek flow-start record; direction is normalized against that row's originator. No duration/end grace is used.",
        "source_archive_md5": manifest["archive_md5_verified"],
        "time_tolerance_seconds": TIME_TOLERANCE_SECONDS,
        "tcp_udp_packets_scanned": total_pcap_packets,
        "packets_assigned_to_segments": total_assigned,
        "packets_on_duplicate_tuple_start_segments_not_assigned": total_ambiguous,
        "first_packet_matches": start_hits,
        "rows_with_counters_by_label": by_label,
        "rows_with_counters_by_label_and_duration_missing": by_label_duration_status,
        "rows_with_counters_by_protocol_and_label": by_proto_label,
        "rows_with_counters_by_day_and_label": by_day_label,
        "input_row_coverage": {
            "all_rows_by_label": {label: input_counts[(label, "all_rows")] for label in labels},
            "rows_with_tcp_udp_counters_by_label": {label: input_counts[(label, "rows_with_counters")] for label in labels},
            "duration_missing_by_label": {label: input_counts[(label, "duration_missing")] for label in labels},
            "non_tcp_udp_by_label": {label: input_counts[(label, "non_tcp_udp")] for label in labels},
            "missing_counters_by_label": {label: input_counts[(label, "missing_counters")] for label in labels},
        },
        "per_day_packet_coverage": per_day_packets,
        "counter_mismatch_examples": [{key: r[key] for key in (
            "day", "label", "proto", "start", "orig_endpoint", "resp_endpoint",
            "duration_missing", "conn_state", "history", "expected_orig_pkts",
            "got_orig_pkts", "expected_resp_pkts", "got_resp_pkts",
            "expected_orig_bytes", "got_orig_bytes", "expected_resp_bytes", "got_resp_bytes")}
            for r in mismatches[:50]],
        "first_packet_mismatch_examples": [{key: r[key] for key in (
            "day", "label", "proto", "start", "orig_endpoint", "resp_endpoint",
            "duration_missing", "conn_state", "history", "expected_orig_pkts",
            "got_orig_pkts", "expected_resp_pkts", "got_resp_pkts")}
            for r in start_mismatches[:50]],
        "limitations": [
            "This is an offline association diagnostic using the complete labeled flow-start table; it is not itself a deployable online flow tracker.",
            "Exact counter agreement supports packet membership but does not verify the correctness or temporal granularity of source labels.",
            "Packets from tuples with duplicate Zeek start timestamps are left unassigned as ambiguous.",
            "The resulting labeled flows are one Echo device capture and include only one malicious detailed category; broader claims require more independent captures.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows_with_counters_by_label": by_label,
                      "input_row_coverage": result["input_row_coverage"],
                      "packets_assigned_to_segments": total_assigned,
                      "ambiguous_packets": total_ambiguous}, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
