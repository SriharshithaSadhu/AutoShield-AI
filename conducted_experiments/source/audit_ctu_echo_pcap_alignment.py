"""Test whether Echo conn.log flows align to packet-capture first packets."""
from __future__ import annotations

import json
import socket
from pathlib import Path

import dpkt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ctu_sme11"
PCAP_DIR = DATA / "echo_pcaps"
FLOW_DIR = DATA / "extracted"
MANIFEST = DATA / "echo_pcap_extraction_manifest.json"
OUT = DATA / "echo_pcap_alignment_audit.json"


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


def packet_key(ts: float, raw: bytes):
    try:
        frame = dpkt.ethernet.Ethernet(raw)
        network = frame.data
        if isinstance(network, dpkt.ip.IP):
            family, protocol = socket.AF_INET, network.p
        elif isinstance(network, dpkt.ip6.IP6):
            family, protocol = socket.AF_INET6, network.nxt
        else:
            return None
        if protocol == dpkt.ip.IP_PROTO_TCP:
            transport = network.data
            name = "tcp"
        elif protocol == dpkt.ip.IP_PROTO_UDP:
            transport = network.data
            name = "udp"
        else:
            return None
        source = socket.inet_ntop(family, network.src)
        destination = socket.inet_ntop(family, network.dst)
        return (source, int(transport.sport), destination, int(transport.dport),
                name, int(round(ts * 1_000_000)))
    except (dpkt.dpkt.UnpackError, AttributeError, ValueError, OSError, TypeError):
        return None


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    flows_by_day = {}
    for path in sorted(FLOW_DIR.glob("*/zeek/conn.log.labeled")):
        day = path.parts[-3]
        frame = read_zeek(path)
        frame["ts_num"] = pd.to_numeric(frame["ts"], errors="coerce")
        expected: dict[tuple, list[str]] = {}
        for _, row in frame.iterrows():
            if row["proto"] not in ("tcp", "udp") or pd.isna(row["ts_num"]):
                continue
            try:
                key = (str(row["id.orig_h"]), int(row["id.orig_p"]),
                       str(row["id.resp_h"]), int(row["id.resp_p"]),
                       str(row["proto"]), int(round(float(row["ts_num"]) * 1_000_000)))
                expected.setdefault(key, []).append(str(row["label"]))
            except (TypeError, ValueError, OverflowError):
                continue
        flows_by_day[day] = {"frame_rows": len(frame), "expected_tcp_udp_starts": expected,
                             "matched_starts": set(), "packet_count": 0,
                             "unique_tcp_udp_packets": 0}

    pcap_by_day = {Path(item["output"]).parts[-2]: Path(ROOT / item["output"])
                   for item in manifest["files"]}
    if set(pcap_by_day) != set(flows_by_day):
        raise ValueError("PCAP and Zeek flow days do not align")
    for day, path in sorted(pcap_by_day.items()):
        report = flows_by_day[day]
        print(f"Scanning {day}: {path.name}", flush=True)
        with path.open("rb") as src:
            reader = dpkt.pcap.Reader(src)
            for ts, raw in reader:
                report["packet_count"] += 1
                key = packet_key(ts, raw)
                if key is None:
                    continue
                report["unique_tcp_udp_packets"] += 1
                if key in report["expected_tcp_udp_starts"]:
                    report["matched_starts"].add(key)

    per_day = []
    total_expected = total_matched = total_packets = 0
    for day, report in sorted(flows_by_day.items()):
        expected_count = len(report["expected_tcp_udp_starts"])
        matched_count = len(report["matched_starts"])
        label_rows = {"Benign": {"expected": 0, "matched": 0},
                      "Malicious": {"expected": 0, "matched": 0},
                      "other": {"expected": 0, "matched": 0}}
        for key, labels in report["expected_tcp_udp_starts"].items():
            for label in labels:
                group = label if label in ("Benign", "Malicious") else "other"
                label_rows[group]["expected"] += 1
                if key in report["matched_starts"]:
                    label_rows[group]["matched"] += 1
        for counts in label_rows.values():
            counts["unmatched"] = counts["expected"] - counts["matched"]
            counts["match_fraction"] = counts["matched"] / max(1, counts["expected"])
        per_day.append({
            "capture_day": day,
            "conn_log_rows": report["frame_rows"],
            "expected_distinct_tcp_udp_starts": expected_count,
            "matched_exact_tuple_and_microsecond_start": matched_count,
            "unmatched": expected_count - matched_count,
            "match_fraction": matched_count / max(1, expected_count),
            "label_conditioned_flow_row_match": label_rows,
            "pcap_packets": report["packet_count"],
            "tcp_udp_packets": report["unique_tcp_udp_packets"],
        })
        total_expected += expected_count
        total_matched += matched_count
        total_packets += report["packet_count"]
    result = {
        "method": "Match Zeek tcp/udp flow records to at least one packet with identical directional 5-tuple and first-packet timestamp rounded to microseconds; no model training.",
        "source_archive_md5": manifest["archive_md5_verified"],
        "total_pcap_packets_scanned": total_packets,
        "distinct_expected_tcp_udp_starts": total_expected,
        "matched_flow_starts": total_matched,
        "overall_match_fraction": total_matched / max(1, total_expected),
        "flow_row_match_by_label": {label: {"expected": 0, "matched": 0}
                                    for label in ("Benign", "Malicious", "other")},
        "per_day": per_day,
        "limitations": [
            "This checks first-packet identity/timestamps, not full Zeek state or flow reconstruction.",
            "Flows without a matching first packet may be unsupported protocols, truncated capture edges, parsing issues, or timestamp precision mismatches; investigate before using PCAP-derived examples.",
        ],
    }
    for day in per_day:
        for label, counts in day["label_conditioned_flow_row_match"].items():
            result["flow_row_match_by_label"][label]["expected"] += counts["expected"]
            result["flow_row_match_by_label"][label]["matched"] += counts["matched"]
    for counts in result["flow_row_match_by_label"].values():
        counts["unmatched"] = counts["expected"] - counts["matched"]
        counts["match_fraction"] = counts["matched"] / max(1, counts["expected"])
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["total_pcap_packets_scanned",
                  "distinct_expected_tcp_udp_starts", "matched_flow_starts",
                  "overall_match_fraction", "flow_row_match_by_label"]}, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
