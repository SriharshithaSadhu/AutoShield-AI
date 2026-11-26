"""Audit whether CTU-SME-11 Echo PCAPs support causal packet prefixes.

This is a data-adequacy check, not a detector experiment. It counts the first
1/2/5/10 TCP/UDP packets after each exact Zeek flow-start match, limited to a
one-second prefix horizon and stopped at TCP FIN/RST or a later matched flow
start using the same bidirectional tuple. It does not claim full session
reconstruction or ground-truth packet labels.
"""
from __future__ import annotations

import hashlib
import json
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
OUTPUT = ROOT / "results" / "ctu_echo_prefix_feasibility.json"
HORIZON_SECONDS = 1.0
PREFIX_THRESHOLDS = (1, 2, 5, 10)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_key(src: tuple[str, int], dst: tuple[str, int], proto: str):
    if src <= dst:
        return (proto, src, dst), 0
    return (proto, dst, src), 1


def packet_info(ts: float, raw: bytes):
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
            flags = int(transport.flags)
        elif proto_num == dpkt.ip.IP_PROTO_UDP:
            transport, proto = ip.data, "udp"
            flags = 0
        else:
            return None
        src = (socket.inet_ntop(family, ip.src), int(transport.sport))
        dst = (socket.inet_ntop(family, ip.dst), int(transport.dport))
        key, direction = canonical_key(src, dst, proto)
        ip_bytes = int(getattr(ip, "len", len(bytes(ip))))
        return key, direction, float(ts), ip_bytes, flags
    except (dpkt.dpkt.UnpackError, AttributeError, ValueError, OSError,
            TypeError, OverflowError):
        return None


def read_flows(path: Path):
    fields = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#fields\t"):
                fields = line.rstrip("\r\n").split("\t")[1:]
                break
    if not fields:
        raise ValueError(f"Missing Zeek #fields header: {path}")
    return pd.read_csv(path, sep="\t", comment="#", names=fields,
                       na_values=["-", "(empty)"], low_memory=False)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    per_day: dict[str, dict] = {}
    pcap_by_day = {
        Path(item["output"]).parts[-2]: ROOT / item["output"]
        for item in manifest["files"]
    }
    flow_paths = sorted(FLOW_DIR.glob("*/zeek/conn.log.labeled"))
    if set(pcap_by_day) != {p.parts[-3] for p in flow_paths}:
        raise ValueError("Echo PCAP and Zeek flow capture days do not match")

    for path in flow_paths:
        day = path.parts[-3]
        frame = read_flows(path)
        starts: dict[tuple, list[int]] = {}
        start_times_by_key: dict[tuple, list[int]] = {}
        rows = []
        for idx, row in frame.iterrows():
            proto = str(row.get("proto", ""))
            if proto not in ("tcp", "udp") or pd.isna(row.get("ts")):
                continue
            try:
                ts = float(row["ts"])
                orig = (str(row["id.orig_h"]), int(row["id.orig_p"]))
                resp = (str(row["id.resp_h"]), int(row["id.resp_p"]))
                key, _ = canonical_key(orig, resp, proto)
                row_info = {
                    "label": str(row.get("label", "other")),
                    "start": ts,
                    "count": 0,
                    "directional_packets": [0, 0],
                    "directional_bytes": [0, 0],
                    "tcp_closed": False,
                }
                row_id = len(rows)
                rows.append(row_info)
                start_key = (key, int(round(ts * 1_000_000)))
                starts.setdefault(start_key, []).append(row_id)
                start_times_by_key.setdefault(key, []).append(start_key[1])
            except (TypeError, ValueError, KeyError, OverflowError):
                continue

        active: dict[tuple, int] = {}
        start_times_by_key = {
            key: sorted(set(times)) for key, times in start_times_by_key.items()
        }
        ambiguous_start_keys = {k for k, ids in starts.items() if len(ids) != 1}
        matched_start_keys: set[tuple] = set()
        scan_packets = packet_regressions = 0
        last_packet_ts = None
        pcap_path = pcap_by_day[day]
        print(f"Scanning {day}: {pcap_path.name}", flush=True)
        with pcap_path.open("rb") as f:
            reader = dpkt.pcap.Reader(f)
            for ts, raw in reader:
                scan_packets += 1
                if last_packet_ts is not None and ts < last_packet_ts:
                    packet_regressions += 1
                last_packet_ts = ts
                parsed = packet_info(ts, raw)
                if parsed is None:
                    continue
                key, direction, packet_ts, ip_bytes, flags = parsed
                start_key = (key, int(round(packet_ts * 1_000_000)))
                all_starts = start_times_by_key.get(key, [])
                start_pos = bisect_right(all_starts, start_key[1]) - 1
                if start_pos < 0:
                    active.pop(key, None)
                    continue
                latest_expected_start = all_starts[start_pos]
                latest_key = (key, latest_expected_start)
                if latest_key in ambiguous_start_keys:
                    active.pop(key, None)
                    continue
                row_id = active.get(key)
                if row_id is None or rows[row_id]["start_us"] != latest_expected_start:
                    # Do not let an earlier flow absorb packets across a later
                    # Zeek start, including starts whose first packet failed
                    # the exact timestamp join.
                    active.pop(key, None)
                    if start_key != latest_key:
                        continue
                    row_id = starts[latest_key][0]
                    active[key] = row_id
                    rows[row_id]["start_us"] = latest_expected_start
                    matched_start_keys.add(latest_key)
                elif start_key == latest_key:
                    # Repeated timestamped packets may round to the same
                    # microsecond as the connection's first packet.
                    matched_start_keys.add(latest_key)
                if row_id is None:
                    continue
                row = rows[row_id]
                if packet_ts < row["start"] or packet_ts - row["start"] > HORIZON_SECONDS:
                    continue
                row["count"] += 1
                row["directional_packets"][direction] += 1
                row["directional_bytes"][direction] += ip_bytes
                if key[0] == "tcp" and flags & (dpkt.tcp.TH_FIN | dpkt.tcp.TH_RST):
                    row["tcp_closed"] = True
                    active.pop(key, None)

        labels: dict[str, Counter] = {}
        for row in rows:
            label = row["label"] if row["label"] in ("Benign", "Malicious") else "other"
            c = labels.setdefault(label, Counter())
            c["flow_rows"] += 1
            for threshold in PREFIX_THRESHOLDS:
                if row["count"] >= threshold:
                    c[f"flows_with_prefix_{threshold}_within_1s"] += 1
            if row["count"] > 0:
                c["flows_with_prefix_1_within_1s"] += 0  # ensure stable key for zero-count groups
            c["tcp_closed_before_horizon"] += int(row["tcp_closed"])

        day_summary = {
            "zeek_flow_rows": len(frame),
            "eligible_tcp_udp_flow_rows": len(rows),
            "exact_start_matches": len(matched_start_keys),
            "duplicate_ambiguous_start_keys": len(ambiguous_start_keys),
            "pcap_packets_scanned": scan_packets,
            "pcap_timestamp_regressions": packet_regressions,
            "label_conditioned_prefix_counts": {
                label: dict(counts) for label, counts in labels.items()
            },
            "limitation": "Packets are assigned only after an exact Zeek start match and by active bidirectional 5-tuple; active assignment is cleared at every later expected Zeek start, including an unmatched start. The 1-second cap bounds but does not prove Zeek session membership for every prefix packet.",
        }
        prefix_one = sum(
            values.get("flows_with_prefix_1_within_1s", 0)
            for values in day_summary["label_conditioned_prefix_counts"].values()
        )
        if prefix_one != day_summary["exact_start_matches"]:
            raise AssertionError(
                f"{day}: prefix-1 count {prefix_one} != exact starts "
                f"{day_summary['exact_start_matches']}"
            )
        per_day[day] = day_summary
        print(day, day_summary["eligible_tcp_udp_flow_rows"],
              day_summary["exact_start_matches"],
              {k: v.get("flows_with_prefix_5_within_1s", 0)
               for k, v in day_summary["label_conditioned_prefix_counts"].items()},
              flush=True)

    totals: dict[str, Counter] = {}
    for report in per_day.values():
        for label, values in report["label_conditioned_prefix_counts"].items():
            totals.setdefault(label, Counter()).update(values)
    total_exact_starts = sum(r["exact_start_matches"] for r in per_day.values())
    total_prefix_one = sum(
        values.get("flows_with_prefix_1_within_1s", 0)
        for values in totals.values()
    )
    if total_prefix_one != total_exact_starts:
        raise AssertionError(
            f"all days: prefix-1 count {total_prefix_one} != exact starts "
            f"{total_exact_starts}"
        )
    result = {
        "purpose": "Feasibility audit only; no model trained and no detection-performance claim.",
        "source_doi": "10.5281/zenodo.7958259",
        "source_archive_md5": manifest["archive_md5_verified"],
        "source_archive_size_bytes": manifest.get("archive_size_bytes"),
        "input_sha256": {
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in [MANIFEST, *flow_paths, *pcap_by_day.values()]
        },
        "prefix_horizon_seconds": HORIZON_SECONDS,
        "prefix_packet_thresholds": list(PREFIX_THRESHOLDS),
        "packet_assignment": "Exact per-flow first-packet tuple+microsecond timestamp match; later packets assigned by the active bidirectional TCP/UDP 5-tuple until every later expected Zeek start (matched or not), TCP FIN/RST, or one-second horizon.",
        "label_conditioned_total_prefix_counts": {
            label: dict(values) for label, values in totals.items()
        },
        "per_day": per_day,
        "limitations": [
            "This audit measures prefix availability, not classifier performance.",
            "A one-second prefix cap is a feasibility convention, not the Zeek session timeout; same-tuple UDP reuse within one second could contaminate a count.",
            "The capture is one Amazon Echo device over seven days and one dominant malicious discovery label; it is a bounded external case study, not broad network confirmation.",
            "Packet capture order is used as arrival order; packet timestamp regressions are reported per day.",
            "Prefix features must be computed only from packets observed up to that prefix; final Zeek counters, duration, conn_state, and labels are prohibited as predictive inputs.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
