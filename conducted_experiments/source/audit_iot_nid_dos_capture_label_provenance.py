"""Compare six creator DoS packet filters with pinned CFM-V3 and IoTID20 flows."""
from __future__ import annotations

import json
import hashlib
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import audit_iot_nid_tcp_packet_flow_correspondence as correspondence

ROOT = Path(__file__).resolve().parent
FILTER_AUDIT = ROOT / "results" / "iot_nid_attack_filter_audit_20261103.json"
INVENTORY = ROOT / "results" / "iot_nid_raw_capture_inventory_20261102.json"
OFFICIAL_CSV = ROOT / "data" / "provenance" / "IoTID20_official_original.csv"
CFM_DIR = ROOT / "results" / "iot_nid_cfm_v3_capture_batch"
OUTPUT = ROOT / "results" / "iot_nid_dos_capture_label_provenance_20261103.json"
REPORT = ROOT / "results" / "IOTNID_DOS_CAPTURE_LABEL_PROVENANCE.md"
TIMEOUT_US = 120_000_000  # Upstream CFM-V3 default; fixed before inspecting these six captures.
FILES = [f"dos-synflooding-{i}-dec.pcap" for i in range(1, 7)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attack_packets(event: dict) -> list[dict]:
    import subprocess

    pcap = correspondence.PCAPS / event["file"]
    result = subprocess.run([
        "wsl", "-e", "tshark", "-r", str(pcap), "-Y", event["filter"],
        "-T", "fields", "-E", "separator=\t", "-e", "frame.time_epoch",
        "-e", "ip.src", "-e", "tcp.srcport", "-e", "ip.dst", "-e", "tcp.dstport", "-e", "ip.proto",
    ], capture_output=True, text=True, timeout=300, check=False)
    if result.returncode:
        raise RuntimeError(f"tshark failed for {event['file']}: {result.stderr[-2000:]}")
    packets = []
    for row in result.stdout.splitlines():
        fields = row.split("\t")
        if len(fields) < 6 or not fields[0] or not fields[1] or not fields[3]:
            continue
        packets.append({
            "time": datetime.fromtimestamp(float(fields[0]), tz=timezone.utc),
            "key": correspondence.tuple_key(fields[1], fields[2], fields[3], fields[4], fields[5]),
        })
    return packets


def identity_pairs(cfm: pd.DataFrame, official: pd.DataFrame,
                   cfm_unlinked: pd.DataFrame, official_unlinked: pd.DataFrame) -> dict:
    def cfm_keys(rows: pd.DataFrame) -> pd.Series:
        ts = pd.to_datetime(rows["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce", utc=True)
        return rows["Flow ID"].astype(str) + "|" + ts.astype(str)

    def official_keys(rows: pd.DataFrame) -> pd.Series:
        ts = pd.to_datetime(rows["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce", utc=True)
        return rows["Flow_ID"].astype(str) + "|" + (ts + pd.Timedelta(hours=4)).astype(str)

    left, right = cfm_keys(cfm), official_keys(official)
    left_unlinked, right_unlinked = cfm_keys(cfm_unlinked), official_keys(official_unlinked)
    lc, rc = Counter(left), Counter(right)
    lu, ru = Counter(left_unlinked), Counter(right_unlinked)
    common = set(lc) & set(rc)
    common_unlinked = set(lu) & set(ru)
    return {
        "all_common_flow_time_keys": len(common),
        "all_one_to_one_flow_time_pairs": sum(lc[k] == rc[k] == 1 for k in common),
        "unlinked_cfm_rows": len(cfm_unlinked),
        "unlinked_official_rows": len(official_unlinked),
        "common_unlinked_flow_time_keys": len(common_unlinked),
        "one_to_one_unlinked_flow_time_pairs": sum(lu[k] == ru[k] == 1 for k in common_unlinked),
    }


def main() -> None:
    filter_audit = json.loads(FILTER_AUDIT.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    capture_by_file = {row["file"]: row for row in inventory["captures"]}
    events_by_file = {row["file"]: row for row in filter_audit["filter_events"] if row["file"] in FILES}
    official_cols = ["Flow_ID", "Timestamp", "Flow_Duration", "Src_IP", "Src_Port", "Dst_IP", "Dst_Port",
                     "Protocol", "Tot_Fwd_Pkts", "Tot_Bwd_Pkts", "Label", "Cat", "Sub_Cat"]
    official_all = pd.read_csv(OFFICIAL_CSV, usecols=official_cols, low_memory=False)
    official_times = pd.to_datetime(
        official_all["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce", utc=True
    ) + pd.Timedelta(hours=4)

    records = []
    for filename in FILES:
        event = events_by_file[filename]
        capture = capture_by_file[filename]
        packets = attack_packets(event)
        cfm_path = CFM_DIR / f"{filename}_flowTimeout_{TIMEOUT_US}.csv"
        cfm = pd.read_csv(cfm_path, low_memory=False)
        cfm_index = correspondence.indexed_flows(
            cfm, "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Flow Duration", False
        )
        cfm_rows, cfm_linked_packets, cfm_pairs, cfm_unmatched_packets = correspondence.links(packets, cfm_index)

        capture_start = datetime.fromisoformat(capture["first_packet_utc"].replace("Z", "+00:00"))
        capture_end = datetime.fromisoformat(capture["last_packet_utc"].replace("Z", "+00:00"))
        mask = official_times.between(capture_start - timedelta(seconds=1),
                                      capture_end + timedelta(seconds=1), inclusive="both")
        official = official_all.loc[mask].copy().reset_index(drop=True)
        official_index = correspondence.indexed_flows(
            official, "Src_IP", "Src_Port", "Dst_IP", "Dst_Port", "Protocol", "Timestamp", "Flow_Duration", True
        )
        official_rows, official_linked_packets, official_pairs, official_unmatched_packets = correspondence.links(packets, official_index)
        cfm_unlinked = cfm.loc[~cfm.index.isin(cfm_rows)]
        official_unlinked = official.loc[~official.index.isin(official_rows)]

        records.append({
            "capture": filename,
            "capture_sha256": capture["sha256"],
            "generated_cfm_csv": str(cfm_path.relative_to(ROOT).as_posix()),
            "generated_cfm_csv_sha256": sha256(cfm_path),
            "category_subcategory_rule": [event["category"], event["subcategory"]],
            "flow_timeout_micros": TIMEOUT_US,
            "creator_attack_packet_count": event["expected_attack_packets"],
            "tshark_attack_packet_count": len(packets),
            "generated_cfm_flow_rows": int(len(cfm)),
            "official_flows_in_capture_interval": int(len(official)),
            "cfm_attack_packets_matched": cfm_linked_packets,
            "cfm_attack_packet_match_fraction": cfm_linked_packets / len(packets) if packets else None,
            "official_attack_packets_matched": official_linked_packets,
            "official_attack_packet_match_fraction": official_linked_packets / len(packets) if packets else None,
            "unique_cfm_flows_with_packet_match": len(cfm_rows),
            "unique_official_flows_with_packet_match": len(official_rows),
            "cfm_unmatched_attack_packets": correspondence.summarize_unmatched(cfm_unmatched_packets),
            "official_unmatched_attack_packets": correspondence.summarize_unmatched(official_unmatched_packets),
            "cfm_flows_without_attack_packet_match": correspondence.summarize_unlinked_flows(cfm_unlinked, False),
            "official_flows_without_attack_packet_match": correspondence.summarize_unlinked_flows(official_unlinked, True),
            "flow_id_timestamp_comparison": identity_pairs(cfm, official, cfm_unlinked, official_unlinked),
        })
        print(f"{filename}: filtered={len(packets)}/{event['expected_attack_packets']}, "
              f"linked CFM={cfm_linked_packets}, linked official={official_linked_packets}, "
              f"unmatched official flows={len(official_unlinked)}", flush=True)

    output = {
        "status": "complete" if all(x["tshark_attack_packet_count"] == x["creator_attack_packet_count"] for x in records) else "filter_count_mismatch",
        "date": "2026-11-03",
        "purpose": "Bounded packet-to-flow correspondence and label-consistency audit across all six same-type DoS-SYN source captures.",
        "cfm_version": "CICFlowMeter-V3 3.0",
        "cfm_commit": "ea19b0783cd6647ee6cd07ad58365b506b947771",
        "cfm_flow_timeout_microseconds": TIMEOUT_US,
        "cfm_activity_timeout_microseconds": 5_000_000,
        "matching_rule": "bidirectional IP/port/protocol tuple plus selected packet time inside the flow start/end interval; one-second tolerance on second-resolution CSV timestamps",
        "observed_csv_to_pcap_offset_minutes": 240,
        "official_csv_sha256": sha256(OFFICIAL_CSV),
        "records": records,
        "aggregate": {
            "capture_count": len(records),
            "creator_attack_packets": sum(x["creator_attack_packet_count"] for x in records),
            "tshark_attack_packets": sum(x["tshark_attack_packet_count"] for x in records),
            "cfm_attack_packets_matched": sum(x["cfm_attack_packets_matched"] for x in records),
            "official_attack_packets_matched": sum(x["official_attack_packets_matched"] for x in records),
            "cfm_flows_without_attack_packet_match": sum(x["cfm_flows_without_attack_packet_match"]["flow_count"] for x in records),
            "official_flows_without_attack_packet_match": sum(x["official_flows_without_attack_packet_match"]["flow_count"] for x in records),
        },
        "limitations": [
            "This reconstructs one event subtype across six captures only; it does not establish the IoTID20 publisher's label-lifting rule.",
            "A packet-filter miss does not prove a flow is benign if the filter is incomplete or the event attribution is different.",
            "A tuple/time match is not packet-level lineage proof when flow keys repeat or timestamps are rounded.",
            "The four-hour offset is empirical and not a verified timezone configuration.",
            "The 120-second timeout is CICFlowMeter-V3's documented code default in this checkout; the IoTID20 source export configuration remains unknown.",
        ],
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")

    total_packet_hits = output["aggregate"]["creator_attack_packets"]
    total_linked = output["aggregate"]["official_attack_packets_matched"]
    lines = [
        "# IoT-NID DoS-SYN packet-to-flow/label audit (2026-11-03)",
        "",
        "This bounded audit applies the creator's validated SYN-flood display filter to all six DoS captures, then matches selected packets to the pinned CICFlowMeter-V3 3.0 default-timeout output and the official IoTID20 flows using bidirectional 5-tuples and flow-time intervals.",
        "",
        f"All {output['aggregate']['tshark_attack_packets']:,} of {output['aggregate']['creator_attack_packets']:,} expected attack packets were reproduced by TShark. The tuple/time rule links {total_linked:,}/{total_packet_hits:,} selected packets ({total_linked / total_packet_hits:.2%}) to official flows. This rate is not a label accuracy estimate.",
        "",
        "| Capture | Filtered packets | Linked attack packets | Official flows in interval | Official rows not packet-matched | Their published subcategory labels |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in records:
        labels = row["official_flows_without_attack_packet_match"]["subcategory_counts"]
        lines.append(
            f"| {row['capture']} | {row['tshark_attack_packet_count']:,} | "
            f"{row['official_attack_packets_matched']:,} ({row['official_attack_packet_match_fraction']:.2%}) | "
            f"{row['official_flows_in_capture_interval']:,} | "
            f"{row['official_flows_without_attack_packet_match']['flow_count']:,} | `{labels}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"Across these captures, {output['aggregate']['official_flows_without_attack_packet_match']:,} official flow rows had no matching packet selected by the creator's explicit attack filter under the declared tuple/time rule. Their per-capture labels are included in the table. The exact DoS-SYN-5 follow-up found that all 80 such rows were labeled DoS-Synflooding; many represented DNS, HTTPS, or other protocol flows. A recent cross-dataset study explicitly describes IoTID20 as assigning one scenario label to all flows associated with a capture ([Rodriguez et al., 2026](https://doi.org/10.1016/j.iot.2026.102030)). This supports capture-level label semantics; the attack-only packet filter need not select every flow in that capture. These unmatched rows do not establish individual label errors or benign status.",
        "",
        "The pinned CFM-V3 writer omits flow records when `packetCount() <= 1` (`FlowGenerator.java`, `dumpLabeledFlowBasedFeatures`); for SYN-5, 41 of 42 unlinked tuples contain only one TCP packet in the entire PCAP, matching that omission rule. One additional two-packet tuple is not represented and remains unexplained. See the per-capture packet diagnostics in the JSON.",
        "",
        "## Limits / next decision",
        "",
        "Do not re-label the benchmark or report a corrected performance score from this evidence. Next, resolve whether unlinked flows are expected background flows, attack-filter omissions, or flow-aggregation artifacts, using creator filter coverage and packet-level tuple/timing evidence across a predeclared subset. Then compare candidate flow labels with the published CSV and disclose disagreements. The result quantifies capture-level label granularity against narrower packet filters; it is not an IDS-method novelty claim.",
        "",
        "Machine-readable output: `iot_nid_dos_capture_label_provenance_20261103.json`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2))


if __name__ == "__main__":
    main()
