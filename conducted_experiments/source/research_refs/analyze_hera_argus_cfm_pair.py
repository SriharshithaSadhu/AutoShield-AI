"""Descriptive comparison of HERA/Argus and pinned CICFlowMeter on two shared PCAPs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARGUS = ROOT / "results" / "flow_exporter_pilot" / "argus"
CFM = ROOT / "results" / "iot_nid_cfm_v3_pilot"
OUT = ROOT / "results" / "flow_exporter_pilot"

summary = {
    "scope": "descriptive row-count/protocol audit only; not model evaluation or causal attribution",
    "argus": {
        "version": "3.0.8.2 (Ubuntu 24.04 noble package)",
        "hera_repo_commit": "9e0190d9dac0cfb36ef0e62027a2561da7c35376",
        "status_interval_seconds": 60,
        "flags": ["-A", "-Z", "-J"],
        "client": "HERA's ra-style CSV projection (-n -M noman -u); schema/row matching not performed",
    },
    "cicflowmeter": {
        "version": "3.0 pinned project checkout",
        "commit": "ea19b0783cd6647ee6cd07ad58365b506b947771",
        "flow_timeout_microseconds": 120_000_000,
    },
    "captures": {},
}

for capture in ["benign-dec", "dos-synflooding-5-dec"]:
    argus = pd.read_csv(ARGUS / f"{capture}.60s.csv", low_memory=False)
    cfm = pd.read_csv(CFM / f"{capture}.pcap_flowTimeout_120000000.csv", low_memory=False)
    argus_proto = argus["Proto"].astype(str).str.lower()
    cfm_proto = pd.to_numeric(cfm["Protocol"], errors="coerce")
    argus_tcp_udp = int(argus_proto.isin(["tcp", "udp"]).sum())
    cfm_tcp_udp = int(cfm_proto.isin([6, 17]).sum())
    summary["captures"][capture] = {
        "same_source_pcap": True,
        "argus_records": len(argus),
        "cicflowmeter_records": len(cfm),
        "argus_tcp_udp_records": argus_tcp_udp,
        "cicflowmeter_tcp_udp_records": cfm_tcp_udp,
        "argus_other_or_unrecognized_protocol_records": len(argus) - argus_tcp_udp,
        "cicflowmeter_non_tcp_udp_records": len(cfm) - cfm_tcp_udp,
        "argus_relative_row_count_difference_vs_cfm": (len(argus) - len(cfm)) / len(cfm),
    }

path = OUT / "hera_argus_vs_cfm_two_capture_audit.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
