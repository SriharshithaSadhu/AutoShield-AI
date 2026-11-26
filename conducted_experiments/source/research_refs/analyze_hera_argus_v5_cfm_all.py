"""Compare row counts from two exporters on the same audited raw PCAPs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERA_CSV = ROOT / "results" / "flow_exporter_pilot" / "argus_v5_42" / "csv"
CFM_DIR = ROOT / "results" / "iot_nid_cfm_v3_capture_batch"
OUT = ROOT / "results" / "flow_exporter_pilot"
OUTPUT_CSV = OUT / "argus_v5_vs_cfm_capture_row_counts.csv"
OUTPUT_JSON = OUT / "argus_v5_vs_cfm_42_capture_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


rows = []
hera_files = sorted(HERA_CSV.glob("*.csv"))
for hera_file in hera_files:
    if not hera_file.name.endswith(".60s.csv"):
        continue
    stem = hera_file.stem.removesuffix(".60s")
    cfm_file = CFM_DIR / f"{stem}.pcap_flowTimeout_120000000.csv"
    if not cfm_file.exists():
        continue
    hera = pd.read_csv(hera_file, usecols=["Proto"], low_memory=False)
    cfm = pd.read_csv(cfm_file, usecols=["Protocol"], low_memory=False)
    hera_proto = hera["Proto"].astype(str).str.lower()
    cfm_proto = pd.to_numeric(cfm["Protocol"], errors="coerce")
    h_tcp_udp = int(hera_proto.isin(["tcp", "udp"]).sum())
    c_tcp_udp = int(cfm_proto.isin([6, 17]).sum())
    rows.append({
        "capture": stem + ".pcap",
        "capture_group": re.sub(r"-\d+-dec$", "", stem),
        "argus_v5_records": len(hera),
        "cfm_v3_records": len(cfm),
        "argus_v5_tcp_udp_records": h_tcp_udp,
        "cfm_v3_tcp_udp_records": c_tcp_udp,
        "argus_relative_count_difference": (len(hera) - len(cfm)) / len(cfm) if len(cfm) else None,
        "argus_csv_sha256": sha256(hera_file),
        "cfm_csv_sha256": sha256(cfm_file),
    })

if not rows:
    raise SystemExit("No paired Argus-v5 and CFM capture CSVs were found.")

frame = pd.DataFrame(rows).sort_values("capture")
frame.to_csv(OUTPUT_CSV, index=False)
summary = {
    "date": "2026-11-05",
    "scope": "paired exporter row-count and transport-protocol count audit on identical PCAPs; not a model evaluation, feature equivalence check, or causal separation of exporter versus settings",
    "hera_source_commit": "9e0190d9dac0cfb36ef0e62027a2561da7c35376",
    "argus_server_version": "5.0.4, source commit 93dd8296cb374a70a07e65537647c3fe17dc5889",
    "argus_client_version": "5.0.4, source commit 62f56ac870ae01c978218bfaa590ebc2b162a4ec",
    "argus_configuration": "-S 60 -A -Z -J; HERA-style default feature projection",
    "cfm_version": "3.0, commit ea19b0783cd6647ee6cd07ad58365b506b947771",
    "cfm_configuration": "flow timeout 120000000 microseconds; activity timeout 5000000 microseconds",
    "matched_capture_count": int(len(frame)),
    "argus_total_records": int(frame["argus_v5_records"].sum()),
    "cfm_total_records": int(frame["cfm_v3_records"].sum()),
    "argus_total_tcp_udp_records": int(frame["argus_v5_tcp_udp_records"].sum()),
    "cfm_total_tcp_udp_records": int(frame["cfm_v3_tcp_udp_records"].sum()),
    "median_capture_relative_record_count_difference": float(frame["argus_relative_count_difference"].median()),
    "captures_argus_higher": int((frame["argus_v5_records"] > frame["cfm_v3_records"]).sum()),
    "captures_argus_lower": int((frame["argus_v5_records"] < frame["cfm_v3_records"]).sum()),
    "captures_equal": int((frame["argus_v5_records"] == frame["cfm_v3_records"]).sum()),
    "max_absolute_capture_relative_difference": float(frame["argus_relative_count_difference"].abs().max()),
    "capture_counts_csv": OUTPUT_CSV.relative_to(ROOT).as_posix(),
    "records": rows,
    "limitations": [
        "Argus -S is its status-report interval; it is not asserted to be semantically equivalent to CFM flow timeout.",
        "Pipelines differ in feature definitions, directionality, handling of non-IP protocols, and version; a row-count difference cannot isolate one causal factor.",
        "No flow-level packet-label coverage, one-to-one row identity, common-feature comparison, or classifier outcomes are included.",
    ],
}
OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2, sort_keys=True))
