"""Run the HERA/Argus capture extraction on the 42 audited PCAPs with Argus 5.0.4."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PCAP_DIR = ROOT / "data" / "iot_nid_raw" / "extracted" / "Packets"
OUT_DIR = ROOT / "results" / "flow_exporter_pilot" / "argus_v5_42"
MANIFEST = OUT_DIR / "manifest.json"
ARGUS = "/mnt/d/AutoML-MOO/research_refs/argus-v5-linux-src/bin/argus"
LD = "/mnt/d/AutoML-MOO/research_refs/argus-v5-deps/usr/lib/x86_64-linux-gnu:/mnt/d/AutoML-MOO/research_refs/argus-v5-deps/lib/x86_64-linux-gnu"
COMMIT = "93dd8296cb374a70a07e65537647c3fe17dc5889"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wsl_path(path: Path) -> str:
    value = path.resolve().as_posix()
    if value.startswith("D:/"):
        return "/mnt/d/" + value[3:]
    raise ValueError(f"Expected path on D: drive, got {value}")


OUT_DIR.mkdir(parents=True, exist_ok=True)
pcaps = sorted(PCAP_DIR.glob("*.pcap"))
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "method": "HERA-2.0 Argus flags -S 60 -A -Z -J; isolated Argus server/client v5.0.4 local build; no post-export filter",
    "source_pcap_count": len(pcaps),
    "argus_commit": COMMIT,
    "argus_client_commit": "62f56ac870ae01c978218bfaa590ebc2b162a4ec",
    "release_security_note": "Official v5.0.4 release notes state a security review, 716 fuzz-discovered crash-input regression cases, and a fix for the matching buffer-overflow startup-abort message observed with Ubuntu Argus 3.0.8.2.",
    "captures": [],
}

for index, pcap in enumerate(pcaps, start=1):
    result = OUT_DIR / f"{pcap.stem}.60s.argus"
    command = (
        f"LD_LIBRARY_PATH={LD} {ARGUS} -S 60 -A -Z -J "
        f"-r {wsl_path(pcap)} -w {wsl_path(result)}"
    )
    run = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command], capture_output=True, text=True)
    entry = {
        "pcap": pcap.name,
        "pcap_sha256": sha256(pcap),
        "argus_output": result.name,
        "returncode": run.returncode,
        "output_bytes": result.stat().st_size if result.exists() else 0,
    }
    if run.stdout.strip():
        entry["stdout"] = run.stdout.strip()[-2000:]
    if run.stderr.strip():
        entry["stderr"] = run.stderr.strip()[-4000:]
    if result.exists() and result.stat().st_size:
        entry["output_sha256"] = sha256(result)
    manifest["captures"].append(entry)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{index}/{len(pcaps)} {pcap.name} bytes={entry['output_bytes']} rc={run.returncode}", flush=True)
    if run.returncode != 0 or not entry["output_bytes"]:
        manifest["stopped_after_error"] = pcap.name
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(f"Stopped after Argus 5.0.4 failure on {pcap.name}; inspect stderr before any retry.")

print(json.dumps({"captures": len(manifest["captures"]), "successful": sum(item["returncode"] == 0 and item["output_bytes"] > 0 for item in manifest["captures"]), "manifest": str(MANIFEST)}, indent=2))
