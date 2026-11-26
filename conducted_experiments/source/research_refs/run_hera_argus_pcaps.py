"""Generate HERA/Argus flow records for the already-audited 42 IoT-NID PCAPs."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PCAP_DIR = ROOT / "data" / "iot_nid_raw" / "extracted" / "Packets"
OUT_DIR = ROOT / "results" / "flow_exporter_pilot" / "argus_42"
PILOT_DIR = ROOT / "results" / "flow_exporter_pilot" / "argus"
ARGUS = "/mnt/d/AutoML-MOO/research_refs/argus-local/usr/sbin/argus"
LD = "/mnt/d/AutoML-MOO/research_refs/argus-local/usr/lib/x86_64-linux-gnu:/mnt/d/AutoML-MOO/research_refs/argus-local/lib/x86_64-linux-gnu"
MANIFEST = OUT_DIR / "manifest.json"


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
    "method": "Ubuntu 24.04 Argus 3.0.8.2 via isolated extracted packages; HERA-2.0 Argus flags; -S 60 -A -Z -J",
    "source_pcap_count": len(pcaps),
    "hera_repository_commit": "9e0190d9dac0cfb36ef0e62027a2561da7c35376",
    "argus_server_deb_sha256": sha256(ROOT / "research_refs" / "argus-package-cache" / "argus-server_2%3a3.0.8.2-2.3build1_amd64.deb"),
    "captures": [],
}

for index, pcap in enumerate(pcaps, start=1):
    result = OUT_DIR / f"{pcap.stem}.60s.argus"
    entry = {
        "pcap": pcap.name,
        "pcap_sha256": sha256(pcap),
        "argus_output": result.name,
    }
    if not result.exists() or result.stat().st_size == 0:
        earlier = PILOT_DIR / result.name
        if earlier.exists() and earlier.stat().st_size:
            shutil.copy2(earlier, result)
            entry["copied_prior_matching_run"] = True
    if not result.exists() or result.stat().st_size == 0:
        command = (
            f"LD_LIBRARY_PATH={LD} {ARGUS} -S 60 -A -Z -J "
            f"-r {wsl_path(pcap)} -w {wsl_path(result)}"
        )
        run = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command], capture_output=True, text=True)
        entry["returncode"] = run.returncode
        if run.stdout.strip():
            entry["stdout"] = run.stdout.strip()[-2000:]
        if run.stderr.strip():
            entry["stderr"] = run.stderr.strip()[-4000:]
    else:
        entry["reused_existing_output"] = True
    if result.exists() and result.stat().st_size:
        entry["output_bytes"] = result.stat().st_size
        entry["output_sha256"] = sha256(result)
    manifest["captures"].append(entry)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{index}/{len(pcaps)} {pcap.name} bytes={entry.get('output_bytes', 0)} rc={entry.get('returncode', 'reused')}", flush=True)
    if entry.get("returncode", 0) != 0:
        manifest["stopped_after_parser_error"] = pcap.name
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(f"Stopped after nonzero Argus exit on {pcap.name}; inspect stderr before any retry.")

print(json.dumps({"captures": len(manifest["captures"]), "successful": sum(bool(item.get("output_bytes")) and item.get("returncode", 0) == 0 for item in manifest["captures"]), "manifest": str(MANIFEST)}, indent=2))
