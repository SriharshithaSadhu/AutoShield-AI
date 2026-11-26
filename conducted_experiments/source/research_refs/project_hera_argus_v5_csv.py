"""Project HERA's documented default Argus fields to CSV for 42 PCAP outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARGUS_DIR = ROOT / "results" / "flow_exporter_pilot" / "argus_v5_42"
CSV_DIR = ARGUS_DIR / "csv"
MANIFEST = CSV_DIR / "projection_manifest.json"
RA = "/mnt/d/AutoML-MOO/research_refs/argus-clients-v5-linux-src/bin/ra"
LD = "/mnt/d/AutoML-MOO/research_refs/argus-v5-deps/usr/lib/x86_64-linux-gnu:/mnt/d/AutoML-MOO/research_refs/argus-v5-deps/lib/x86_64-linux-gnu"
FIELDS = [
    "rank", "stime", "ltime", "proto", "saddr", "sport", "daddr", "dport",
    "bytes", "sbytes", "dbytes", "pkts", "spkts", "dpkts", "dur",
    "runtime", "idle", "flgs", "tcpopt",
]


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


CSV_DIR.mkdir(parents=True, exist_ok=True)
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "ra_version": "Argus clients 5.0.4 source commit 62f56ac870ae01c978218bfaa590ebc2b162a4ec",
    "hera_projection_equivalent": "ra -n -M noman -u -r <file> -c , -s rank stime ltime proto saddr sport daddr dport + HERA default features",
    "fields": FIELDS,
    "captures": [],
}

for index, argus_path in enumerate(sorted(ARGUS_DIR.glob("*.60s.argus")), start=1):
    output = CSV_DIR / f"{argus_path.stem}.csv"
    field_string = " ".join(FIELDS)
    command = (
        f"LD_LIBRARY_PATH={LD} {RA} -n -M noman -u -r {wsl_path(argus_path)} "
        f"-c , -s {field_string}"
    )
    run = subprocess.run(["wsl.exe", "-e", "bash", "-lc", command], capture_output=True, text=True)
    entry = {"argus_file": argus_path.name, "argus_sha256": sha256(argus_path), "returncode": run.returncode}
    if run.returncode == 0 and run.stdout.strip():
        with output.open("w", newline="", encoding="utf-8") as stream:
            stream.write(run.stdout)
        with output.open("r", newline="", encoding="utf-8") as stream:
            row_count = max(sum(1 for _ in csv.reader(stream)) - 1, 0)
        entry.update({"csv_file": output.name, "rows": row_count, "csv_bytes": output.stat().st_size, "csv_sha256": sha256(output)})
    else:
        entry["stderr"] = run.stderr.strip()[-3000:]
    manifest["captures"].append(entry)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{index}/{len(list(ARGUS_DIR.glob('*.60s.argus')))} {argus_path.name} rows={entry.get('rows', 0)} rc={run.returncode}", flush=True)
    if run.returncode != 0 or "rows" not in entry:
        manifest["stopped_after_error"] = argus_path.name
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(f"Stopped on Argus client projection error for {argus_path.name}.")

print(json.dumps({"captures": len(manifest["captures"]), "rows": sum(item.get("rows", 0) for item in manifest["captures"]), "manifest": str(MANIFEST)}, indent=2))
