"""Download and checksum small official 5GORAN-IDD summary files from Zenodo.

This gate downloads the network flow CSV (227 MB) and radio telemetry DB (5 MB),
not the large raw-PCAP archives. Run from the repository root.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RECORD_ID = "18923275"
BASE = f"https://zenodo.org/api/records/{RECORD_ID}/files"
OUT = Path(__file__).resolve().parents[1] / "data" / "5g_oran"
TARGETS = {
    "Network_Dataset.csv": {"bytes": 227_184_406, "md5": "68d0a0c964a1134aca5bd1a287170315"},
    "Lower_Layer_Data.db": {"bytes": 5_402_624, "md5": "c3af05f535b12c547a4dbaf858a25458"},
}


def fetch(path: Path, url: str) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "AutoML-MOO 5GORAN provenance audit"})
    size = 0
    digest = hashlib.md5(usedforsecurity=False)
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as sink:
        while block := response.read(1024 * 1024):
            sink.write(block)
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def md5_file(path: Path) -> tuple[int, str]:
    size = 0
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


OUT.mkdir(parents=True, exist_ok=True)
manifest = {
    "record": f"https://zenodo.org/records/{RECORD_ID}",
    "record_id": RECORD_ID,
    "license": "CC BY 4.0 (Zenodo API metadata)",
    "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "files": [],
}
for name, expected in TARGETS.items():
    target = OUT / name
    if target.exists():
        size, digest = md5_file(target)
    else:
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            size, digest = fetch(tmp, f"{BASE}/{name}/content")
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        if size != expected["bytes"] or digest != expected["md5"]:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum/size mismatch for {name}: size={size}, md5={digest}")
        tmp.replace(target)
    if size != expected["bytes"] or digest != expected["md5"]:
        raise RuntimeError(f"Existing file failed verification: {name}")
    manifest["files"].append({"filename": name, "bytes": size, "md5": digest, "verified": True})
    print(f"Verified {name}: {size:,} bytes, MD5 {digest}")

(OUT / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Manifest: {OUT / 'download_manifest.json'}")
