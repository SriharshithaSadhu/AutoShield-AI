"""Acquire the official UQ NF-ToN-IoT-v3 archive and record its checksum.

The API response contains a short-lived signed object URL. Do not print or save
that URL: it is a temporary download credential. The dataset terms permit
academic reuse with attribution and restrict commercial reuse.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATASET_UUID = "343e2e8c-6e6e-4a0c-813d-a46acea1b7f4"
API = "https://api.rdm.uq.edu.au"
EXPECTED_BYTES = 417_476_635  # Confirmed using a one-byte HTTP Range response.
ROOT = Path(__file__).resolve().parents[1] / "data" / "nf_ton_iot_v3"
ARCHIVE = ROOT / "NF-ToN-IoT-v3_UQ.zip"
MANIFEST = ROOT / "download_manifest.json"


def sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


ROOT.mkdir(parents=True, exist_ok=True)
if not ARCHIVE.exists():
    request = urllib.request.Request(
        f"{API}/datasets/{DATASET_UUID}/download",
        headers={"User-Agent": "AutoML-MOO research data provenance audit"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        package = json.load(response)
    if package.get("accessType") != "OPEN" or package.get("canDownload") is not True:
        raise RuntimeError("Official UQ API did not authorize public download")
    signed_url = package["url"]
    tmp = ARCHIVE.with_suffix(ARCHIVE.suffix + ".part")
    try:
        digest = hashlib.sha256()
        size = 0
        download_request = urllib.request.Request(
            signed_url,
            headers={"User-Agent": "AutoML-MOO research data provenance audit"},
        )
        with urllib.request.urlopen(download_request, timeout=180) as response, tmp.open("wb") as sink:
            while block := response.read(4 * 1024 * 1024):
                sink.write(block)
                digest.update(block)
                size += len(block)
        if size != EXPECTED_BYTES:
            raise RuntimeError(f"Archive size mismatch: expected {EXPECTED_BYTES}, received {size}")
        tmp.replace(ARCHIVE)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

sha256, size = sha256_and_size(ARCHIVE)
if size != EXPECTED_BYTES:
    raise RuntimeError(f"Existing archive size mismatch: expected {EXPECTED_BYTES}, received {size}")
record = {
    "dataset": "NF-ToN-IoT-v3",
    "record": "https://rdm.uq.edu.au/files/343e2e8c-6e6e-4a0c-813d-a46acea1b7f4",
    "eSpace_pid": "UQ:44d7c5e",
    "license": "Permitted Re-Use with Commercial Use Restriction; academic research reuse permitted with citation",
    "archive_filename": ARCHIVE.name,
    "archive_bytes": size,
    "archive_sha256": sha256,
    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    "expected_size_source": "HTTP 206 Range bytes 0-0 response: Content-Range total 417476635",
    "verified_against_publisher_sha256": False,
}
MANIFEST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
