"""Fetch the public Fairdata/Metax metadata manifest for 5G-NIDD.

This audit intentionally downloads no dataset payload. Run from the repository
root with Python 3.10+: python research_refs/audit_5gnidd_dataset_record.py
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATASET_ID = "9d13ef28-2ca7-44b0-9950-225359afac65"
BASE = "https://metax.fairdata.fi/v3"
OUT = Path(__file__).resolve().parents[1] / "data" / "5g_nidd" / "metadata_audit.json"


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "AutoML-MOO research provenance audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


dataset = get_json(f"{BASE}/datasets/{DATASET_ID}")
files_response = get_json(f"{BASE}/datasets/{DATASET_ID}/files")
files = files_response.get("results", [])
OUT.parent.mkdir(parents=True, exist_ok=True)
access_rights = dataset.get("access_rights", {})
licenses = access_rights.get("license", []) if isinstance(access_rights, dict) else []
license_labels = [item.get("pref_label", {}).get("en") for item in licenses]
record = {
    "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "source": f"https://etsin.fairdata.fi/dataset/{DATASET_ID}",
    "metadata_api": f"{BASE}/datasets/{DATASET_ID}",
    "files_api": f"{BASE}/datasets/{DATASET_ID}/files",
    "dataset_id": DATASET_ID,
    "title": dataset.get("title", {}).get("en") if isinstance(dataset.get("title"), dict) else dataset.get("title"),
    "access_rights": access_rights,
    "persistent_identifier": dataset.get("persistent_identifier"),
    "license": license_labels,
    "file_count": files_response.get("count", len(files)),
    "total_file_bytes": sum(int(f.get("size", 0)) for f in files),
    "files": [
        {k: f.get(k) for k in ("id", "filename", "pathname", "size", "checksum", "storage_service", "published")}
        for f in files
    ],
    "payload_downloaded": False,
}
OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUT}")
print(f"Files: {record['file_count']}; bytes: {record['total_file_bytes']:,}; payload downloaded: no")
print("License metadata:", record["license"])
