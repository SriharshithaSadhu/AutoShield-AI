"""Download the small labeled flow tables from the official TU Wien dataset record."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "tuwien_smart_factory"
BASE = "https://researchdata.tuwien.ac.at/api/records/ghdc6-45k78/files/extractions.zip/container/labeled"
FILES = {
    "a_day1_flows_labeled.csv": 29.7,
    "a_day2_flows_labeled.csv": 27.8,
    "s_day1_flows_labeled.csv": 16.0,
    "s_day2_flows_labeled.csv": 13.7,
    "tf_a_flows_labeled.csv": 63.2,
    "tf_s_flows_labeled.csv": 45.2,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "AutoML-MOO research data audit/1.0"})
    records = []
    for name, displayed_mib in FILES.items():
        target = OUT / name
        digest = hashlib.sha256()
        size = 0
        response = session.get(f"{BASE}/{name}", stream=True, timeout=(30, 180))
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text/csv" not in content_type:
            raise RuntimeError(f"Unexpected content type for {name}: {content_type}")
        with target.open("wb") as f:
            for block in response.iter_content(1024 * 1024):
                if block:
                    f.write(block)
                    digest.update(block)
                    size += len(block)
        response.close()
        if size < displayed_mib * 1_000_000 * 0.95:
            raise RuntimeError(f"Unexpectedly short file: {name} ({size:,} bytes)")
        records.append({
            "filename": name,
            "source": f"{BASE}/{name}",
            "preview_size_mb_approx": displayed_mib,
            "actual_bytes": size,
            "sha256": digest.hexdigest(),
        })
        print(f"{name}: {size:,} bytes sha256={digest.hexdigest()}", flush=True)
    manifest = {
        "dataset": "Malware communication in smart factories: A network traffic data set",
        "record_doi": "10.48436/ghdc6-45k78",
        "license": "CC BY 4.0",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
        "excluded_large_artifacts": ["PCAP archives", "packet-level CSVs", "unlabeled flow tables"],
    }
    (OUT / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'download_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
