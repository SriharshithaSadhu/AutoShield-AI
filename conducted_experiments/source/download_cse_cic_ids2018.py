"""Fetch the official CSE-CIC-IDS2018 daily flow CSVs from the public AWS bucket.

This acquisition is scoped to the chronological drift/adaptation experiment.
The file names preserve the dataset's collection-day sequence; downstream
replay must sort rows by Timestamp and must not use timestamps as features.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests


BUCKET = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com"
PREFIX = "Processed Traffic Data for ML Algorithms/"
DESTINATION = Path(__file__).parent / "data" / "cse_cic_ids2018" / "raw"
MANIFEST = DESTINATION.parent / "manifest.json"
CSV_SUFFIX = "_TrafficForML_CICFlowMeter.csv"


def list_objects(session: requests.Session) -> list[dict[str, str | int]]:
    response = session.get(
        BUCKET,
        params={"list-type": "2", "prefix": PREFIX, "max-keys": "1000"},
        timeout=60,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    objects = []
    for item in root.findall("s3:Contents", ns):
        key = item.findtext("s3:Key", namespaces=ns) or ""
        if key.endswith(CSV_SUFFIX):
            objects.append({
                "key": key,
                "name": key.rsplit("/", 1)[-1],
                "size_bytes": int(item.findtext("s3:Size", namespaces=ns) or 0),
            })
    return sorted(objects, key=lambda item: str(item["name"]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(session: requests.Session, obj: dict[str, str | int]) -> dict[str, object]:
    name = str(obj["name"])
    size = int(obj["size_bytes"])
    output = DESTINATION / name
    if output.exists() and output.stat().st_size == size:
        return {**obj, "sha256": sha256(output), "status": "already_present"}

    partial = output.with_suffix(output.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    url = f"{BUCKET}/{quote(str(obj['key']), safe='/')}"
    for attempt in range(4):
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                if offset and response.status_code != 206:
                    partial.unlink(missing_ok=True)
                    offset = 0
                    headers = {}
                    continue
                response.raise_for_status()
                mode = "ab" if offset else "wb"
                received = offset
                last_report = received
                with partial.open(mode) as out:
                    for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not block:
                            continue
                        out.write(block)
                        received += len(block)
                        if received - last_report >= 128 * 1024 * 1024:
                            print(f"{name}: {received / size:.0%} ({received:,}/{size:,} bytes)", flush=True)
                            last_report = received
            if partial.stat().st_size != size:
                raise IOError(f"Expected {size} bytes, got {partial.stat().st_size}")
            partial.replace(output)
            return {**obj, "sha256": sha256(output), "status": "downloaded"}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
    raise RuntimeError(f"Failed to download {name}")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        listed_objects = list_objects(session)
        objects = listed_objects
        if not objects:
            raise RuntimeError("No official daily CSE-CIC-IDS2018 flow CSVs found")
        total_bytes = sum(int(obj["size_bytes"]) for obj in objects)
        print(f"Downloading all {len(objects)} official daily flow files ({total_bytes / 1e9:.2f} GB)", flush=True)
        entries = [download_one(session, obj) for obj in objects]
    MANIFEST.write_text(json.dumps({
        "source": "Official CSE-CIC-IDS2018 AWS bucket",
        "source_prefix": f"s3://cse-cic-ids2018/{PREFIX}",
        "official_dataset_page": "https://www.unb.ca/cic/datasets/ids-2018.html",
        "purpose": "Chronological drift/adaptation experiment only; no full AutoML search",
        "excluded_objects": [],
        "files": entries,
    }, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
