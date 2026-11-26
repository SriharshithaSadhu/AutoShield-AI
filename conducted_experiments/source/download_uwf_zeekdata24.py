"""Acquire the official UWF-ZeekData24 weekly Parquet partitions for audit.

Source URLs and published directory sizes are taken from the official UWF
dataset host. SHA-256 values are computed locally and written into a manifest.
No CSV class-specific subsets are downloaded.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "uwf_zeekdata24" / "parquet"
BASE = "https://datasets.uwf.edu/data/UWF-ZeekData24/parquet"
PARTS = [
    ("2024-02-25 - 2024-03-03", "part-00000-8b838a85-76eb-4896-a0b6-2fc425e828c2-c000.snappy.parquet", 18_779_592),
    ("2024-03-03 - 2024-03-10", "part-00000-0955ed97-8460-41bd-872a-7375a7f0207e-c000.snappy.parquet", 3_733_862),
    ("2024-03-10 - 2024-03-17", "part-00000-071774ae-97f3-4f31-9700-8bfcdf41305a-c000.snappy.parquet", 12_282_299),
    ("2024-03-17 - 2024-03-24", "part-00000-5f556208-a1fc-40a1-9cc2-a4e24c76aeb3-c000.snappy.parquet", 27_802_118),
    ("2024-03-24 - 2024-03-31", "part-00000-ea3a47a3-0973-4d6b-a3a2-8dd441ee7901-c000.snappy.parquet", 7_834_447),
    ("2024-10-27 - 2024-11-03", "part-00000-69700ccb-c1c1-4763-beb7-cd0f1a61c268-c000.snappy.parquet", 33_163_904),
    ("2024-11-03 - 2024-11-10", "part-00000-f078acc1-ab56-40a6-a6e1-99d780645c57-c000.snappy.parquet", 36_635_805),
]


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest = {
        "dataset": "UWF-ZeekData24",
        "source": "official University of West Florida dataset host",
        "source_directory": f"{BASE}/",
        "license": "CC BY (per official UWF dataset page; cite dataset paper)",
        "downloaded_representation": "dated weekly Parquet partitions only; no class-separated CSV subsets",
        "files": [],
    }
    DEST.mkdir(parents=True, exist_ok=True)
    for interval, filename, expected_bytes in PARTS:
        safe_interval = interval.replace(" ", "_")
        path = DEST / f"{safe_interval}__{filename}"
        source_url = f"{BASE}/{urllib.request.quote(interval)}/{filename}"
        if not path.exists() or path.stat().st_size != expected_bytes:
            request = urllib.request.Request(source_url, headers={"User-Agent": "research-dataset-audit/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, path.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(f"Size mismatch for {filename}: {actual_bytes} != {expected_bytes}")
        record = {
            "interval": interval, "filename": path.name, "source_url": source_url,
            "expected_bytes": expected_bytes, "actual_bytes": actual_bytes,
            "sha256": hash_file(path), "status": "downloaded_and_verified_by_size",
        }
        manifest["files"].append(record)
        print(f"{interval}: {actual_bytes:,} bytes sha256={record['sha256']}", flush=True)
    manifest["total_bytes"] = sum(row["actual_bytes"] for row in manifest["files"])
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    out_path = ROOT / "data" / "uwf_zeekdata24" / "download_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)}; total={manifest['total_bytes']:,} bytes", flush=True)


if __name__ == "__main__":
    main()
