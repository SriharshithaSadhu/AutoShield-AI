"""Acquire timestamped CICIDS2017 labeled-flow Parquet files from a pinned mirror."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "cicids2017_temporal"
RAW_DIR = DATA_DIR / "raw"
REPO = "bvsam/cic-ids-2017"
REVISION = "70bac6246d99cf046186a02e1cce6883e2ffe7ea"
API_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"


def fetch_json(url: str) -> object:
    request = Request(url, headers={"User-Agent": "AutoML-IDS-research-audit/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    request = Request(url, headers={"User-Agent": "AutoML-IDS-research-audit/1.0"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tree = fetch_json(API_URL)
    files = [
        item for item in tree
        if item.get("type") == "file"
        and item.get("path", "").startswith("traffic_labels/")
        and item.get("path", "").endswith(".parquet")
    ]
    if len(files) != 8:
        raise RuntimeError(f"Expected eight dated traffic-label Parquet files, found {len(files)}")

    records = []
    for item in sorted(files, key=lambda row: row["path"]):
        relative = item["path"]
        target = RAW_DIR / Path(relative).name
        expected_size = int(item["size"])
        expected_sha256 = item["lfs"]["oid"]
        if not target.exists() or target.stat().st_size != expected_size or sha256_file(target) != expected_sha256:
            download(f"{BASE_URL}/{relative}", target)
        actual_sha256 = sha256_file(target)
        if target.stat().st_size != expected_size or actual_sha256 != expected_sha256:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Integrity verification failed for {relative}")
        records.append({
            "source_path": relative,
            "local_path": str(target.relative_to(ROOT)),
            "size_bytes": expected_size,
            "sha256": actual_sha256,
            "repository_blob_oid": item["oid"],
        })
        print(f"verified {target.name}: {expected_size:,} bytes sha256={actual_sha256}")

    manifest = {
        "dataset": "CICIDS2017 timestamped labeled flows",
        "purpose": "candidate chronological validation; overlap with original project sample must be audited before model evaluation",
        "primary_dataset_page": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "mirror": f"https://huggingface.co/datasets/{REPO}",
        "mirror_revision": REVISION,
        "mirror_readme": f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md",
        "selected_configuration": "traffic_labels; author README states timestamps are normalized to UTC",
        "files": records,
    }
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
