"""Download the official CTU-SME-11 Raspberry Pi archive with verification."""
from __future__ import annotations

import concurrent.futures
import hashlib
import os
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ctu_sme11"
NAME = "CTU-SME-11_Honeypot-Minicomputer-RaspberryPi-Gen3-20_v1.0.0.tar.bz2"
URL = f"https://zenodo.org/records/7958259/files/{NAME}"
FINAL = DATA / NAME
SIZE = 4_897_530_463
MD5 = "94285412e422f9983b7431c32acaf5b9"
CHUNK = 32 * 1024 * 1024
WORKERS = 24
HEADERS = {"User-Agent": "AutoML-IDS-research/1.0 (Zenodo dataset download)"}


def get_part(i: int, parts_dir: Path) -> Path:
    start = i * CHUNK
    end = min(SIZE - 1, start + CHUNK - 1)
    part = parts_dir / f"part_{i:03d}.bin"
    if part.exists() and part.stat().st_size == end - start + 1:
        return part
    tmp = part.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    last_error = None
    for attempt in range(5):
        try:
            with requests.get(URL, headers={**HEADERS, "Range": f"bytes={start}-{end}"},
                              stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                wanted = f"bytes {start}-{end}/{SIZE}"
                if response.status_code != 206 or response.headers.get("Content-Range") != wanted:
                    raise RuntimeError(f"Unexpected range response: {response.status_code}, {response.headers.get('Content-Range')}")
                received = 0
                with tmp.open("wb") as f:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            f.write(block)
                            received += len(block)
                if received != end - start + 1:
                    raise IOError(f"Range {i} returned {received} bytes")
            os.replace(tmp, part)
            return part
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            time.sleep(1 + attempt)
    raise RuntimeError(f"Could not download range {i}: {last_error}")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    parts_dir = DATA / "raspberrypi_download_parts"
    parts_dir.mkdir(exist_ok=True)
    count = (SIZE + CHUNK - 1) // CHUNK
    parts: dict[int, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(get_part, i, parts_dir): i for i in range(count)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            parts[i] = future.result()
            done = sum(p.stat().st_size for p in parts.values())
            print(f"Completed Raspberry Pi range {i + 1}/{count}; {done:,}/{SIZE:,} bytes", flush=True)

    assembled = DATA / (NAME + ".assembled.tmp")
    with assembled.open("wb") as output:
        for i in range(count):
            with parts[i].open("rb") as part:
                for block in iter(lambda: part.read(8 * 1024 * 1024), b""):
                    output.write(block)
    if assembled.stat().st_size != SIZE:
        raise IOError(f"Assembled size {assembled.stat().st_size} != {SIZE}")
    digest = hashlib.md5()
    with assembled.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != MD5:
        raise ValueError(f"Official MD5 mismatch: {actual} != {MD5}")
    os.replace(assembled, FINAL)
    for part in parts.values():
        part.unlink(missing_ok=True)
    parts_dir.rmdir()
    print(f"Verified official archive: {FINAL} ({SIZE:,} bytes; MD5 {actual})")


if __name__ == "__main__":
    main()
