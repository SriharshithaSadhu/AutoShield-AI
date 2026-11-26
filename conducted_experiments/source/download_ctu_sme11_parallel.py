"""Range-download the official CTU-SME-11 Echo archive with hash verification.

Only the official Zenodo URL is used. Partial range chunks are resumable; the
archive is installed at its final path only after size and published MD5 match.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import os
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "ctu_sme11"
URL = (
    "https://zenodo.org/records/7958259/files/"
    "CTU-SME-11_Honeypot-Assistant-Amazon-Echo1stGen-1_v1.0.0.tar.bz2"
)
FINAL = OUT_DIR / "CTU-SME-11_Honeypot-Assistant-Amazon-Echo1stGen-1_v1.0.0.tar.bz2"
SIZE = 742_890_363
MD5 = "311cbd100eeddf76c8056919fe35b0cb"
CHUNK = 32 * 1024 * 1024
WORKERS = 8
HEADERS = {"User-Agent": "AutoML-IDS-research/1.0 (Zenodo dataset download)"}


def expected_range(i: int) -> tuple[int, int]:
    start = i * CHUNK
    return start, min(SIZE - 1, start + CHUNK - 1)


def get_part(i: int, parts_dir: Path) -> Path:
    start, end = expected_range(i)
    part = parts_dir / f"part_{i:03d}.bin"
    if part.exists() and part.stat().st_size == end - start + 1:
        return part
    tmp = part.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    last_error = None
    for attempt in range(5):
        try:
            with requests.get(
                URL,
                headers={**HEADERS, "Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                wanted = f"bytes {start}-{end}/{SIZE}"
                if response.status_code != 206 or response.headers.get("Content-Range") != wanted:
                    raise RuntimeError(
                        f"Unexpected range response {response.status_code}: "
                        f"{response.headers.get('Content-Range')} (expected {wanted})"
                    )
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
        except Exception as exc:  # retry a bounded number of times
            last_error = exc
            tmp.unlink(missing_ok=True)
            time.sleep(1 + attempt)
    raise RuntimeError(f"Could not download range {i}: {last_error}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts_dir = OUT_DIR / "download_parts"
    parts_dir.mkdir(exist_ok=True)

    # Preserve the earlier interrupted single-stream bytes for provenance.
    if FINAL.exists() and FINAL.stat().st_size != SIZE:
        partial = OUT_DIR / (FINAL.name + ".single_stream_incomplete")
        if not partial.exists():
            os.replace(FINAL, partial)
        else:
            raise FileExistsError(f"Refusing to overwrite {partial}")

    count = (SIZE + CHUNK - 1) // CHUNK
    parts: dict[int, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(get_part, i, parts_dir): i for i in range(count)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            parts[i] = future.result()
            done = sum(p.stat().st_size for p in parts.values())
            print(f"Completed range {i + 1}/{count}; {done:,}/{SIZE:,} bytes", flush=True)

    assembled = OUT_DIR / (FINAL.name + ".assembled.tmp")
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
