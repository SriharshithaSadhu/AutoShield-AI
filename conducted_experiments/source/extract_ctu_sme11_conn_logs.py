"""Verify CTU-SME-11 Echo archive and extract only labeled Zeek conn logs.

The raw PCAP members are deliberately not extracted. The output contains the
seven `conn.log.labeled` files, the seven per-day labels.config files, and the
dataset README needed to audit label semantics and temporal coverage.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ctu_sme11"
ARCHIVE = DATA / "CTU-SME-11_Honeypot-Assistant-Amazon-Echo1stGen-1_v1.0.0.tar.bz2"
DEST = DATA / "extracted"
OUT_MANIFEST = DATA / "extracted_manifest.json"
EXPECTED_MD5 = "311cbd100eeddf76c8056919fe35b0cb"
EXPECTED_DATES = {f"2023-02-{day:02d}" for day in range(20, 27)}


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_output_path(member_name: str) -> tuple[str, Path] | None:
    parts = PurePosixPath(member_name).parts
    if ".." in parts or not parts or parts[0] != "CTU-SME-11":
        return None
    # Capture layout: CTU-SME-11/<device>/<date>/<subdir>/<filename>
    if len(parts) < 5:
        return None
    date = parts[2]
    if date not in EXPECTED_DATES:
        return None
    subpath = PurePosixPath(*parts[3:])
    if subpath.name == "conn.log.labeled" and subpath.parts[:1] == ("zeek",):
        rel = Path(date) / "zeek" / "conn.log.labeled"
    elif subpath.name == "labels.config" and subpath.parts[:1] == ("artifacts",):
        rel = Path(date) / "artifacts" / "labels.config"
    elif subpath == PurePosixPath("README.md"):
        rel = Path("README.md")
    else:
        return None
    final = (DEST / rel).resolve()
    if DEST.resolve() not in final.parents:
        raise ValueError(f"Refusing unsafe archive member {member_name}")
    return member_name, final


def main() -> None:
    if digest(ARCHIVE, "md5") != EXPECTED_MD5:
        raise ValueError("Archive MD5 does not match the official Zenodo checksum")
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite existing extraction directory: {DEST}")
    DEST.mkdir(parents=True)
    extracted = []
    with tarfile.open(ARCHIVE, mode="r:bz2") as tar:
        for member in tar:
            selected = safe_output_path(member.name)
            if not selected:
                continue
            name, output = selected
            if not member.isfile():
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                raise IOError(f"Could not read archive member {name}")
            h = hashlib.sha256()
            written = 0
            with src, output.open("wb") as f:
                for block in iter(lambda: src.read(1024 * 1024), b""):
                    f.write(block)
                    h.update(block)
                    written += len(block)
            if written != member.size:
                raise IOError(f"Member size mismatch for {name}: {written} vs {member.size}")
            extracted.append({
                "archive_member": name,
                "output": str(output.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": written,
                "sha256": h.hexdigest(),
            })

    conn_dates = {Path(x["output"]).parts[3] for x in extracted
                  if x["output"].endswith("/zeek/conn.log.labeled")}
    label_dates = {Path(x["output"]).parts[3] for x in extracted
                   if x["output"].endswith("/artifacts/labels.config")}
    if conn_dates != EXPECTED_DATES or label_dates != EXPECTED_DATES:
        raise ValueError(f"Expected seven conn/config files; got conn={conn_dates}, labels={label_dates}")
    manifest = {
        "archive_md5_verified": EXPECTED_MD5,
        "extracted_files": extracted,
        "conn_log_dates": sorted(conn_dates),
        "label_config_dates": sorted(label_dates),
        "pcaps_in_this_extraction_directory": False,
        "separate_pcap_extraction_manifest": "data/ctu_sme11/echo_pcap_extraction_manifest.json",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted {len(extracted)} audit files to {DEST}")
    print(f"Conn logs: {len(conn_dates)} daily files; label configs: {len(label_dates)}")
    print(f"Manifest: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
