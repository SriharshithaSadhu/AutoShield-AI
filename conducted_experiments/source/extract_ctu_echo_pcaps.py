"""Extract the seven Echo PCAPs from the checksum-verified official archive."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ctu_sme11"
ARCHIVE = DATA / "CTU-SME-11_Honeypot-Assistant-Amazon-Echo1stGen-1_v1.0.0.tar.bz2"
DEST = DATA / "echo_pcaps"
MANIFEST = DATA / "echo_pcap_extraction_manifest.json"
EXPECTED_MD5 = "311cbd100eeddf76c8056919fe35b0cb"
EXPECTED_DAYS = {f"2023-02-{day:02d}" for day in range(20, 27)}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    actual = md5(ARCHIVE)
    if actual != EXPECTED_MD5:
        raise ValueError(f"Official Echo MD5 mismatch: {actual}")
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty extraction directory {DEST}")
    DEST.mkdir(parents=True, exist_ok=True)
    extracted = []
    with tarfile.open(ARCHIVE, mode="r:bz2") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if len(parts) < 5 or parts[0] != "CTU-SME-11" or ".." in parts:
                continue
            day = parts[2]
            tail = PurePosixPath(*parts[3:])
            if day not in EXPECTED_DAYS or tail.parts[0] != "raw" or not tail.name.endswith(".pcap"):
                continue
            if not member.isfile():
                continue
            relative = Path(day) / tail.name
            target = (DEST / relative).resolve()
            if DEST.resolve() not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                raise IOError(f"Could not extract {member.name}")
            digest = hashlib.sha256()
            written = 0
            with src, target.open("wb") as out:
                for block in iter(lambda: src.read(1024 * 1024), b""):
                    out.write(block)
                    digest.update(block)
                    written += len(block)
            if written != member.size:
                raise IOError(f"Member size mismatch: {member.name}")
            extracted.append({"archive_member": member.name,
                              "output": str(target.relative_to(ROOT)).replace("\\", "/"),
                              "size_bytes": written,
                              "sha256": digest.hexdigest()})
    days = {Path(item["output"]).parts[-2] for item in extracted}
    if len(extracted) != 7 or days != EXPECTED_DAYS:
        raise ValueError(f"Expected 7 daily PCAPs; got {len(extracted)} files, days={sorted(days)}")
    manifest = {"archive_md5_verified": actual, "pcaps_extracted": len(extracted),
                "total_uncompressed_bytes": sum(item["size_bytes"] for item in extracted),
                "files": extracted}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted {len(extracted)} PCAPs, {manifest['total_uncompressed_bytes']:,} bytes")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
