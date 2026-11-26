"""Extract only the Raspberry Pi CTU-SME-11 labeled Zeek flow logs."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ctu_sme11"
ARCHIVE = DATA / "CTU-SME-11_Honeypot-Minicomputer-RaspberryPi-Gen3-20_v1.0.0.tar.bz2"
DEST = DATA / "raspberrypi_extracted"
MANIFEST = DATA / "raspberrypi_extracted_manifest.json"
EXPECTED_MD5 = "94285412e422f9983b7431c32acaf5b9"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if md5(ARCHIVE) != EXPECTED_MD5:
        raise ValueError("Archive MD5 does not match the official Zenodo checksum")
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite {DEST}")
    DEST.mkdir(parents=True)
    extracted = []
    with tarfile.open(ARCHIVE, "r:bz2") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if len(parts) < 5 or parts[0] != "CTU-SME-11" or ".." in parts:
                continue
            day = parts[2]
            tail = PurePosixPath(*parts[3:])
            if len(day) != 10 or day[4] != "-" or not member.isfile():
                continue
            if tail == PurePosixPath("zeek/conn.log.labeled"):
                relative = Path(day) / "zeek" / "conn.log.labeled"
            elif tail == PurePosixPath("artifacts/labels.config"):
                relative = Path(day) / "artifacts" / "labels.config"
            else:
                continue
            target = (DEST / relative).resolve()
            if DEST.resolve() not in target.parents:
                raise ValueError(f"Unsafe member path: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise IOError(f"Could not extract {member.name}")
            digest = hashlib.sha256()
            written = 0
            with source, target.open("wb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    written += len(block)
            if written != member.size:
                raise IOError(f"Size mismatch for {member.name}")
            extracted.append({
                "archive_member": member.name,
                "output": str(target.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": written,
                "sha256": digest.hexdigest(),
            })
    days = sorted({Path(x["output"]).parts[-3] for x in extracted
                   if x["output"].endswith("/zeek/conn.log.labeled")})
    labels_days = sorted({Path(x["output"]).parts[-3] for x in extracted
                          if x["output"].endswith("/artifacts/labels.config")})
    if len(days) != 7 or days != labels_days:
        raise ValueError(f"Expected seven matching days; conn={days}, labels={labels_days}")
    manifest = {"archive_md5_verified": EXPECTED_MD5, "capture_days": days,
                "extracted_files": extracted, "pcaps_extracted": False}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted {len(extracted)} labeled flow/config files across {len(days)} days")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
