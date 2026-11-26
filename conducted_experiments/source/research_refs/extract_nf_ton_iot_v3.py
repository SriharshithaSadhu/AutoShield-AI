"""Extract and BagIt-SHA1-verify the two data payloads in official UQ package."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "nf_ton_iot_v3"
ARCHIVE = ROOT / "NF-ToN-IoT-v3_UQ.zip"
EXPECTED = {
    "data/NF-ToN-IoT-v3.csv": "24854ca3072ab7c2ade9ebfb202345a666a94cfe",
    "data/NetFlow_v3_Features.csv": "d028597391217f78df3db31c1dbd96805203196f",
}


def copy_and_sha1(source, destination: Path) -> tuple[str, int]:
    digest = hashlib.sha1(usedforsecurity=False)
    size = 0
    with destination.open("wb") as sink:
        while block := source.read(4 * 1024 * 1024):
            sink.write(block)
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


if not ARCHIVE.is_file():
    raise FileNotFoundError(ARCHIVE)
results = []
with zipfile.ZipFile(ARCHIVE) as archive:
    members = {info.filename: info for info in archive.infolist()}
    root = next(iter(members)).split("/", 1)[0] + "/"
    bag_manifest = archive.read(root + "manifest-sha1.txt").decode("ascii")
    declared = {line.split(None, 1)[1].strip(): line.split(None, 1)[0] for line in bag_manifest.splitlines() if line.strip()}
    if declared != EXPECTED:
        raise RuntimeError(f"Publisher BagIt manifest differs from audited values: {declared}")
    for relative, expected_sha1 in EXPECTED.items():
        member = root + relative
        if member not in members:
            raise RuntimeError(f"Missing expected archive member: {relative}")
        target = ROOT / Path(relative).name
        if target.exists():
            digest = hashlib.sha1(usedforsecurity=False)
            size = 0
            with target.open("rb") as source:
                while block := source.read(4 * 1024 * 1024):
                    digest.update(block)
                    size += len(block)
            actual_sha1 = digest.hexdigest()
        else:
            tmp = target.with_suffix(target.suffix + ".part")
            try:
                with archive.open(member) as source:
                    actual_sha1, size = copy_and_sha1(source, tmp)
                if actual_sha1 != expected_sha1:
                    raise RuntimeError(f"BagIt SHA-1 mismatch for {relative}: {actual_sha1}")
                tmp.replace(target)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
        if actual_sha1 != expected_sha1:
            raise RuntimeError(f"BagIt SHA-1 mismatch for existing {relative}: {actual_sha1}")
        results.append({"path": str(target.relative_to(ROOT)), "bytes": size, "sha1_bagit": actual_sha1, "verified": True})
report = {"archive": ARCHIVE.name, "extracted_and_verified": results}
(ROOT / "extraction_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
