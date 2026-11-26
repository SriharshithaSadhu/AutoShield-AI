"""Download pinned, anonymized UGR'16 feature variants v3 and v4.

The files are the compact 1-minute feature release from the dataset authors,
not the 236 GB raw flow corpus. Every downloaded byte is verified against the
GitHub Git-blob SHA-1 advertised by the pinned author-repository commit; a local
SHA-256 manifest is also written for later reproduction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "ugr16_feature_data" / "raw"
COMMIT = "760863d31413739a129b3be2d1d52cde217962e8"
BASE = f"https://raw.githubusercontent.com/josecamachop/UGR16_FeatureData/{COMMIT}/csv"
FILES = {
    "UGR16v3.Xtest.csv": (11_397_594, "2c701c3b41b2a51e09b92f172bf07a0ca3a1d8eb"),
    "UGR16v3.Xtrain.csv": (29_942_889, "3cbb438fad330959e0659c17a07589b172faad27"),
    "UGR16v3.Ytest.csv": (861_370, "cc558b377a97dc0dc134483a81bc7f27c8749aec"),
    "UGR16v3.Ytrain.csv": (2_246_131, "33f452ce7c1dfb8e097ebab7865b616a7cb4d4f5"),
    "UGR16v4.Xtest.csv": (12_478_279, "68012a690e52751c1598e8aca66e9061b61b49e3"),
    "UGR16v4.Xtrain.csv": (32_686_693, "ea16b0666baa0fabcd9aa0a8989bfde6ae5aae5e"),
    "UGR16v4.Ytest.csv": (861_370, "cc558b377a97dc0dc134483a81bc7f27c8749aec"),
    "UGR16v4.Ytrain.csv": (2_246_131, "33f452ce7c1dfb8e097ebab7865b616a7cb4d4f5"),
}


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, (expected_size, expected_git_sha1) in FILES.items():
        path = OUT / name
        if path.exists() and path.stat().st_size == expected_size:
            payload = path.read_bytes()
        else:
            request = Request(f"{BASE}/{name}", headers={"User-Agent": "research-reproducibility-audit"})
            with urlopen(request, timeout=120) as response:
                payload = response.read()
            if len(payload) != expected_size:
                raise ValueError(f"{name}: expected {expected_size} bytes, received {len(payload)}")
            path.write_bytes(payload)
        observed_git_sha1 = git_blob_sha1(payload)
        if observed_git_sha1 != expected_git_sha1:
            raise ValueError(f"{name}: Git blob SHA-1 mismatch: {observed_git_sha1}")
        entries.append({
            "file": name,
            "bytes": len(payload),
            "git_blob_sha1": observed_git_sha1,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source_url": f"https://github.com/josecamachop/UGR16_FeatureData/blob/{COMMIT}/csv/{name}",
        })
        print(f"verified {name}: {len(payload):,} bytes")
    manifest = {
        "dataset": "UGR'16 Feature Data, anonymized variants v3 and v4",
        "source_repository": "https://github.com/josecamachop/UGR16_FeatureData",
        "source_commit": COMMIT,
        "files": entries,
        "dataset_license_source": "CC BY-NC, as stated by the 2025 Data paper; verify attribution/noncommercial terms before redistribution",
        "paper_doi": ["10.3390/data10030033", "10.1016/j.cose.2017.11.004"],
    }
    manifest_path = OUT.parent / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
