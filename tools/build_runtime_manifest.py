#!/usr/bin/env python3
"""Build a root manifest over per-frame runtime-input manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "manifest.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifests = sorted(root.glob("**/frame??/manifest.json"))
    if len(manifests) != args.expected:
        raise RuntimeError(
            f"found {len(manifests)} frame manifests under {root}; "
            f"expected {args.expected}"
        )
    for path in manifests:
        cases = json.loads(path.read_text(encoding="utf-8"))
        if len(cases) != 3 or {case["name"] for case in cases} != {
            "spiral65", "radial65", "golden65"
        }:
            raise RuntimeError(f"incomplete trajectory matrix in {path}")
        for case in cases:
            if case["measurement_scale"] != 1:
                raise RuntimeError(
                    "runtime measurements must be unscaled; raw scale 512 "
                    "belongs to operator normalization, not measurement export"
                )
            expected_files = {
                "forward_coils.c64.bin", "measurement.c64.bin", "scaling.f32.bin",
                "density.f32.bin", "sensitivity_maps.c64.bin", "truth.f32.bin", "truth.c64.bin",
            }
            entries = case["entries"]
            if len(entries) != len(expected_files) or {e["file"] for e in entries} != expected_files:
                raise RuntimeError(f"runtime manifest has missing or duplicate payloads: {path}")
            for entry in case["entries"]:
                payload = path.parent / case["name"] / entry["file"]
                if payload.stat().st_size != entry["bytes"] or sha256(payload) != entry["sha256"]:
                    raise RuntimeError(f"runtime payload fails its manifest: {payload}")
    record = {
        "schema_version": 1,
        "root": ".",
        "frame_manifests": [
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256(path),
            }
            for path in manifests
        ],
    }
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] runtime manifest: {len(manifests)} frames")


if __name__ == "__main__":
    main()
