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
