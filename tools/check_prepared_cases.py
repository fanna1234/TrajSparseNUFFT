#!/usr/bin/env python3
"""Validate the semantic contract of the three prepared OCMR cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "fs0152": {
        "source_sha256": "65ff79868b4b273aeee550c5bbf3e4ce266d12f0d9c23d435e406d5467e026b5",
        "source_coils": 15,
    },
    "fs0005_v2": {
        "source_sha256": "1f3beee40b9186337b18f5acb6ff803c03e7b3ce4b68b3d8dc9b619cbcd320d4",
        "source_coils": 18,
    },
    "fs0016_v2": {
        "source_sha256": "763525771bf7617a2e5cea5c3d106f503df553d2de78f20d5fd2a3f6d4404740",
        "source_coils": 30,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases_root", type=Path)
    args = parser.parse_args()
    for name, expected in EXPECTED.items():
        manifest = json.loads(
            (args.cases_root / name / "manifest.json").read_text(encoding="utf-8")
        )
        checks = {
            "source SHA-256": manifest["source_sha256"] == expected["source_sha256"],
            "source coils": manifest["source_coils"] == expected["source_coils"],
            "selected phases": manifest["selected_phases"] == [0, 6, 13],
            "output shape": manifest["output_shape"] == [256, 256],
            "virtual coils": manifest["output_virtual_coils"] == 8,
            "retrospective flag": manifest["retrospective_noncartesian"] is True,
        }
        failed = [label for label, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"{name} preparation failed: {', '.join(failed)}")
        print(f"[OK] {name} prepared-case contract")


if __name__ == "__main__":
    main()
