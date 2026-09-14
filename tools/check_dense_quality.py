#!/usr/bin/env python3
"""Validate a fresh dense-control reconstruction-quality matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quality_contract import validate_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    validate_rows(summary["rows"])
    checks = {
        "rows": len(summary["rows"]) == 9,
        "application quality": summary["all_application_pass"],
        "eager/Graph determinism": summary["all_eager_graph_bitwise"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("dense-control quality failed: " + ", ".join(failed))
    print("[OK] dense-control quality: 9/9 rows and eager/Graph determinism")


if __name__ == "__main__":
    main()
