#!/usr/bin/env python3
"""Fail unless a fresh development plus held-out quality run closes 27 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    args = parser.parse_args()
    development = json.loads(args.development.read_text(encoding="utf-8"))
    heldout = json.loads(args.heldout.read_text(encoding="utf-8"))

    checks = {
        "development rows": len(development["rows"]) == 9,
        "development quality": development["all_application_pass"],
        "development determinism": development["all_eager_graph_bitwise"],
        "held-out rows": heldout["row_count"] == 18,
        "held-out quality": heldout["all_admission_pass"],
        "held-out determinism": heldout["all_eager_graph_bitwise"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("quality gate failed: " + ", ".join(failed))
    print("[OK] 27/27 quality rows and eager/Graph determinism")


if __name__ == "__main__":
    main()
