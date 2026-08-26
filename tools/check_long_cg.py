#!/usr/bin/env python3
"""Validate the fixed 10--50-step long-CG stability sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    iterations = [row["cg_iterations"] for row in summary["rows"]]
    if iterations != [10, 20, 30, 50]:
        raise RuntimeError(f"unexpected long-CG endpoints: {iterations}")
    if not summary["all_endpoints_pass"]:
        raise RuntimeError("at least one long-CG endpoint failed")
    if not summary["all_native_denominators_positive_finite"]:
        raise RuntimeError("long-CG curvature gate failed")
    print("[OK] long-CG endpoints 10/20/30/50 and positive curvature")


if __name__ == "__main__":
    main()
