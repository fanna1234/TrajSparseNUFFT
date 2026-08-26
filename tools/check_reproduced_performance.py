#!/usr/bin/env python3
"""Compare a fresh paired run with its frozen matched-quality anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ANCHORS = {
    "cufinufft": {"eager": 2.514278441069968},
    "dense": {
        "eager": 1.4082191001195759,
        "graph": 1.4137558002480994,
    },
}


def classify(value: float, anchor: float) -> tuple[str, bool]:
    if value >= anchor:
        return "OK >= anchor", True
    if value >= 0.97 * anchor:
        return "WITHIN 3%", True
    return "LOW", False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=sorted(ANCHORS))
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    if args.kind == "cufinufft":
        rows = {"eager": summary}
    else:
        rows = {row["mode"]: row["aggregate"] for row in summary["summaries"]}

    passed = True
    for mode, anchor in ANCHORS[args.kind].items():
        row = rows[mode]
        value = float(row["paired_geomean_speedup"])
        wins = int(row["wins"])
        status, accepted = classify(value, anchor)
        print(f"[{status}] {args.kind}/{mode}: {value:.4f}x; wins={wins}/6")
        passed = passed and accepted and wins == 6
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
