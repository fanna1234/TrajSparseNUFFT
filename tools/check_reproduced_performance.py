#!/usr/bin/env python3
"""Compare a fresh paired run with its frozen matched-quality anchor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ANCHORS = {
    "cufinufft": {"eager": 2.514278441069968},
    "dense": {
        "eager": 1.4082191001195759,
        "graph": 1.4137558002480994,
    },
}


def classify(value: float, anchor: float) -> tuple[str, bool]:
    if not math.isfinite(value) or value <= 0:
        return "INVALID", False
    if value >= anchor:
        return "OK >= anchor", True
    if value >= 0.97 * anchor:
        return "WITHIN 3%", True
    return "LOW", False


def validate_pairs(summary: dict, kind: str) -> None:
    pairs = summary["aggregate_rows"] if kind == "cufinufft" else summary["rows"]
    if len(pairs) != 6 or {row["pair"] for row in pairs} != set(range(6)):
        raise RuntimeError("performance requires six complete, distinct process pairs")
    ratios = []
    for row in pairs:
        numerator = float(row["cufinufft_sum_us" if kind == "cufinufft" else "dense_sum_us"])
        denominator = float(row["production_sum_us" if kind == "cufinufft" else "sparse_sum_us"])
        if not all(math.isfinite(x) and x > 0 for x in (numerator, denominator)):
            raise RuntimeError("invalid latency in process-pair record")
        ratio = numerator / denominator
        saved = float(row["cufinufft_over_production" if kind == "cufinufft" else "sparse_speedup"])
        if not math.isclose(ratio, saved, rel_tol=1e-9):
            raise RuntimeError("process-pair ratio does not match its latencies")
        ratios.append(ratio)
    measured = math.exp(sum(map(math.log, ratios)) / len(ratios))
    if not math.isclose(measured, float(summary["paired_geomean_speedup"]), rel_tol=1e-9):
        raise RuntimeError("aggregate speedup does not match process-pair records")
    if summary["wins"] != sum(ratio > 1 for ratio in ratios):
        raise RuntimeError("win count does not match process-pair records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=sorted(ANCHORS))
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    if args.kind == "cufinufft":
        if summary["pairs"] != 6:
            raise RuntimeError("external comparison requires six process pairs")
        rows = {"eager": summary}
    else:
        modes = [row["mode"] for row in summary["summaries"]]
        if sorted(modes) != ["eager", "graph"]:
            raise RuntimeError("dense comparison requires exactly eager and Graph modes")
        rows = {row["mode"]: row["aggregate"] for row in summary["summaries"]}

    passed = True
    for mode, anchor in ANCHORS[args.kind].items():
        row = rows[mode]
        validate_pairs(row, args.kind)
        value = float(row["paired_geomean_speedup"])
        wins = int(row["wins"])
        status, accepted = classify(value, anchor)
        print(f"[{status}] {args.kind}/{mode}: {value:.4f}x; wins={wins}/6")
        passed = passed and accepted and wins == 6
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
