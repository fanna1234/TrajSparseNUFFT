#!/usr/bin/env python3
"""Admit a native cuFINUFFT configuration before paired timing."""

from __future__ import annotations

import argparse
import json
import itertools
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = summary["rows"]
    expected = set(itertools.product(("spiral", "radial", "golden"), range(3)))
    if len(rows) != 9 or {(r["trajectory"], r["frame"]) for r in rows} != expected:
        raise RuntimeError("cuFINUFFT quality has missing or duplicate logical rows")
    for row in rows:
        replicas = row["replica_quality"]
        if len(replicas) != 2 or not all(
            q["application_vs_fp32_pass"] is True and q["finite"] is True
            for q in replicas
        ):
            raise RuntimeError("cuFINUFFT replica quality is incomplete or failed")
        if not math.isfinite(float(row["repeat_reconstruction_complex_rel_l2"])):
            raise RuntimeError("cuFINUFFT repeat diagnostic is nonfinite")
    if (summary["forward_method"], summary["adjoint_method"], summary["gpu_sort"],
        summary["scale_rule"]["raw_scale"]) != (1, 2, 0, 512):
        raise RuntimeError("cuFINUFFT configuration differs from the frozen contract")
    checks = {
        "logical rows": summary["row_count"] == 9,
        "replica rows": summary["replica_row_count"] == 18,
        "replica passes": summary["replica_application_passes"] == 18,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("cuFINUFFT admission failed: " + ", ".join(failed))
    stable_repeats = sum(
        row["repeat_reconstruction_complex_rel_l2"] <= 0.002
        for row in summary["rows"]
    )
    admission = {
        "schema_version": 1,
        "baseline": "cuFINUFFT hybrid type2-method1/type1-method2 sort0",
        "all_output_application_pass": True,
        "output_application_passes": summary["replica_application_passes"],
        "output_count": summary["replica_row_count"],
        "repeat_stability_passes_at_2e3": stable_repeats,
        "repeat_pair_count": summary["row_count"],
        "raw_scale": summary["scale_rule"]["raw_scale"],
        "binary_sha256": summary["binary_sha256"],
        "forward_method": summary["forward_method"],
        "adjoint_method": summary["adjoint_method"],
        "gpu_sort": summary["gpu_sort"],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(admission, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "[OK] cuFINUFFT quality: "
        f"{admission['output_application_passes']}/{admission['output_count']} outputs; "
        f"repeat diagnostic={stable_repeats}/{summary['row_count']}"
    )


if __name__ == "__main__":
    main()
