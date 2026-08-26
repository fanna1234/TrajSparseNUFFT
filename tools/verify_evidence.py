#!/usr/bin/env python3
"""Recompute the result record and validate its evidence gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from build_evidence import derive


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    committed = json.loads(
        (repo / "evidence/frozen_results.json").read_text(encoding="utf-8")
    )
    recomputed = derive(repo)
    require(committed == recomputed, "frozen result record or source hash changed")

    quality = recomputed["quality"]
    require(quality["rows"] == 27, "quality matrix must contain 27 rows")
    require(quality["application_passes"] == 27, "not all quality rows pass")
    require(quality["all_eager_graph_bitwise"], "eager/Graph determinism failed")

    baseline = recomputed["cufinufft_quality"]
    require(
        baseline["output_application_passes"] == baseline["output_count"] == 36,
        "cuFINUFFT output quality is incomplete",
    )

    performance = recomputed["performance"]
    external = performance["cufinufft"]
    require(external["wins"] == external["pairs"] == 6, "external pairs incomplete")
    require(close(external["speedup"], 2.514278441069968), "external anchor changed")
    require(
        close(performance["dense_eager"]["speedup"], 1.4082191001195759),
        "dense eager anchor changed",
    )
    require(
        close(performance["dense_graph"]["speedup"], 1.4137558002480994),
        "dense Graph anchor changed",
    )
    require(recomputed["hardening"]["all_memcheck_pass"], "memcheck gate failed")
    require(recomputed["hardening"]["all_stress_pass"], "stress gate failed")

    print("[OK] source hashes and derived result record")
    print("[OK] 27/27 reconstruction-quality rows")
    print("[OK] 36/36 cuFINUFFT admission outputs")
    print("[OK] 6/6 external and dense-control process-pair wins")


if __name__ == "__main__":
    main()
