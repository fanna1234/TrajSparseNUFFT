#!/usr/bin/env python3
"""Derive the compact result record from the shipped source summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_FILES = {
    "development_quality": "development_quality.json",
    "heldout_quality": "heldout_quality.json",
    "dense_quality": "dense_quality.json",
    "cufinufft_quality_1": "cufinufft_quality_1.json",
    "cufinufft_quality_2": "cufinufft_quality_2.json",
    "cufinufft_performance": "cufinufft_performance.json",
    "dense_performance": "dense_performance.json",
    "long_cg": "long_cg.json",
    "hardening": "hardening.json",
    "scale_latency": "scale_latency.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sources(repo: Path) -> tuple[dict[str, dict], dict[str, str]]:
    root = repo / "evidence/source"
    rows = {}
    hashes = {}
    for name, filename in SOURCE_FILES.items():
        path = root / filename
        rows[name] = json.loads(path.read_text(encoding="utf-8"))
        hashes[str(path.relative_to(repo))] = sha256(path)
    return rows, hashes


def derive(repo: Path) -> dict:
    source, hashes = load_sources(repo)
    development = source["development_quality"]
    heldout = source["heldout_quality"]
    dense_quality = source["dense_quality"]
    external = source["cufinufft_performance"]
    dense_modes = {
        row["mode"]: row["aggregate"]
        for row in source["dense_performance"]["summaries"]
    }
    quality_rows = development["rows"] + heldout["rows"]
    baseline_quality = [
        source["cufinufft_quality_1"],
        source["cufinufft_quality_2"],
    ]
    long_row = source["long_cg"]["rows"][-1]
    long_quality = long_row["quality"]
    hardening = source["hardening"]

    return {
        "schema_version": 1,
        "decision": "accepted",
        "source_sha256": hashes,
        "quality": {
            "rows": len(quality_rows),
            "application_passes": sum(
                bool(row["application_pass"]) for row in quality_rows
            ),
            "all_eager_graph_bitwise": bool(
                development["all_eager_graph_bitwise"]
                and heldout["all_eager_graph_bitwise"]
            ),
            "worst_magnitude_nrmse": max(
                row["native_vs_fp32_magnitude_nrmse"] for row in quality_rows
            ),
            "minimum_ssim": min(
                row["native_vs_fp32_ssim"] for row in quality_rows
            ),
            "minimum_psnr_db": min(
                row["native_vs_fp32_psnr_db"] for row in quality_rows
            ),
        },
        "cufinufft_quality": {
            "output_count": sum(row["replica_row_count"] for row in baseline_quality),
            "output_application_passes": sum(
                row["replica_application_passes"] for row in baseline_quality
            ),
            "repeat_pair_count": sum(row["row_count"] for row in baseline_quality),
            "repeat_stability_passes": sum(
                item["repeat_reconstruction_complex_rel_l2"] <= 0.002
                for row in baseline_quality
                for item in row["rows"]
            ),
        },
        "performance": {
            "cufinufft": {
                "speedup": external["paired_geomean_speedup"],
                "ci95_low": external["paired_ci95_low"],
                "ci95_high": external["paired_ci95_high"],
                "pairs": external["pairs"],
                "wins": external["wins"],
            },
            "dense_eager": {
                "speedup": dense_modes["eager"]["paired_geomean_speedup"],
                "ci95_low": dense_modes["eager"]["paired_ci95_low"],
                "ci95_high": dense_modes["eager"]["paired_ci95_high"],
                "wins": dense_modes["eager"]["wins"],
            },
            "dense_graph": {
                "speedup": dense_modes["graph"]["paired_geomean_speedup"],
                "ci95_low": dense_modes["graph"]["paired_ci95_low"],
                "ci95_high": dense_modes["graph"]["paired_ci95_high"],
                "wins": dense_modes["graph"]["wins"],
            },
        },
        "dense_quality": {
            "rows": len(dense_quality["rows"]),
            "all_application_pass": dense_quality["all_application_pass"],
            "all_eager_graph_bitwise": dense_quality["all_eager_graph_bitwise"],
        },
        "long_cg": {
            "iterations": long_row["cg_iterations"],
            "endpoint_pass": long_row["endpoint_pass"],
            "minimum_denominator": long_row["minimum_denominator"],
            "magnitude_nrmse": long_quality["native_vs_fp32_magnitude_nrmse"],
            "ssim": long_quality["native_vs_fp32_ssim"],
        },
        "hardening": {
            "all_memcheck_pass": hardening["all_memcheck_pass"],
            "all_stress_pass": hardening["all_stress_pass"],
            "cufinufft_runtime_contains_truth": hardening[
                "cufinufft_runtime_contains_truth"
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = derive(args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
