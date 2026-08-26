#!/usr/bin/env python3
"""Build production-only G/GH packs concurrently for the five-case matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


CASES = [
    ("spiral65", "spiral_standard_256.npy"),
    ("radial32", "radial_128x256.npy"),
    ("radial65", "radial_256x256.npy"),
    ("golden32", "golden_128x256.npy"),
    ("golden65", "golden_256x256.npy"),
]
PHASE_FILES = {
    "header.bin",
    "group_offsets.bin",
    "group_row_ids.bin",
    "tile_col_ids.bin",
    "tile_a_comp.bin",
    "tile_pair_meta.bin",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def view_summary(root: Path) -> dict:
    real = json.loads((root / "real/manifest.json").read_text(encoding="utf-8"))
    imag = json.loads((root / "imag/manifest.json").read_text(encoding="utf-8"))
    real_files = {entry["file"]: entry["bytes"] for entry in real["files"]}
    imag_files = {entry["file"]: entry["bytes"] for entry in imag["files"]}
    shared = {"header.bin", "group_offsets.bin", "group_row_ids.bin", "tile_col_ids.bin"}
    component = {"tile_a_comp.bin", "tile_pair_meta.bin"}
    phase_bytes = sum(real_files[name] for name in shared) + sum(
        real_files[name] + imag_files[name] for name in component
    )
    return {
        "root": str(root),
        "tiles": real["tiles"],
        "groups": real["groups"],
        "real_utilization": real["fp16_useful_slot_utilization"],
        "imag_utilization": imag["fp16_useful_slot_utilization"],
        "shared_structure_pack_seconds": real["shared_structure_pack_seconds"],
        "real_export_seconds": real["component_export_seconds"],
        "imag_export_seconds": imag["component_export_seconds"],
        "phase_live_bytes": phase_bytes,
        "production_only": bool(real["production_only"] and imag["production_only"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner-workers", type=int, default=8)
    parser.add_argument("--component-workers", type=int, default=2)
    parser.add_argument("--policy", default="morton")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    python = args.python
    if not python.is_absolute():
        python = repo / python
    python = python.absolute()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    exporter = repo / "src/export_torchkbnufft_gpu_case.py"
    trajectory_root = (
        repo / "experiments/data_prep/trajectories"
    )
    rows = []
    for case, trajectory_name in CASES:
        case_root = output / case
        processes = []
        started = time.perf_counter()
        for view, directory in (("G", "G"), ("G_T", "G_T")):
            command = [
                str(python),
                str(exporter),
                "--trajectory-file",
                str(trajectory_root / trajectory_name),
                "--view",
                view,
                "--im-size",
                "256",
                "--grid-size",
                "512",
                "--numpoints",
                "6",
                "--policy",
                args.policy,
                "--optimized-pack",
                "--planner-workers",
                str(args.planner_workers),
                "--component-workers",
                str(args.component_workers),
                "--production-only",
                "--seed",
                "20260823",
                "--experiment-id",
                "packing",
                "--output-root",
                str(case_root / directory),
            ]
            log_path = raw / f"{case}_{view}.log"
            log = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, text=True
            )
            processes.append((view, process, log, log_path, command))
        for view, process, log, log_path, command in processes:
            status = process.wait()
            log.close()
            if status != 0:
                raise RuntimeError(f"planner failed for {case}/{view}: {status}")
        wall_seconds = time.perf_counter() - started
        row = {
            "case": case,
            "trajectory": str(trajectory_root / trajectory_name),
            "wall_seconds": wall_seconds,
            "forward": view_summary(case_root / "G"),
            "adjoint": view_summary(case_root / "G_T"),
        }
        row["dual_view_phase_live_bytes"] = (
            row["forward"]["phase_live_bytes"]
            + row["adjoint"]["phase_live_bytes"]
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    summary = {
        "experiment_id": "packing",
        "exporter": str(exporter),
        "exporter_sha256": sha256(exporter),
        "python": str(python),
        "policy": args.policy,
        "planner_workers_per_view": args.planner_workers,
        "component_workers_per_view": args.component_workers,
        "views_run_concurrently": True,
        "rows": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
