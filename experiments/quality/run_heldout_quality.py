#!/usr/bin/env python3
"""Run eager/Graph native CG and FP32 validation on held-out OCMR cases."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ACQUISITIONS = {
    "fs0005": "fs0005_v2",
    "fs0016": "fs0016_v2",
}
TRAJECTORY_FILES = {
    "spiral": "spiral_standard_256.npy",
    "radial": "radial_256x256.npy",
    "golden": "golden_256x256.npy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command: list[str], log: Path, label: str) -> None:
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    log.write_text(
        "$ "
        + " ".join(command)
        + "\n\n[stdout]\n"
        + process.stdout
        + "\n[stderr]\n"
        + process.stderr
    )
    if process.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {process.returncode}")


def gpu_snapshot() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader",
    ]
    gpu = subprocess.run(command, text=True, capture_output=True, check=True)
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return {
        "gpu": [line for line in gpu.stdout.splitlines() if line.strip()],
        "compute_processes": [
            line for line in processes.stdout.splitlines() if line.strip()
        ],
    }


def foreign_gpu_processes(snapshot: dict) -> list[str]:
    allowed = ("gnome-remote-desktop-daemon", "Xorg")
    return [
        process
        for process in snapshot["compute_processes"]
        if not any(name in process for name in allowed)
    ]


def assert_gpu_available(snapshot: dict) -> None:
    foreign = foreign_gpu_processes(snapshot)
    if foreign:
        raise RuntimeError(
            "GPU is occupied by a foreign compute process: " + "; ".join(foreign)
        )


def assert_quality_memory_headroom(snapshot: dict, minimum_free_mib: int = 8192) -> None:
    free_values = []
    for row in snapshot["gpu"]:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) < 2:
            continue
        match = re.fullmatch(r"(\d+) MiB", fields[-2])
        if match:
            free_values.append(int(match.group(1)))
    if not free_values or min(free_values) < minimum_free_mib:
        raise RuntimeError(
            f"quality run requires at least {minimum_free_mib} MiB free: {snapshot}"
        )


def parse_scales(path: Path) -> dict[tuple[str, str, int], dict]:
    ledger = json.loads(path.read_text())
    scales = {
        (row["acquisition"], row["trajectory"], int(row["frame"])): row
        for row in ledger["rows"]
    }
    if len(scales) != len(ledger["rows"]):
        raise ValueError("duplicate scale ledger key")
    return scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    scale_group = parser.add_mutually_exclusive_group(required=True)
    scale_group.add_argument("--scales", type=Path)
    scale_group.add_argument(
        "--raw-scale",
        type=float,
        help="truth-free raw scale shared by every selected row",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-co-resident-quality",
        action="store_true",
        help="allow recorded foreign GPU residency; all timing remains inadmissible",
    )
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        help="optional ACQUISITION:TRAJECTORY:FRAME selector",
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    binary = args.binary.resolve()
    runtime_root = args.runtime_root.resolve()
    packed_root = args.packed_root.resolve()
    cases_root = args.cases_root.resolve()
    scale_path = args.scales.resolve() if args.scales else None
    if args.raw_scale is not None and args.raw_scale <= 0.0:
        raise ValueError("raw-scale must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    selections = {
        (parts[0], parts[1], int(parts[2]))
        for item in args.select
        for parts in [item.split(":")]
        if len(parts) == 3
    }
    if len(selections) != len(args.select):
        raise ValueError("selectors must be ACQUISITION:TRAJECTORY:FRAME")

    production = (
        repo / "experiments/production"
    )
    validator = production / "validate_native_cg.py"
    trajectory_root = (
        repo / "experiments/data_prep/trajectories"
    )
    scales = parse_scales(scale_path) if scale_path else {}
    initial_gpu = gpu_snapshot()
    if args.allow_co_resident_quality:
        assert_quality_memory_headroom(initial_gpu)
    else:
        assert_gpu_available(initial_gpu)
    rows = []
    launch_gpu_snapshots = []

    with open("/tmp/trajsparsenufft_gpu.lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for acquisition, case_name in ACQUISITIONS.items():
            case = cases_root / case_name
            for trajectory in ("spiral", "radial", "golden"):
                pack = packed_root / f"{trajectory}65"
                trajectory_path = trajectory_root / TRAJECTORY_FILES[trajectory]
                for frame in range(3):
                    key = (acquisition, trajectory, frame)
                    if selections and key not in selections:
                        continue
                    if args.raw_scale is not None:
                        raw_scale = float(args.raw_scale)
                        native_scale = raw_scale / 512.0
                        scale_row = {
                            "kind": "fixed_truth_free",
                            "raw_scale": raw_scale,
                            "native_scale": native_scale,
                        }
                    else:
                        if key not in scales:
                            raise KeyError(f"missing scale {key}")
                        scale_row = scales[key]
                        raw_scale = float(scale_row["raw_scale"])
                        native_scale = float(scale_row["native_scale"])
                    runtime = (
                        runtime_root
                        / acquisition
                        / f"frame{frame:02d}"
                        / f"{trajectory}65"
                    )
                    mode_dirs = {}
                    for graph in (0, 1):
                        mode = "graph" if graph else "eager"
                        current_gpu = gpu_snapshot()
                        launch_gpu_snapshots.append(
                            {
                                "acquisition": acquisition,
                                "trajectory": trajectory,
                                "frame": frame,
                                "mode": mode,
                                "snapshot": current_gpu,
                            }
                        )
                        if args.allow_co_resident_quality:
                            assert_quality_memory_headroom(current_gpu)
                        else:
                            assert_gpu_available(current_gpu)
                        mode_dir = (
                            output
                            / f"{acquisition}_{trajectory}_frame{frame}_{mode}"
                        )
                        mode_dir.mkdir(parents=True, exist_ok=False)
                        command = [
                            str(binary),
                            str(pack / "G/real"),
                            str(pack / "G/imag"),
                            str(pack / "G_T/real"),
                            str(pack / "G_T/imag"),
                            str(runtime),
                            repr(native_scale),
                            "0",
                            "1",
                            "0",
                            str(graph),
                            "10",
                            str(mode_dir),
                        ]
                        run_logged(
                            command,
                            mode_dir / "run.log",
                            f"native {key}/{mode}",
                        )
                        mode_dirs[mode] = mode_dir

                    validation_path = mode_dirs["eager"] / "quality.json"
                    validate_command = [
                        sys.executable,
                        str(validator),
                        "--repo-root",
                        str(repo),
                        "--case",
                        str(case),
                        "--frame",
                        str(frame),
                        "--runtime-dir",
                        str(runtime),
                        "--trajectory",
                        str(trajectory_path),
                        "--operator-scale",
                        repr(native_scale),
                        "--emulation-operator-scale",
                        repr(raw_scale),
                        "--native-dir",
                        str(mode_dirs["eager"]),
                        "--output",
                        str(validation_path),
                        "--save-emulation",
                        str(mode_dirs["eager"] / "emulation_reconstruction.c64.bin"),
                        "--save-fp32",
                        str(mode_dirs["eager"] / "fp32_reconstruction.c64.bin"),
                    ]
                    run_logged(
                        validate_command,
                        mode_dirs["eager"] / "validate.log",
                        f"validation {key}",
                    )
                    row = json.loads(validation_path.read_text())
                    eager_reconstruction = (
                        mode_dirs["eager"] / "reconstruction.c64.bin"
                    )
                    graph_reconstruction = (
                        mode_dirs["graph"] / "reconstruction.c64.bin"
                    )
                    eager_residuals = mode_dirs["eager"] / "residuals.f32.bin"
                    graph_residuals = mode_dirs["graph"] / "residuals.f32.bin"
                    row.update(
                        acquisition=acquisition,
                        case=case_name,
                        trajectory=trajectory,
                        phase=(0, 6, 13)[frame],
                        scale_ledger_row=scale_row,
                        eager_graph_reconstruction_bitwise=(
                            eager_reconstruction.read_bytes()
                            == graph_reconstruction.read_bytes()
                        ),
                        eager_graph_residuals_bitwise=(
                            eager_residuals.read_bytes()
                            == graph_residuals.read_bytes()
                        ),
                        reconstruction_sha256=sha256(eager_reconstruction),
                        residuals_sha256=sha256(eager_residuals),
                    )
                    row["admission_pass"] = bool(
                        row["application_pass"]
                        and row["application_vs_fp32_pass"]
                        and row["emulation_status"]["finite"]
                        and row["emulation_status"]["positive_curvature"]
                        and row["emulation_status"]["completed_iterations"] == 10
                        and row["fp32_status"]["finite"]
                        and row["fp32_status"]["positive_curvature"]
                        and row["fp32_status"]["completed_iterations"] == 10
                        and row["eager_graph_reconstruction_bitwise"]
                        and row["eager_graph_residuals_bitwise"]
                    )
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

    expected = len(selections) if selections else 18
    if len(rows) != expected:
        raise RuntimeError(f"collected {len(rows)} rows, expected {expected}")
    by_acquisition = {}
    for acquisition in ACQUISITIONS:
        acquisition_rows = [
            row for row in rows if row["acquisition"] == acquisition
        ]
        if acquisition_rows:
            by_acquisition[acquisition] = {
                "rows": len(acquisition_rows),
                "admission_passes": sum(
                    bool(row["admission_pass"]) for row in acquisition_rows
                ),
                "strict_binary_passes": sum(
                    bool(row["strict_binary_pass"]) for row in acquisition_rows
                ),
                "worst_native_vs_fp32_complex_rel_l2": max(
                    row["native_vs_fp32_complex_rel_l2"]
                    for row in acquisition_rows
                ),
                "worst_native_vs_fp32_magnitude_nrmse": max(
                    row["native_vs_fp32_magnitude_nrmse"]
                    for row in acquisition_rows
                ),
                "minimum_native_vs_fp32_ssim": min(
                    row["native_vs_fp32_ssim"] for row in acquisition_rows
                ),
                "minimum_native_vs_fp32_psnr_db": min(
                    row["native_vs_fp32_psnr_db"] for row in acquisition_rows
                ),
                "worst_truth_nrmse_regression_vs_fp32": max(
                    row["native_truth_nrmse"] - row["fp32_truth_nrmse"]
                    for row in acquisition_rows
                ),
            }

    summary = {
        "experiment_id": "quality",
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "scale_rule": (
            {
                "kind": "fixed_truth_free",
                "raw_scale": float(args.raw_scale),
                "native_scale": float(args.raw_scale) / 512.0,
            }
            if args.raw_scale is not None
            else {
                "kind": "ledger",
                "path": str(scale_path),
                "sha256": sha256(scale_path),
            }
        ),
        "runtime_manifest": str(runtime_root / "manifest.json"),
        "runtime_manifest_sha256": sha256(runtime_root / "manifest.json"),
        "initial_gpu_snapshot": initial_gpu,
        "final_gpu_snapshot": gpu_snapshot(),
        "co_resident_quality_allowed": bool(args.allow_co_resident_quality),
        "launch_gpu_snapshots": launch_gpu_snapshots,
        "foreign_processes_observed": sorted(
            {
                process
                for launch in launch_gpu_snapshots
                for process in foreign_gpu_processes(launch["snapshot"])
            }
        ),
        "timing_admissible": False,
        "rows": rows,
        "by_acquisition": by_acquisition,
        "row_count": len(rows),
        "admission_passes": sum(bool(row["admission_pass"]) for row in rows),
        "strict_binary_passes": sum(
            bool(row["strict_binary_pass"]) for row in rows
        ),
        "all_admission_pass": all(row["admission_pass"] for row in rows),
        "all_eager_graph_bitwise": all(
            row["eager_graph_reconstruction_bitwise"]
            and row["eager_graph_residuals_bitwise"]
            for row in rows
        ),
        "worst_native_vs_fp32_complex_rel_l2": max(
            row["native_vs_fp32_complex_rel_l2"] for row in rows
        ),
        "worst_native_vs_fp32_magnitude_nrmse": max(
            row["native_vs_fp32_magnitude_nrmse"] for row in rows
        ),
        "minimum_native_vs_fp32_ssim": min(
            row["native_vs_fp32_ssim"] for row in rows
        ),
        "minimum_native_vs_fp32_psnr_db": min(
            row["native_vs_fp32_psnr_db"] for row in rows
        ),
        "worst_truth_nrmse_regression_vs_fp32": max(
            row["native_truth_nrmse"] - row["fp32_truth_nrmse"]
            for row in rows
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
