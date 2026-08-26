#!/usr/bin/env python3
"""Run native cuFINUFFT CG over the frozen 3x3 development quality matrix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


RAW_SCALES = {
    ("spiral", 0): 1185.363037109375,
    ("spiral", 1): 1167.283447265625,
    ("spiral", 2): 1173.318359375,
    ("radial", 0): 1187.88720703125,
    ("radial", 1): 1169.583984375,
    ("radial", 2): 1175.6646728515625,
    ("golden", 0): 1187.2781982421875,
    ("golden", 1): 1169.1214599609375,
    ("golden", 2): 1175.176025390625,
}
TRAJECTORY_NPY = {
    "spiral": "spiral_standard_256.npy",
    "radial": "radial_256x256.npy",
    "golden": "golden_256x256.npy",
}
TRAJECTORY_BIN = {
    "spiral": "experiments/data_prep/trajectories/spiral_standard_256.f32xy.bin",
    "radial": "experiments/data_prep/trajectories/radial_256x256.f32xy.bin",
    "golden": "experiments/data_prep/trajectories/golden_256x256.f32xy.bin",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_result(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("{"):
            payload = json.loads(line)
            if payload.get("variant") == "cufinufft_native_cg10":
                return payload
    raise RuntimeError("missing native cuFINUFFT result")


def run_logged(command: list[str], path: Path, label: str) -> subprocess.CompletedProcess:
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    path.write_text(
        "$ " + " ".join(command) + "\n\n[stdout]\n" + process.stdout
        + "\n[stderr]\n" + process.stderr
    )
    if process.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {process.returncode}")
    return process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", type=int, default=2)
    parser.add_argument("--forward-method", type=int)
    parser.add_argument("--adjoint-method", type=int)
    parser.add_argument("--gpu-sort", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--raw-scale",
        type=float,
        help="optional truth-free raw scale shared by every row",
    )
    args = parser.parse_args()
    if args.raw_scale is not None and args.raw_scale <= 0.0:
        raise ValueError("raw-scale must be positive")

    repo = args.repo_root.resolve()
    binary = args.binary.resolve()
    runtime_root = args.runtime_root.resolve()
    case_root = args.case.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    trajectory_root = (
        repo / "experiments/data_prep/trajectories"
    )
    validator = (
        repo
        / "experiments/production"
        / "validate_native_cg.py"
    )
    rows = []
    with open("/tmp/trajsparsenufft_gpu.lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for trajectory in ("spiral", "radial", "golden"):
            trajectory_bin = repo / TRAJECTORY_BIN[trajectory]
            trajectory_npy = trajectory_root / TRAJECTORY_NPY[trajectory]
            for frame in range(3):
                runtime = runtime_root / f"frame{frame:02d}" / f"{trajectory}65"
                replicas = []
                for replica in range(2):
                    replica_dir = output / f"{trajectory}_frame{frame}_run{replica}"
                    replica_dir.mkdir(parents=True, exist_ok=False)
                    forward_method = args.forward_method or args.method
                    adjoint_method = args.adjoint_method or args.method
                    command = ["env"]
                    if args.raw_scale is not None:
                        command.append(
                            f"JKF_CUFINUFFT_OPERATOR_SCALE={args.raw_scale}"
                        )
                    command.extend([
                        f"JKF_CUFINUFFT_FORWARD_METHOD={forward_method}",
                        f"JKF_CUFINUFFT_ADJOINT_METHOD={adjoint_method}",
                        f"JKF_CUFINUFFT_GPU_SORT={args.gpu_sort}",
                        str(binary), str(trajectory_bin), str(runtime),
                        "0", "1", "0", "10", str(args.method), "5e-4",
                        str(replica_dir),
                    ])
                    process = run_logged(
                        command,
                        replica_dir / "run.log",
                        f"{trajectory}/frame{frame}/run{replica}",
                    )
                    replicas.append((replica_dir, parse_result(process.stdout)))

                primary, payload = replicas[0]
                raw_scale = (
                    args.raw_scale
                    if args.raw_scale is not None
                    else RAW_SCALES[(trajectory, frame)]
                )
                replica_quality = []
                for replica_index, (replica_dir, replica_payload) in enumerate(
                    replicas
                ):
                    validation_path = replica_dir / "quality.json"
                    validation_command = [
                        sys.executable,
                        str(validator),
                        "--repo-root", str(repo),
                        "--case", str(case_root),
                        "--frame", str(frame),
                        "--runtime-dir", str(runtime),
                        "--trajectory", str(trajectory_npy),
                        "--operator-scale",
                        repr(float(replica_payload["operator_scale"])),
                        "--emulation-operator-scale", repr(raw_scale),
                        "--native-dir", str(replica_dir),
                        "--output", str(validation_path),
                    ]
                    if replica_index == 0:
                        validation_command.extend(
                            [
                                "--save-fp32",
                                str(primary / "fp32_reconstruction.c64.bin"),
                            ]
                        )
                    run_logged(
                        validation_command,
                        replica_dir / "validate.log",
                        f"validation {trajectory}/frame{frame}/run{replica_index}",
                    )
                    replica_quality.append(json.loads(validation_path.read_text()))

                row = dict(replica_quality[0])
                first_reconstruction = primary / "reconstruction.c64.bin"
                second_reconstruction = replicas[1][0] / "reconstruction.c64.bin"
                first_residuals = primary / "residuals.f32.bin"
                second_residuals = replicas[1][0] / "residuals.f32.bin"
                reconstruction_a = np.fromfile(
                    first_reconstruction, dtype=np.complex64
                )
                reconstruction_b = np.fromfile(
                    second_reconstruction, dtype=np.complex64
                )
                residuals_a = np.fromfile(first_residuals, dtype=np.float32)
                residuals_b = np.fromfile(second_residuals, dtype=np.float32)
                repeat_rel_l2 = float(
                    np.linalg.norm(reconstruction_a - reconstruction_b)
                    / max(float(np.linalg.norm(reconstruction_a)), 1e-30)
                )
                row.update(
                    trajectory=trajectory,
                    method=args.method,
                    baseline_payload=payload,
                    replica_quality=replica_quality,
                    repeat_reconstruction_bitwise=(
                        first_reconstruction.read_bytes()
                        == second_reconstruction.read_bytes()
                    ),
                    repeat_residuals_bitwise=(
                        first_residuals.read_bytes() == second_residuals.read_bytes()
                    ),
                    repeat_reconstruction_complex_rel_l2=repeat_rel_l2,
                    repeat_residual_max_abs=float(
                        np.max(np.abs(residuals_a - residuals_b))
                    ),
                    reconstruction_sha256=sha256(first_reconstruction),
                    residuals_sha256=sha256(first_residuals),
                )
                row["baseline_application_pass"] = bool(
                    all(q["application_vs_fp32_pass"] for q in replica_quality)
                    and repeat_rel_l2 <= 0.002
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    quality_rows = [q for row in rows for q in row["replica_quality"]]
    summary = {
        "experiment_id": "cufinufft_baseline",
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "method": args.method,
        "forward_method": args.forward_method or args.method,
        "adjoint_method": args.adjoint_method or args.method,
        "gpu_sort": args.gpu_sort,
        "scale_rule": (
            {
                "kind": "fixed_truth_free",
                "raw_scale": args.raw_scale,
            }
            if args.raw_scale is not None
            else {"kind": "legacy_truth_calibration"}
        ),
        "rows": rows,
        "row_count": len(rows),
        "replica_row_count": len(quality_rows),
        "replica_application_passes": sum(
            q["application_vs_fp32_pass"] for q in quality_rows
        ),
        "application_passes": sum(r["baseline_application_pass"] for r in rows),
        "all_application_pass": all(r["baseline_application_pass"] for r in rows),
        "all_repeat_bitwise": all(
            r["repeat_reconstruction_bitwise"] and r["repeat_residuals_bitwise"]
            for r in rows
        ),
        "all_repeat_stable": all(
            r["repeat_reconstruction_complex_rel_l2"] <= 0.002 for r in rows
        ),
        "worst_repeat_reconstruction_complex_rel_l2": max(
            r["repeat_reconstruction_complex_rel_l2"] for r in rows
        ),
        "worst_native_vs_fp32_complex_rel_l2": max(
            r["native_vs_fp32_complex_rel_l2"] for r in quality_rows
        ),
        "worst_native_vs_fp32_magnitude_nrmse": max(
            r["native_vs_fp32_magnitude_nrmse"] for r in quality_rows
        ),
        "minimum_native_vs_fp32_ssim": min(
            r["native_vs_fp32_ssim"] for r in quality_rows
        ),
        "minimum_native_vs_fp32_psnr_db": min(
            r["native_vs_fp32_psnr_db"] for r in quality_rows
        ),
        "worst_truth_nrmse_regression_vs_fp32": max(
            r["native_truth_nrmse"] - r["fp32_truth_nrmse"]
            for r in quality_rows
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
