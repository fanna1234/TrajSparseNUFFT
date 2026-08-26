#!/usr/bin/env python3
"""Run the preregistered truth-free long-CG study on the hardest held-out row."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


ITERATIONS = (10, 20, 30, 50)
RAW_SCALE = 512.0
NATIVE_SCALE = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command: list[str], path: Path, label: str) -> subprocess.CompletedProcess:
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    path.write_text(
        "$ "
        + " ".join(command)
        + "\n\n[stdout]\n"
        + process.stdout
        + "\n[stderr]\n"
        + process.stderr
    )
    if process.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {process.returncode}")
    return process


def parse_payload(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("{"):
            payload = json.loads(line)
            if payload.get("correct") and payload.get("telemetry"):
                return payload
    raise RuntimeError("missing telemetry result")


def gpu_snapshot() -> dict:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
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
        "processes": [
            line for line in processes.stdout.splitlines() if line.strip()
        ],
    }


def assert_idle(snapshot: dict) -> None:
    allowed = ("gnome-remote-desktop-daemon", "Xorg")
    foreign = [
        row
        for row in snapshot["processes"]
        if not any(name in row for name in allowed)
    ]
    if foreign:
        raise RuntimeError("foreign GPU process: " + "; ".join(foreign))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    binary = args.binary.resolve()
    runtime = args.runtime_root.resolve() / "fs0005/frame02/golden65"
    pack = args.packed_root.resolve() / "golden65"
    case = args.cases_root.resolve() / "fs0005_v2"
    trajectory = (
        repo
        / "experiments/data_prep/trajectories/golden_256x256.npy"
    )
    validator = (
        repo
        / "experiments/production/validate_native_cg.py"
    )
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)

    initial_snapshot = gpu_snapshot()
    assert_idle(initial_snapshot)
    rows = []
    with open("/tmp/trajsparsenufft_gpu.lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for cg_iterations in ITERATIONS:
            snapshot = gpu_snapshot()
            assert_idle(snapshot)
            run_dir = output / f"cg{cg_iterations}"
            run_dir.mkdir(parents=True, exist_ok=False)
            command = [
                str(binary),
                str(pack / "G/real"),
                str(pack / "G/imag"),
                str(pack / "G_T/real"),
                str(pack / "G_T/imag"),
                str(runtime),
                repr(NATIVE_SCALE),
                "0",
                "1",
                "0",
                "0",
                str(cg_iterations),
                str(run_dir),
            ]
            native_process = run_logged(
                command, run_dir / "run.log", f"native cg{cg_iterations}"
            )
            native_payload = parse_payload(native_process.stdout)
            quality_path = run_dir / "quality.json"
            validation_command = [
                sys.executable,
                str(validator),
                "--repo-root",
                str(repo),
                "--case",
                str(case),
                "--frame",
                "2",
                "--runtime-dir",
                str(runtime),
                "--trajectory",
                str(trajectory),
                "--operator-scale",
                repr(NATIVE_SCALE),
                "--emulation-operator-scale",
                repr(RAW_SCALE),
                "--cg-iterations",
                str(cg_iterations),
                "--native-dir",
                str(run_dir),
                "--output",
                str(quality_path),
                "--save-emulation",
                str(run_dir / "emulation_reconstruction.c64.bin"),
                "--save-fp32",
                str(run_dir / "fp32_reconstruction.c64.bin"),
            ]
            run_logged(
                validation_command,
                run_dir / "validate.log",
                f"validation cg{cg_iterations}",
            )
            quality = json.loads(quality_path.read_text())
            denominators = np.fromfile(
                run_dir / "denominators.f32.bin", dtype=np.float32
            )
            alphas = np.fromfile(run_dir / "alphas.f32.bin", dtype=np.float32)
            if len(denominators) != cg_iterations or len(alphas) != cg_iterations:
                raise RuntimeError(f"incomplete telemetry for cg{cg_iterations}")
            denominator_pass = bool(
                np.isfinite(denominators).all() and np.all(denominators > 0.0)
            )
            alpha_pass = bool(np.isfinite(alphas).all())
            endpoint_pass = bool(
                quality["application_vs_fp32_pass"]
                and quality["finite"]
                and quality["emulation_status"]["finite"]
                and quality["emulation_status"]["positive_curvature"]
                and quality["emulation_status"]["completed_iterations"]
                == cg_iterations
                and quality["fp32_status"]["finite"]
                and quality["fp32_status"]["positive_curvature"]
                and quality["fp32_status"]["completed_iterations"]
                == cg_iterations
                and denominator_pass
                and alpha_pass
            )
            row = {
                "cg_iterations": cg_iterations,
                "native": native_payload,
                "quality": quality,
                "denominators": denominators.tolist(),
                "alphas": alphas.tolist(),
                "minimum_denominator": float(denominators.min()),
                "maximum_denominator": float(denominators.max()),
                "all_native_denominators_positive_finite": denominator_pass,
                "all_native_alphas_finite": alpha_pass,
                "endpoint_pass": endpoint_pass,
                "gpu_snapshot": snapshot,
                "reconstruction_sha256": sha256(
                    run_dir / "reconstruction.c64.bin"
                ),
                "residuals_sha256": sha256(run_dir / "residuals.f32.bin"),
                "denominators_sha256": sha256(
                    run_dir / "denominators.f32.bin"
                ),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "experiment_id": "frozen_evidence",
        "case": "fs0005/golden/frame2",
        "scale_rule": {
            "kind": "grid_width",
            "raw_scale": RAW_SCALE,
            "native_scale": NATIVE_SCALE,
            "truth_free": True,
        },
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "initial_gpu_snapshot": initial_snapshot,
        "final_gpu_snapshot": gpu_snapshot(),
        "rows": rows,
        "all_endpoints_pass": all(row["endpoint_pass"] for row in rows),
        "all_native_denominators_positive_finite": all(
            row["all_native_denominators_positive_finite"] for row in rows
        ),
        "minimum_native_vs_fp32_ssim": min(
            row["quality"]["native_vs_fp32_ssim"] for row in rows
        ),
        "minimum_native_vs_fp32_psnr_db": min(
            row["quality"]["native_vs_fp32_psnr_db"] for row in rows
        ),
        "worst_native_vs_fp32_magnitude_nrmse": max(
            row["quality"]["native_vs_fp32_magnitude_nrmse"] for row in rows
        ),
        "worst_truth_nrmse_regression_vs_fp32": max(
            row["quality"]["native_truth_nrmse"]
            - row["quality"]["fp32_truth_nrmse"]
            for row in rows
        ),
    }
    if not math.isfinite(summary["worst_native_vs_fp32_magnitude_nrmse"]):
        raise RuntimeError("nonfinite aggregate quality")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
