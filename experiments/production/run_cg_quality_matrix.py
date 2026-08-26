#!/usr/bin/env python3
"""Run native eager/Graph CG quality over the frozen 3x3 OCMR matrix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    packed_root = args.packed_root.resolve()
    case_root = args.case.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    experiment = repo / "experiments/production"
    validator = experiment / "validate_native_cg.py"
    trajectory_root = (
        repo / "experiments/data_prep/trajectories"
    )
    rows = []
    with open("/tmp/trajsparsenufft_gpu.lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for trajectory in ("spiral", "radial", "golden"):
            pack = packed_root / f"{trajectory}65"
            trajectory_path = trajectory_root / TRAJECTORY_FILES[trajectory]
            for frame in range(3):
                raw_scale = (
                    args.raw_scale
                    if args.raw_scale is not None
                    else RAW_SCALES[(trajectory, frame)]
                )
                native_scale = raw_scale / 512.0
                runtime = runtime_root / f"frame{frame:02d}" / f"{trajectory}65"
                mode_dirs = {}
                for graph in (0, 1):
                    mode = "graph" if graph else "eager"
                    mode_dir = output / f"{trajectory}_frame{frame}_{mode}"
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
                    process = subprocess.run(
                        command, text=True, capture_output=True, check=False
                    )
                    (mode_dir / "run.log").write_text(
                        "$ "
                        + " ".join(command)
                        + "\n\n[stdout]\n"
                        + process.stdout
                        + "\n[stderr]\n"
                        + process.stderr
                    )
                    if process.returncode != 0:
                        raise RuntimeError(
                            f"native failure {trajectory}/frame{frame}/{mode}"
                        )
                    mode_dirs[mode] = mode_dir

                validation_path = mode_dirs["eager"] / "quality.json"
                validate_command = [
                    sys.executable,
                    str(validator),
                    "--repo-root",
                    str(repo),
                    "--case",
                    str(case_root),
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
                ]
                validation = subprocess.run(
                    validate_command, text=True, capture_output=True, check=False
                )
                (mode_dirs["eager"] / "validate.log").write_text(
                    "$ "
                    + " ".join(validate_command)
                    + "\n\n[stdout]\n"
                    + validation.stdout
                    + "\n[stderr]\n"
                    + validation.stderr
                )
                if validation.returncode != 0:
                    raise RuntimeError(
                        f"validation failure {trajectory}/frame{frame}"
                    )
                row = json.loads(validation_path.read_text())
                eager_reconstruction = mode_dirs["eager"] / "reconstruction.c64.bin"
                graph_reconstruction = mode_dirs["graph"] / "reconstruction.c64.bin"
                eager_residuals = mode_dirs["eager"] / "residuals.f32.bin"
                graph_residuals = mode_dirs["graph"] / "residuals.f32.bin"
                row.update(
                    trajectory=trajectory,
                    eager_graph_reconstruction_bitwise=(
                        eager_reconstruction.read_bytes()
                        == graph_reconstruction.read_bytes()
                    ),
                    eager_graph_residuals_bitwise=(
                        eager_residuals.read_bytes() == graph_residuals.read_bytes()
                    ),
                    reconstruction_sha256=sha256(eager_reconstruction),
                    residuals_sha256=sha256(eager_residuals),
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "experiment_id": "production",
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "scale_rule": (
            {
                "kind": "fixed_truth_free",
                "raw_scale": args.raw_scale,
                "native_scale": args.raw_scale / 512.0,
            }
            if args.raw_scale is not None
            else {"kind": "legacy_per_row"}
        ),
        "rows": rows,
        "all_application_pass": all(row["application_pass"] for row in rows),
        "all_strict_binary_pass": all(row["strict_binary_pass"] for row in rows),
        "all_eager_graph_bitwise": all(
            row["eager_graph_reconstruction_bitwise"]
            and row["eager_graph_residuals_bitwise"]
            for row in rows
        ),
        "worst_complex_rel_l2": max(
            row["native_vs_emulation_complex_rel_l2"] for row in rows
        ),
        "worst_magnitude_nrmse": max(
            row["native_vs_emulation_magnitude_nrmse"] for row in rows
        ),
        "worst_truth_regression": max(
            row["native_truth_nrmse"] - row["emulation_truth_nrmse"]
            for row in rows
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
