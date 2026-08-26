#!/usr/bin/env python3
"""Compare native FP16x2 CG with deterministic Torch sparse emulation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torchkbnufft as tkbn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def rel_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(float(np.linalg.norm(reference)), 1e-30)
    )


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("mixed_precision_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--operator-scale", type=float, required=True)
    parser.add_argument("--emulation-operator-scale", type=float)
    parser.add_argument("--cg-iterations", type=int, default=10)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-emulation", type=Path)
    parser.add_argument("--save-fp32", type=Path)
    args = parser.parse_args()
    if args.cg_iterations <= 0:
        raise ValueError("cg-iterations must be positive")

    runner = load_runner(
        args.repo_root
        / "experiments/production"
        / "reference_sparse_solver.py"
    )
    device = torch.device("cpu")
    smaps_np = np.load(args.case / "sensitivity_maps.npy").astype(np.complex64)
    truth_np = np.load(args.case / "reference_rss.npy").astype(np.float32)
    trajectory = np.load(args.trajectory).reshape(-1, 2).astype(np.float32)
    samples = trajectory.shape[0]
    measurement_np = np.fromfile(
        args.runtime_dir / "measurement.c64.bin", dtype=np.complex64
    ).reshape(8, samples)
    omega_cpu = torch.from_numpy(
        (trajectory.T * (2.0 * np.pi)).astype(np.float32)
    )
    real_matrix, imag_matrix = tkbn.calc_tensor_spmatrix(
        omega_cpu, (256, 256), grid_size=(512, 512), numpoints=6
    )
    forward_op = tkbn.KbNufft(
        im_size=(256, 256), grid_size=(512, 512), numpoints=6, dtype=torch.float32
    )
    adjoint_op = tkbn.KbNufftAdjoint(
        im_size=(256, 256), grid_size=(512, 512), numpoints=6, dtype=torch.float32
    )
    smaps = torch.from_numpy(smaps_np[args.frame : args.frame + 1])
    measurement = torch.from_numpy(measurement_np).unsqueeze(0)
    forward, adjoint = runner.make_component_operator(
        forward_op,
        adjoint_op,
        real_matrix,
        imag_matrix,
        smaps,
        "fp16",
        "split2",
        8,
        512,
        device,
        "csr",
    )
    scale = args.emulation_operator_scale or args.operator_scale
    reconstruction, residuals, status = runner.cg_sense(
        lambda value: forward(value) / scale,
        lambda value: adjoint(value) / scale,
        measurement / scale,
        args.cg_iterations,
        1e-4 / (scale * scale),
        "native-validation",
        False,
    )
    forward_fp32, adjoint_fp32 = runner.make_component_operator(
        forward_op,
        adjoint_op,
        real_matrix,
        imag_matrix,
        smaps,
        "fp32",
        "fp32",
        8,
        512,
        device,
        "csr",
    )
    fp32_reconstruction, fp32_residuals, fp32_status = runner.cg_sense(
        lambda value: forward_fp32(value) / scale,
        lambda value: adjoint_fp32(value) / scale,
        measurement / scale,
        args.cg_iterations,
        1e-4 / (scale * scale),
        "native-fp32-validation",
        False,
    )
    reference = reconstruction[0, 0].detach().numpy()
    fp32_reference = fp32_reconstruction[0, 0].detach().numpy()
    if args.save_emulation is not None:
        args.save_emulation.parent.mkdir(parents=True, exist_ok=True)
        np.ascontiguousarray(reference.astype(np.complex64)).tofile(
            args.save_emulation
        )
    if args.save_fp32 is not None:
        args.save_fp32.parent.mkdir(parents=True, exist_ok=True)
        np.ascontiguousarray(fp32_reference.astype(np.complex64)).tofile(
            args.save_fp32
        )
    native = np.fromfile(
        args.native_dir / "reconstruction.c64.bin", dtype=np.complex64
    ).reshape(256, 256)
    native_residuals = np.fromfile(
        args.native_dir / "residuals.f32.bin", dtype=np.float32
    )
    truth = truth_np[args.frame]
    reference_magnitude = np.abs(reference)
    fp32_magnitude = np.abs(fp32_reference)
    native_magnitude = np.abs(native)
    data_range = max(
        float(reference_magnitude.max() - reference_magnitude.min()), 1e-12
    )
    row = {
        "experiment_id": "production",
        "frame": args.frame,
        "cg_iterations": args.cg_iterations,
        "samples": samples,
        "operator_scale": scale,
        "native_operator_scale": args.operator_scale,
        "emulation_status": status,
        "fp32_status": fp32_status,
        "native_vs_emulation_complex_rel_l2": rel_l2(native, reference),
        "native_vs_emulation_magnitude_nrmse": rel_l2(
            native_magnitude, reference_magnitude
        ),
        "native_vs_emulation_ssim": float(
            structural_similarity(
                reference_magnitude, native_magnitude, data_range=data_range
            )
        ),
        "native_vs_emulation_psnr_db": float(
            peak_signal_noise_ratio(
                reference_magnitude, native_magnitude, data_range=data_range
            )
        ),
        "native_vs_fp32_complex_rel_l2": rel_l2(native, fp32_reference),
        "native_vs_fp32_magnitude_nrmse": rel_l2(native_magnitude, fp32_magnitude),
        "native_vs_fp32_ssim": float(
            structural_similarity(fp32_magnitude, native_magnitude, data_range=max(float(fp32_magnitude.max() - fp32_magnitude.min()), 1e-12))
        ),
        "native_vs_fp32_psnr_db": float(
            peak_signal_noise_ratio(fp32_magnitude, native_magnitude, data_range=max(float(fp32_magnitude.max() - fp32_magnitude.min()), 1e-12))
        ),
        "emulation_truth_nrmse": rel_l2(reference_magnitude, truth),
        "native_truth_nrmse": rel_l2(native_magnitude, truth),
        "fp32_truth_nrmse": rel_l2(fp32_magnitude, truth),
        "native_residuals": native_residuals.tolist(),
        "emulation_residuals": residuals,
        "fp32_residuals": fp32_residuals,
        "finite": bool(np.isfinite(native).all()),
    }
    row["strict_binary_pass"] = bool(
        row["finite"]
        and row["native_vs_emulation_complex_rel_l2"] <= 2e-5
        and row["native_vs_emulation_magnitude_nrmse"] <= 2e-5
    )
    row["application_pass"] = bool(
        row["finite"]
        and row["native_vs_emulation_ssim"] >= 0.9999
        and row["native_vs_emulation_psnr_db"] >= 60.0
        and row["native_truth_nrmse"] - row["emulation_truth_nrmse"] <= 0.002
    )
    row["application_vs_fp32_pass"] = bool(
        row["finite"]
        and row["native_vs_fp32_ssim"] >= 0.9999
        and row["native_vs_fp32_psnr_db"] >= 60.0
        and row["native_truth_nrmse"] - row["fp32_truth_nrmse"] <= 0.002
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
