#!/usr/bin/env python3
"""MRI-NUFFT/cuFINUFFT ten-step CG reference for native-glue validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
import mrinufft


def rel_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(float(np.linalg.norm(reference)), 1e-30)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--operator-scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-reference", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--eps", type=float, default=5e-4)
    parser.add_argument("--method", type=int, default=1)
    args = parser.parse_args()

    trajectory = np.load(args.trajectory).reshape(-1, 2).astype(np.float32)
    samples = trajectory.shape[0]
    smaps_np = np.fromfile(
        args.runtime_dir / "sensitivity_maps.c64.bin", dtype=np.complex64
    ).reshape(8, 256, 256)
    measurement_np = np.fromfile(
        args.runtime_dir / "measurement.c64.bin", dtype=np.complex64
    ).reshape(8, samples)
    smaps = cp.asarray(smaps_np)
    measurement = cp.asarray(measurement_np).reshape(1, 8, samples)
    operator_cls = mrinufft.get_operator("cufinufft")
    operator = operator_cls(
        trajectory,
        (256, 256),
        n_coils=8,
        n_batchs=1,
        smaps=smaps,
        smaps_cached=True,
        density=False,
        squeeze_dims=False,
        n_trans=8,
        async_transfer=False,
        eps=args.eps,
        gpu_method=args.method,
        gpu_device_id=0,
    )
    # MRI-NUFFT reshapes these low-level caller-owned buffers on return.
    kspace = cp.empty((8, samples), dtype=cp.complex64)
    image = cp.empty((1, 256, 256), dtype=cp.complex64)

    def forward_raw(value: cp.ndarray) -> cp.ndarray:
        return operator.op(value, kspace) * np.float32(512.0)

    def adjoint_raw(value: cp.ndarray) -> cp.ndarray:
        image.fill(0)
        return operator.adj_op(value, image) * np.float32(512.0)

    inverse_scale_squared = np.float32(1.0 / (args.operator_scale**2))
    b = adjoint_raw(measurement).copy() * inverse_scale_squared
    x = cp.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rho = cp.vdot(r.ravel(), r.ravel()).real
    residuals = [float(cp.sqrt(cp.maximum(rho, 0.0)).get())]
    positive_curvature = True
    scaled_lambda = np.float32(1e-4) * inverse_scale_squared
    for _ in range(args.iterations):
        ap = (
            adjoint_raw(forward_raw(p)).copy() * inverse_scale_squared
            + scaled_lambda * p
        )
        denominator = cp.vdot(p.ravel(), ap.ravel()).real
        denominator_host = float(denominator.get())
        if not np.isfinite(denominator_host) or denominator_host <= 0.0:
            positive_curvature = False
            break
        alpha = rho / denominator
        x += alpha * p
        r -= alpha * ap
        rho_new = cp.vdot(r.ravel(), r.ravel()).real
        residuals.append(float(cp.sqrt(cp.maximum(rho_new, 0.0)).get()))
        p = r + (rho_new / cp.maximum(rho, np.float32(1e-30))) * p
        rho = rho_new
    cp.cuda.get_current_stream().synchronize()
    reference = cp.asnumpy(x[0, 0]).astype(np.complex64)
    args.save_reference.parent.mkdir(parents=True, exist_ok=True)
    reference.tofile(args.save_reference)
    native = np.fromfile(
        args.native_dir / "reconstruction.c64.bin", dtype=np.complex64
    ).reshape(256, 256)
    row = {
        "iterations": len(residuals) - 1,
        "finite": bool(np.isfinite(reference).all()),
        "positive_curvature": positive_curvature,
        "native_vs_mrinufft_cufinufft_complex_rel_l2": rel_l2(
            native, reference
        ),
        "reference_residuals": residuals,
        "parameters": {
            "eps": args.eps,
            "gpu_method": args.method,
            "n_trans": 8,
            "operator_scale": args.operator_scale,
            "normalization": "raw forward/adjoint; normal, rhs, and lambda divide by operator_scale^2",
        },
    }
    row["pass"] = bool(
        row["finite"]
        and row["positive_curvature"]
        and row["iterations"] == args.iterations
        and row["native_vs_mrinufft_cufinufft_complex_rel_l2"] <= 1e-4
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
