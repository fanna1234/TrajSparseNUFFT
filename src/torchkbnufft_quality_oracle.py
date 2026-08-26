#!/usr/bin/env python3
"""Quality oracle for TorchKbNufft sparse matrices and FP16 interpolation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchkbnufft as tkbn
from scipy.sparse import coo_matrix
from torchkbnufft._nufft.fft import fft_and_scale, ifft_and_scale


def relative_l2(candidate, reference) -> float:
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    im_size = (256, 256)
    grid_size = (512, 512)
    coils = 8
    trajectory = np.load(args.trajectory_file).reshape(-1, 2).astype(np.float32)
    omega = torch.from_numpy((trajectory.T * 2.0 * np.pi).astype(np.float32))

    started = time.perf_counter()
    real_tensor, imag_tensor = tkbn.calc_tensor_spmatrix(
        omega, im_size, grid_size=grid_size, numpoints=6
    )
    real_tensor = real_tensor.coalesce()
    imag_tensor = imag_tensor.coalesce()
    spmatrix_build_seconds = time.perf_counter() - started
    indices = real_tensor.indices().cpu().numpy()
    real_values = real_tensor.values().cpu().numpy().astype(np.float32)
    imag_values = imag_tensor.values().cpu().numpy().astype(np.float32)
    shape = tuple(map(int, real_tensor.shape))
    ar = coo_matrix((real_values, (indices[0], indices[1])), shape=shape).tocsr()
    ai = coo_matrix((imag_values, (indices[0], indices[1])), shape=shape).tocsr()
    ar_q = coo_matrix(
        (real_values.astype(np.float16).astype(np.float32), (indices[0], indices[1])),
        shape=shape,
    ).tocsr()
    ai_q = coo_matrix(
        (imag_values.astype(np.float16).astype(np.float32), (indices[0], indices[1])),
        shape=shape,
    ).tocsr()

    forward = tkbn.KbNufft(
        im_size=im_size, grid_size=grid_size, numpoints=6, dtype=torch.float32
    )
    adjoint = tkbn.KbNufftAdjoint(
        im_size=im_size, grid_size=grid_size, numpoints=6, dtype=torch.float32
    )
    image = torch.randn(1, coils, *im_size, dtype=torch.complex64)
    samples = torch.randn(1, coils, shape[0], dtype=torch.complex64)
    forward_sp = forward(image, omega, (real_tensor, imag_tensor))
    forward_table = forward(image, omega)
    adjoint_sp = adjoint(samples, omega, (real_tensor, imag_tensor))
    adjoint_table = adjoint(samples, omega)
    left = torch.vdot(forward_sp.flatten(), samples.flatten())
    right = torch.vdot(image.flatten(), adjoint_sp.flatten())
    spmatrix_adjoint_defect = float(
        torch.abs(left - right)
        / (torch.linalg.vector_norm(forward_sp) * torch.linalg.vector_norm(samples))
    )

    grid = fft_and_scale(
        image=image,
        scaling_coef=forward.scaling_coef,
        im_size=forward.im_size,
        grid_size=forward.grid_size,
        norm=None,
    )
    grid_np = grid.detach().cpu().numpy().reshape(coils, -1).T
    grid_q = (
        grid_np.real.astype(np.float16).astype(np.float32)
        + 1j * grid_np.imag.astype(np.float16).astype(np.float32)
    )
    forward_q = (ar_q @ grid_q.real - ai_q @ grid_q.imag) + 1j * (
        ar_q @ grid_q.imag + ai_q @ grid_q.real
    )
    forward_reference = forward_sp.detach().cpu().numpy().reshape(coils, -1).T

    samples_np = samples.detach().cpu().numpy().reshape(coils, -1).T
    samples_q = (
        samples_np.real.astype(np.float16).astype(np.float32)
        + 1j * samples_np.imag.astype(np.float16).astype(np.float32)
    )
    adjoint_grid_q = (ar_q.T @ samples_q.real + ai_q.T @ samples_q.imag) + 1j * (
        ar_q.T @ samples_q.imag - ai_q.T @ samples_q.real
    )
    adjoint_grid_tensor = torch.from_numpy(
        adjoint_grid_q.T.reshape(1, coils, *grid_size).astype(np.complex64)
    )
    adjoint_q = ifft_and_scale(
        image=adjoint_grid_tensor,
        scaling_coef=adjoint.scaling_coef,
        im_size=adjoint.im_size,
        grid_size=adjoint.grid_size,
        norm=None,
    ).detach().cpu().numpy()
    adjoint_reference = adjoint_sp.detach().cpu().numpy()

    interpolation_x = rng.standard_normal((shape[1], coils), dtype=np.float32) + 1j * rng.standard_normal(
        (shape[1], coils), dtype=np.float32
    )
    interpolation_xq = (
        interpolation_x.real.astype(np.float16).astype(np.float32)
        + 1j * interpolation_x.imag.astype(np.float16).astype(np.float32)
    )
    interp_ref = (ar @ interpolation_x.real - ai @ interpolation_x.imag) + 1j * (
        ar @ interpolation_x.imag + ai @ interpolation_x.real
    )
    interp_q = (ar_q @ interpolation_xq.real - ai_q @ interpolation_xq.imag) + 1j * (
        ar_q @ interpolation_xq.imag + ai_q @ interpolation_xq.real
    )

    result = {
        "experiment_id": "trajsparsenufft_structural_oracle",
        "evidence_class": "partial",
        "torchkbnufft_version": getattr(tkbn, "__version__", "unknown"),
        "trajectory_file": str(args.trajectory_file),
        "im_size": list(im_size),
        "grid_size": list(grid_size),
        "coils": coils,
        "samples": shape[0],
        "nnz": len(real_values),
        "spmatrix_build_seconds": spmatrix_build_seconds,
        "forward_spmatrix_vs_table_rel_l2": float(
            torch.linalg.vector_norm(forward_sp - forward_table)
            / torch.linalg.vector_norm(forward_table)
        ),
        "adjoint_spmatrix_vs_table_rel_l2": float(
            torch.linalg.vector_norm(adjoint_sp - adjoint_table)
            / torch.linalg.vector_norm(adjoint_table)
        ),
        "spmatrix_adjoint_defect": spmatrix_adjoint_defect,
        "random_grid_fp16_interpolation_rel_l2": relative_l2(interp_q, interp_ref),
        "full_forward_fp16_interpolation_rel_l2": relative_l2(
            forward_q, forward_reference
        ),
        "full_adjoint_fp16_interpolation_rel_l2": relative_l2(
            adjoint_q, adjoint_reference
        ),
        "real_values_underflow_to_zero_fp16": int(
            np.count_nonzero(real_values)
            - np.count_nonzero(real_values.astype(np.float16))
        ),
        "imag_values_underflow_to_zero_fp16": int(
            np.count_nonzero(imag_values)
            - np.count_nonzero(imag_values.astype(np.float16))
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
