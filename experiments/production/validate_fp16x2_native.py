#!/usr/bin/env python3
"""Validate the native FP16x2 binary against the Torch sparse arithmetic model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchkbnufft as tkbn
from torchkbnufft._nufft.fft import fft_and_scale, ifft_and_scale


def rel_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(float(np.linalg.norm(reference)), 1e-30)
    )


def quantized(matrix: torch.Tensor, device: torch.device) -> torch.Tensor:
    matrix = matrix.coalesce()
    return torch.sparse_coo_tensor(
        matrix.indices().to(device),
        matrix.values().to(device).half().float(),
        matrix.shape,
        device=device,
    ).coalesce()


def split2(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    high = value.half().float()
    residual = (value - high).half().float()
    return high, residual


def sparse_complex_forward(
    real_q: torch.Tensor,
    imag_q: torch.Tensor,
    real_parts: tuple[torch.Tensor, torch.Tensor],
    imag_parts: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    output_real = sum(torch.sparse.mm(real_q, part) for part in real_parts)
    output_real -= sum(torch.sparse.mm(imag_q, part) for part in imag_parts)
    output_imag = sum(torch.sparse.mm(real_q, part) for part in imag_parts)
    output_imag += sum(torch.sparse.mm(imag_q, part) for part in real_parts)
    return torch.complex(output_real, output_imag)


def sparse_complex_adjoint(
    real_q_t: torch.Tensor,
    imag_q_t: torch.Tensor,
    real_parts: tuple[torch.Tensor, torch.Tensor],
    imag_parts: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    output_real = sum(torch.sparse.mm(real_q_t, part) for part in real_parts)
    output_real += sum(torch.sparse.mm(imag_q_t, part) for part in imag_parts)
    output_imag = sum(torch.sparse.mm(real_q_t, part) for part in imag_parts)
    output_imag -= sum(torch.sparse.mm(imag_q_t, part) for part in real_parts)
    return torch.complex(output_real, output_imag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    trajectory = np.load(args.trajectory).reshape(-1, 2).astype(np.float32)
    samples = trajectory.shape[0]
    image_np = np.fromfile(
        args.runtime_dir / "forward_coils.c64.bin", dtype=np.complex64
    ).reshape(8, 256, 256)
    measurement_np = np.fromfile(
        args.runtime_dir / "measurement.c64.bin", dtype=np.complex64
    ).reshape(8, samples)

    omega_cpu = torch.from_numpy(
        (trajectory.T * (2.0 * np.pi)).astype(np.float32)
    )
    forward_op = tkbn.KbNufft(
        im_size=(256, 256),
        grid_size=(512, 512),
        numpoints=6,
        dtype=torch.float32,
    ).to(device)
    adjoint_op = tkbn.KbNufftAdjoint(
        im_size=(256, 256),
        grid_size=(512, 512),
        numpoints=6,
        dtype=torch.float32,
    ).to(device)
    inv_norm_factor = 1.0 / 512.0
    real_matrix, imag_matrix = tkbn.calc_tensor_spmatrix(
        omega_cpu, (256, 256), grid_size=(512, 512), numpoints=6
    )
    real_q = quantized(real_matrix, device)
    imag_q = quantized(imag_matrix, device)
    real_q_t = real_q.transpose(0, 1).coalesce()
    imag_q_t = imag_q.transpose(0, 1).coalesce()

    image = torch.from_numpy(image_np).to(device)
    measurement = torch.from_numpy(measurement_np).to(device)
    with torch.no_grad():
        grid = fft_and_scale(
            image=image.unsqueeze(0),
            scaling_coef=forward_op.scaling_coef,
            im_size=forward_op.im_size,
            grid_size=forward_op.grid_size,
            norm=None,
        )
        dense = grid.reshape(8, -1).T
        forward_real_parts = split2(dense.real)
        forward_imag_parts = split2(dense.imag)
        arithmetic_forward = sparse_complex_forward(
            real_q, imag_q, forward_real_parts, forward_imag_parts
        ).T
        arithmetic_forward *= inv_norm_factor

        dense_y = measurement.T
        adjoint_real_parts = split2(dense_y.real)
        adjoint_imag_parts = split2(dense_y.imag)
        arithmetic_grid = sparse_complex_adjoint(
            real_q_t, imag_q_t, adjoint_real_parts, adjoint_imag_parts
        ).T.reshape(1, 8, 512, 512)
        arithmetic_adjoint = ifft_and_scale(
            image=arithmetic_grid,
            scaling_coef=adjoint_op.scaling_coef,
            im_size=adjoint_op.im_size,
            grid_size=adjoint_op.grid_size,
            norm=None,
        ).squeeze(0)
        arithmetic_adjoint *= inv_norm_factor
        torch.cuda.synchronize()

    native_forward = np.fromfile(
        args.dump_dir / "forward_samples.c64.bin", dtype=np.complex64
    ).reshape(8, samples)
    native_adjoint = np.fromfile(
        args.dump_dir / "adjoint_image.c64.bin", dtype=np.complex64
    ).reshape(8, 256, 256)
    forward_high = np.fromfile(
        args.dump_dir / "forward_dense.f16.bin", dtype=np.float16
    ).reshape(512 * 512, 16)
    forward_residual = np.fromfile(
        args.dump_dir / "forward_dense_residual.f16.bin", dtype=np.float16
    ).reshape(512 * 512, 16)
    forward_fp32_endpoint = np.fromfile(
        args.dump_dir / "forward_fp32_endpoint.c64.bin", dtype=np.complex64
    ).reshape(8, 512, 512)
    adjoint_high = np.fromfile(
        args.dump_dir / "adjoint_dense.f16.bin", dtype=np.float16
    ).reshape(samples, 16)
    adjoint_residual = np.fromfile(
        args.dump_dir / "adjoint_dense_residual.f16.bin", dtype=np.float16
    ).reshape(samples, 16)

    exact_forward_panel = np.concatenate(
        [dense.real.detach().cpu().numpy(), dense.imag.detach().cpu().numpy()], axis=1
    ).astype(np.float32)
    native_forward_panel_complex = forward_fp32_endpoint.reshape(8, -1).T
    native_forward_panel = np.concatenate(
        [native_forward_panel_complex.real, native_forward_panel_complex.imag], axis=1
    ).astype(np.float32)
    exact_adjoint_panel = np.concatenate(
        [dense_y.real.detach().cpu().numpy(), dense_y.imag.detach().cpu().numpy()],
        axis=1,
    ).astype(np.float32)
    expected_adjoint_high = exact_adjoint_panel.astype(np.float16)
    expected_adjoint_residual = (
        exact_adjoint_panel - expected_adjoint_high.astype(np.float32)
    ).astype(np.float16)

    arithmetic_forward_np = arithmetic_forward.detach().cpu().numpy()
    arithmetic_adjoint_np = arithmetic_adjoint.detach().cpu().numpy()
    lhs = np.vdot(
        native_forward.reshape(-1).astype(np.complex128),
        measurement_np.reshape(-1).astype(np.complex128),
    )
    rhs = np.vdot(
        image_np.reshape(-1).astype(np.complex128),
        native_adjoint.reshape(-1).astype(np.complex128),
    )
    arithmetic_lhs = np.vdot(
        arithmetic_forward_np.reshape(-1).astype(np.complex128),
        measurement_np.reshape(-1).astype(np.complex128),
    )
    arithmetic_rhs = np.vdot(
        image_np.reshape(-1).astype(np.complex128),
        arithmetic_adjoint_np.reshape(-1).astype(np.complex128),
    )
    row = {
        "experiment_id": "production",
        "samples": samples,
        "forward_split_panel_vs_fp32_rel_l2": rel_l2(
            forward_high.astype(np.float32) + forward_residual.astype(np.float32),
            native_forward_panel,
        ),
        "native_fp32_endpoint_vs_torch_rel_l2": rel_l2(
            native_forward_panel, exact_forward_panel
        ),
        "adjoint_high_bitwise_equal": bool(
            np.array_equal(adjoint_high.view(np.uint16), expected_adjoint_high.view(np.uint16))
        ),
        "adjoint_residual_bitwise_equal": bool(
            np.array_equal(
                adjoint_residual.view(np.uint16),
                expected_adjoint_residual.view(np.uint16),
            )
        ),
        "adjoint_split_panel_vs_fp32_rel_l2": rel_l2(
            adjoint_high.astype(np.float32) + adjoint_residual.astype(np.float32),
            exact_adjoint_panel,
        ),
        "native_forward_vs_arithmetic_rel_l2": rel_l2(
            native_forward, arithmetic_forward_np
        ),
        "native_adjoint_vs_arithmetic_rel_l2": rel_l2(
            native_adjoint, arithmetic_adjoint_np
        ),
        "native_adjoint_defect": float(
            abs(lhs - rhs)
            / max(float(np.linalg.norm(native_forward) * np.linalg.norm(measurement_np)), 1e-30)
        ),
        "arithmetic_adjoint_defect": float(
            abs(arithmetic_lhs - arithmetic_rhs)
            / max(
                float(
                    np.linalg.norm(arithmetic_forward_np)
                    * np.linalg.norm(measurement_np)
                ),
                1e-30,
            )
        ),
        "finite": bool(
            np.isfinite(native_forward).all() and np.isfinite(native_adjoint).all()
        ),
    }
    row["quality_pass"] = bool(
        row["finite"]
        and row["forward_split_panel_vs_fp32_rel_l2"] <= 2e-7
        and row["adjoint_split_panel_vs_fp32_rel_l2"] <= 2e-7
        and row["adjoint_high_bitwise_equal"]
        and row["adjoint_residual_bitwise_equal"]
        and row["native_forward_vs_arithmetic_rel_l2"] <= 2e-5
        and row["native_adjoint_vs_arithmetic_rel_l2"] <= 2e-5
        and row["native_adjoint_defect"] <= 1e-6
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
