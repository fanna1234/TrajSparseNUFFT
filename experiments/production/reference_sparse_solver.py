#!/usr/bin/env python3
"""Frozen FP32 and component-wise sparse-solver oracle for production validation.

This module is a numerical reference, not a performance path. Its wall times
must never be reported as production performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchkbnufft as tkbn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchkbnufft._nufft.fft import fft_and_scale, ifft_and_scale


EXPERIMENT_ID = "trajsparsenufft_reference_sparse_solver"
DEFAULT_VARIANTS = [
    "fp16x1",
    "fp16x2",
    "split2x1",
    "split2x2",
    "fp16xfp32",
    "fp32x1",
    "fp32xfp32",
]
VARIANTS = {
    "fp16x1": {"matrix": "fp16", "panel": "fp16", "sparse_mma_work": 1},
    "fp16x2": {"matrix": "fp16", "panel": "split2", "sparse_mma_work": 2},
    "split2x1": {"matrix": "split2", "panel": "fp16", "sparse_mma_work": 2},
    "split2x2": {"matrix": "split2", "panel": "split2", "sparse_mma_work": 4},
    "fp16xfp32": {"matrix": "fp16", "panel": "fp32", "sparse_mma_work": None},
    "fp32x1": {"matrix": "fp32", "panel": "fp16", "sparse_mma_work": None},
    "fp32xfp32": {"matrix": "fp32", "panel": "fp32", "sparse_mma_work": None},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(candidate - reference)
    denominator = torch.clamp(torch.linalg.vector_norm(reference), min=1e-30)
    return float(numerator / denominator)


def nrmse(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), 1e-30)
    return float(np.linalg.norm(candidate - reference) / denominator)


def dense_components(value: torch.Tensor, mode: str) -> list[torch.Tensor]:
    value = value.float()
    if mode == "fp32":
        return [value]
    high = value.half().float()
    if mode == "fp16":
        return [high]
    if mode == "split2":
        low = (value - high).half().float()
        return [high, low]
    raise ValueError(f"unknown dense component mode: {mode}")


def sparse_components(
    matrix: torch.Tensor, mode: str, device: torch.device, layout: str
) -> list[torch.Tensor]:
    matrix = matrix.coalesce()
    indices = matrix.indices().to(device)
    values = matrix.values().to(device).float()
    if mode == "fp32":
        component_values = [values]
    else:
        high = values.half().float()
        if mode == "fp16":
            component_values = [high]
        elif mode == "split2":
            low = (values - high).half().float()
            component_values = [high, low]
        else:
            raise ValueError(f"unknown sparse component mode: {mode}")
    components = [
        torch.sparse_coo_tensor(
            indices, component, matrix.shape, device=device, dtype=torch.float32
        ).coalesce()
        for component in component_values
    ]
    if layout == "coo":
        return components
    if layout == "csr":
        return [component.to_sparse_csr() for component in components]
    raise ValueError(f"unknown sparse layout: {layout}")


def transpose_sparse_component(component: torch.Tensor, layout: str) -> torch.Tensor:
    transpose = component.transpose(0, 1)
    if layout == "coo":
        return transpose.coalesce()
    if layout == "csr":
        return transpose.to_sparse_csr()
    raise ValueError(f"unknown sparse layout: {layout}")


def sum_sparse_products(
    matrices: list[torch.Tensor], panels: list[torch.Tensor]
) -> torch.Tensor:
    output = None
    for matrix in matrices:
        for panel in panels:
            product = torch.sparse.mm(matrix, panel)
            output = product if output is None else output + product
    if output is None:
        raise RuntimeError("empty component product")
    return output


def make_component_operator(
    forward_op,
    adjoint_op,
    real_matrix: torch.Tensor,
    imag_matrix: torch.Tensor,
    smaps: torch.Tensor,
    matrix_mode: str,
    panel_mode: str,
    coils: int,
    grid_size: int,
    device: torch.device,
    sparse_layout: str,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    real_components = sparse_components(real_matrix, matrix_mode, device, sparse_layout)
    imag_components = sparse_components(imag_matrix, matrix_mode, device, sparse_layout)
    real_transpose = [
        transpose_sparse_component(component, sparse_layout)
        for component in real_components
    ]
    imag_transpose = [
        transpose_sparse_component(component, sparse_layout)
        for component in imag_components
    ]

    def forward(image: torch.Tensor) -> torch.Tensor:
        coil_image = image * smaps
        grid = fft_and_scale(
            image=coil_image,
            scaling_coef=forward_op.scaling_coef,
            im_size=forward_op.im_size,
            grid_size=forward_op.grid_size,
            norm=None,
        )
        panel = grid.reshape(coils, -1).T
        panel_real = dense_components(panel.real, panel_mode)
        panel_imag = dense_components(panel.imag, panel_mode)
        output_real = sum_sparse_products(real_components, panel_real) - sum_sparse_products(
            imag_components, panel_imag
        )
        output_imag = sum_sparse_products(real_components, panel_imag) + sum_sparse_products(
            imag_components, panel_real
        )
        return torch.complex(output_real, output_imag).T.unsqueeze(0)

    def adjoint(data: torch.Tensor) -> torch.Tensor:
        panel = data.squeeze(0).T
        panel_real = dense_components(panel.real, panel_mode)
        panel_imag = dense_components(panel.imag, panel_mode)
        grid_real = sum_sparse_products(real_transpose, panel_real) + sum_sparse_products(
            imag_transpose, panel_imag
        )
        grid_imag = sum_sparse_products(real_transpose, panel_imag) - sum_sparse_products(
            imag_transpose, panel_real
        )
        grid = torch.complex(grid_real, grid_imag).T.reshape(
            1, coils, grid_size, grid_size
        )
        coil_image = ifft_and_scale(
            image=grid,
            scaling_coef=adjoint_op.scaling_coef,
            im_size=adjoint_op.im_size,
            grid_size=adjoint_op.grid_size,
            norm=None,
        )
        return torch.sum(smaps.conj() * coil_image, dim=1, keepdim=True)

    return forward, adjoint


def cg_sense(
    forward: Callable[[torch.Tensor], torch.Tensor],
    adjoint: Callable[[torch.Tensor], torch.Tensor],
    data: torch.Tensor,
    iterations: int,
    lam: float,
    label: str,
    debug: bool,
    rhs_adjoint: Callable[[torch.Tensor], torch.Tensor] | None = None,
    reliable_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
    reliable_adjoint: Callable[[torch.Tensor], torch.Tensor] | None = None,
    reliable_update_period: int = 0,
    reliable_update_direction: str = "restart",
    scheduled_reference_period: int = 0,
    explicit_reference_steps: frozenset[int] = frozenset(),
) -> tuple[torch.Tensor, list[float], dict]:
    if reliable_update_period < 0:
        raise ValueError("reliable_update_period must be non-negative")
    if (
        reliable_update_period > 0
        or scheduled_reference_period > 0
        or explicit_reference_steps
    ) and (
        reliable_forward is None or reliable_adjoint is None
    ):
        raise ValueError("reference step requires reference forward and adjoint")
    if reliable_update_direction not in ("restart", "beta"):
        raise ValueError(f"unknown reliable update direction: {reliable_update_direction}")
    status = {
        "finite": True,
        "positive_curvature": True,
        "completed_iterations": 0,
        "failure": None,
        "reliable_update_period": reliable_update_period,
        "reliable_update_direction": reliable_update_direction,
        "reliable_updates": [],
        "scheduled_reference_period": scheduled_reference_period,
        "scheduled_reference_steps": [],
        "curvature_history": [],
    }
    b = (rhs_adjoint or adjoint)(data)
    x = torch.zeros_like(b)
    if not bool(torch.isfinite(b).all()):
        status.update(finite=False, failure="nonfinite_rhs")
        return x, [], status
    r = b.clone()
    p = r.clone()
    rs_old = torch.vdot(r.flatten(), r.flatten()).real
    residuals = [float(torch.sqrt(torch.clamp(rs_old, min=0.0)))]
    for iteration in range(iterations):
        completed_iteration = iteration + 1
        use_scheduled_reference = (
            completed_iteration in explicit_reference_steps
            or (
                scheduled_reference_period > 0
                and completed_iteration % scheduled_reference_period == 0
            )
        )
        if use_scheduled_reference:
            ap = reliable_adjoint(reliable_forward(p)) + lam * p
            status["scheduled_reference_steps"].append(completed_iteration)
        else:
            ap = adjoint(forward(p)) + lam * p
        denominator = torch.vdot(p.flatten(), ap.flatten()).real
        boundary = {
            "cg_boundary": label,
            "iteration": iteration,
            "rs": float(rs_old),
            "denominator": float(denominator),
            "p_finite": bool(torch.isfinite(p).all()),
            "ap_finite": bool(torch.isfinite(ap).all()),
            "p_norm": float(torch.linalg.vector_norm(p)),
            "ap_norm": float(torch.linalg.vector_norm(ap)),
        }
        status["curvature_history"].append(boundary)
        if debug:
            print(json.dumps(boundary, sort_keys=True), flush=True)
        if not boundary["p_finite"] or not boundary["ap_finite"] or not bool(
            torch.isfinite(denominator)
        ):
            status.update(finite=False, failure=f"nonfinite_iteration_{iteration}")
            break
        if float(denominator) <= 0.0:
            status.update(
                positive_curvature=False,
                failure=f"nonpositive_curvature_iteration_{iteration}",
            )
            break
        alpha = rs_old / torch.clamp(denominator, min=1e-30)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.vdot(r.flatten(), r.flatten()).real
        if not bool(torch.isfinite(rs_new)) or not bool(torch.isfinite(x).all()):
            status.update(finite=False, failure=f"nonfinite_update_{iteration}")
            break
        recursive_residual = float(torch.sqrt(torch.clamp(rs_new, min=0.0)))
        if (
            reliable_update_period > 0
            and completed_iteration % reliable_update_period == 0
            and completed_iteration < iterations
        ):
            reliable_normal = reliable_adjoint(reliable_forward(x)) + lam * x
            r = b - reliable_normal
            rs_new = torch.vdot(r.flatten(), r.flatten()).real
            if not bool(torch.isfinite(rs_new)) or not bool(torch.isfinite(r).all()):
                status.update(
                    finite=False,
                    failure=f"nonfinite_reliable_update_{completed_iteration}",
                )
                break
            reliable_residual = float(torch.sqrt(torch.clamp(rs_new, min=0.0)))
            status["reliable_updates"].append(
                {
                    "after_iteration": completed_iteration,
                    "recursive_residual": recursive_residual,
                    "reliable_residual": reliable_residual,
                }
            )
            residuals.append(reliable_residual)
            if reliable_update_direction == "restart":
                p = r.clone()
            else:
                beta = rs_new / torch.clamp(rs_old, min=1e-30)
                p = r + beta * p
            rs_old = rs_new
            status["completed_iterations"] = completed_iteration
            continue
        residuals.append(recursive_residual)
        beta = rs_new / torch.clamp(rs_old, min=1e-30)
        p = r + beta * p
        rs_old = rs_new
        status["completed_iterations"] = completed_iteration
    return x, residuals, status


def route_adjoint_defect(
    forward: Callable[[torch.Tensor], torch.Tensor],
    adjoint: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    samples: torch.Tensor,
) -> float:
    forward_image = forward(image)
    adjoint_samples = adjoint(samples)
    left = torch.vdot(forward_image.flatten(), samples.flatten())
    right = torch.vdot(image.flatten(), adjoint_samples.flatten())
    denominator = torch.clamp(
        torch.linalg.vector_norm(forward_image)
        * torch.linalg.vector_norm(samples),
        min=1e-30,
    )
    return float(torch.abs(left - right) / denominator)


def environment_snapshot(device: torch.device) -> dict:
    try:
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception as error:  # pragma: no cover - diagnostic fallback
        driver = f"unavailable: {error}"
    snapshot = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchkbnufft": getattr(tkbn, "__version__", "unknown"),
        "execution_device": str(device),
        "nvidia_smi": driver,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        snapshot.update(
            gpu_name=properties.name,
            gpu_capability=[properties.major, properties.minor],
        )
    return snapshot


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def write_summary(output: Path, metadata: dict, rows: list[dict]) -> None:
    payload = dict(metadata)
    payload["rows"] = rows
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument(
        "--trajectory", action="append", required=True, help="NAME=PATH; repeatable"
    )
    parser.add_argument("--frame", action="append", type=int)
    parser.add_argument(
        "--selection",
        action="append",
        help="Optional NAME:FRAME pair; repeat to run a non-Cartesian product subset.",
    )
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--lambda-value", type=float, default=1e-4)
    parser.add_argument("--quality-gate", type=float, default=1e-3)
    parser.add_argument("--truth-regression-gate", type=float)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--reliable-update-period", type=int, default=0)
    parser.add_argument("--baseline-reliable-update-period", type=int, default=-1)
    parser.add_argument(
        "--reliable-update-direction",
        choices=("restart", "beta"),
        default="restart",
    )
    parser.add_argument("--scheduled-fp32-step-period", type=int, default=0)
    parser.add_argument("--scheduled-fp32-step", action="append", type=int)
    parser.add_argument(
        "--solver-reference",
        choices=("table", "sparse-fp32"),
        default="table",
        help="Reference operator used to generate data and run the baseline CG solver.",
    )
    parser.add_argument(
        "--sparse-layout",
        choices=("coo", "csr"),
        default="coo",
        help="Torch sparse emulation layout; CSR is the deterministic-control candidate.",
    )
    parser.add_argument("--debug-cg", action="store_true")
    args = parser.parse_args()

    if args.reliable_update_period < 0:
        raise ValueError("reliable update period must be non-negative")
    if args.scheduled_fp32_step_period < 0:
        raise ValueError("scheduled FP32 step period must be non-negative")
    explicit_scheduled_steps = frozenset(args.scheduled_fp32_step or [])
    if any(step < 1 or step > args.iterations for step in explicit_scheduled_steps):
        raise ValueError("explicit scheduled FP32 steps must be within 1..iterations")
    baseline_reliable_period = (
        args.reliable_update_period
        if args.baseline_reliable_update_period < 0
        else args.baseline_reliable_update_period
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    frames_to_run = args.frame or [0]
    variants_to_run = args.variant or DEFAULT_VARIANTS
    trajectories: list[tuple[str, Path]] = []
    for specification in args.trajectory:
        name, path = specification.split("=", 1)
        trajectories.append((name, Path(path)))
    selections: set[tuple[str, int]] = set()
    for selection in args.selection or []:
        name, frame_text = selection.rsplit(":", 1)
        selections.add((name, int(frame_text)))
    trajectory_names = {name for name, _ in trajectories}
    unknown_selection_names = {name for name, _ in selections} - trajectory_names
    if unknown_selection_names:
        raise ValueError(f"selection names not present in trajectories: {unknown_selection_names}")

    coil_path = args.case / "coil_images.npy"
    smaps_path = args.case / "sensitivity_maps.npy"
    reference_path = args.case / "reference_rss.npy"
    coil_images_np = np.load(coil_path).astype(np.complex64)
    smaps_np = np.load(smaps_path).astype(np.complex64)
    reference_np = np.load(reference_path).astype(np.float32)
    del coil_images_np  # The frozen reconstruction oracle uses the prepared RSS truth.
    frame_count, coils, height, width = smaps_np.shape
    if (coils, height, width) != (8, 256, 256):
        raise ValueError(f"unsupported prepared shape {smaps_np.shape}")
    if any(frame < 0 or frame >= frame_count for frame in frames_to_run):
        raise ValueError(f"frames {frames_to_run} outside prepared range 0..{frame_count - 1}")

    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "rows.jsonl"
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {raw_path}")

    hashes = {
        str(coil_path): sha256(coil_path),
        str(smaps_path): sha256(smaps_path),
        str(reference_path): sha256(reference_path),
    }
    for _, trajectory_path in trajectories:
        hashes[str(trajectory_path)] = sha256(trajectory_path)
    metadata = {
        "experiment_id": args.experiment_id,
        "evidence_class": "real OCMR anatomy/coils with retrospective trajectories",
        "timing_evidence": "diagnostic Torch sparse emulation only",
        "iterations": args.iterations,
        "lambda": args.lambda_value,
        "quality_gate_candidate_vs_reference_nrmse": args.quality_gate,
        "truth_nrmse_absolute_regression_gate": args.truth_regression_gate,
        "solver_reference": args.solver_reference,
        "sparse_layout": args.sparse_layout,
        "reliable_update_period": args.reliable_update_period,
        "baseline_reliable_update_period": baseline_reliable_period,
        "reliable_update_direction": args.reliable_update_direction,
        "scheduled_fp32_step_period": args.scheduled_fp32_step_period,
        "explicit_scheduled_fp32_steps": sorted(explicit_scheduled_steps),
        "planned_reliable_update_iterations": [
            iteration
            for iteration in range(1, args.iterations)
            if args.reliable_update_period > 0
            and iteration % args.reliable_update_period == 0
        ],
        "planned_baseline_reliable_update_iterations": [
            iteration
            for iteration in range(1, args.iterations)
            if baseline_reliable_period > 0
            and iteration % baseline_reliable_period == 0
        ],
        "planned_scheduled_fp32_steps": sorted(
            explicit_scheduled_steps
            | {
                iteration
                for iteration in range(1, args.iterations + 1)
                if args.scheduled_fp32_step_period > 0
                and iteration % args.scheduled_fp32_step_period == 0
            }
        ),
        "frames": frames_to_run,
        "selections": sorted([f"{name}:{frame}" for name, frame in selections]),
        "variants": variants_to_run,
        "hashes": hashes,
        "environment": environment_snapshot(device),
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    rows: list[dict] = []
    torch.manual_seed(20260824)
    for trajectory_name, trajectory_path in trajectories:
        trajectory = np.load(trajectory_path).reshape(-1, 2).astype(np.float32)
        omega_cpu = torch.from_numpy((trajectory.T * 2.0 * np.pi).astype(np.float32))
        omega = omega_cpu.to(device)
        matrix_started = time.perf_counter()
        real_matrix, imag_matrix = tkbn.calc_tensor_spmatrix(
            omega_cpu, (height, width), grid_size=(512, 512), numpoints=6
        )
        matrix_seconds = time.perf_counter() - matrix_started
        real_matrix = real_matrix.coalesce().to(device)
        imag_matrix = imag_matrix.coalesce().to(device)

        forward_op = tkbn.KbNufft(
            im_size=(height, width),
            grid_size=(512, 512),
            numpoints=6,
            dtype=torch.float32,
        ).to(device)
        adjoint_op = tkbn.KbNufftAdjoint(
            im_size=(height, width),
            grid_size=(512, 512),
            numpoints=6,
            dtype=torch.float32,
        ).to(device)

        for frame in frames_to_run:
            if selections and (trajectory_name, frame) not in selections:
                continue
            smaps = torch.from_numpy(smaps_np[frame : frame + 1]).to(device)
            truth = torch.from_numpy(
                reference_np[frame : frame + 1, None].astype(np.complex64)
            ).to(device)

            def forward_table(image: torch.Tensor) -> torch.Tensor:
                return forward_op(image, omega, smaps=smaps)

            def adjoint_table(data: torch.Tensor) -> torch.Tensor:
                return adjoint_op(data, omega, smaps=smaps)

            with torch.no_grad():
                table_samples = forward_table(truth)
                if args.solver_reference == "table":
                    forward_reference = forward_table
                    adjoint_reference = adjoint_table
                else:
                    forward_reference, adjoint_reference = make_component_operator(
                        forward_op,
                        adjoint_op,
                        real_matrix,
                        imag_matrix,
                        smaps,
                        "fp32",
                        "fp32",
                        coils,
                        512,
                        device,
                        args.sparse_layout,
                    )
                reference_samples = forward_reference(truth)
                normal_rhs = adjoint_reference(reference_samples)
                operator_scale = float(
                    torch.sqrt(torch.clamp(torch.max(torch.abs(normal_rhs)), min=1e-12))
                )
                scaled_samples = reference_samples / operator_scale
                scaled_lambda = args.lambda_value / (operator_scale * operator_scale)
                baseline_started = time.perf_counter()
                baseline_reconstruction, baseline_residuals, baseline_status = cg_sense(
                    lambda value: forward_reference(value) / operator_scale,
                    lambda value: adjoint_reference(value) / operator_scale,
                    scaled_samples,
                    args.iterations,
                    scaled_lambda,
                    f"{trajectory_name}/frame{frame}/{args.solver_reference}",
                    args.debug_cg,
                    rhs_adjoint=lambda value: adjoint_reference(value)
                    / operator_scale,
                    reliable_forward=lambda value: forward_reference(value)
                    / operator_scale,
                    reliable_adjoint=lambda value: adjoint_reference(value)
                    / operator_scale,
                    reliable_update_period=baseline_reliable_period,
                    reliable_update_direction=args.reliable_update_direction,
                    scheduled_reference_period=0,
                    explicit_reference_steps=frozenset(),
                )
                synchronize(device)
                baseline_seconds = time.perf_counter() - baseline_started
                if (
                    not baseline_status["finite"]
                    or not baseline_status["positive_curvature"]
                    or baseline_status["completed_iterations"] != args.iterations
                ):
                    raise RuntimeError(f"baseline CG failed: {baseline_status}")

                truth_image = torch.abs(truth[0, 0]).cpu().numpy()
                baseline_image = torch.abs(baseline_reconstruction[0, 0]).cpu().numpy()
                baseline_truth_nrmse = nrmse(baseline_image, truth_image)
                random_samples = torch.randn_like(table_samples)

                for variant_name in variants_to_run:
                    specification = VARIANTS[variant_name]
                    forward_candidate, adjoint_candidate = make_component_operator(
                        forward_op,
                        adjoint_op,
                        real_matrix,
                        imag_matrix,
                        smaps,
                        specification["matrix"],
                        specification["panel"],
                        coils,
                        512,
                        device,
                        args.sparse_layout,
                    )
                    forward_samples = forward_candidate(truth)
                    candidate_adjoint = adjoint_candidate(random_samples)
                    table_adjoint = adjoint_table(random_samples)
                    reference_adjoint = adjoint_reference(random_samples)
                    defect = route_adjoint_defect(
                        forward_candidate, adjoint_candidate, truth, random_samples
                    )
                    candidate_started = time.perf_counter()
                    reference_needed = (
                        args.reliable_update_period > 0
                        or args.scheduled_fp32_step_period > 0
                        or bool(explicit_scheduled_steps)
                    )
                    candidate_reconstruction, candidate_residuals, candidate_status = cg_sense(
                        lambda value: forward_candidate(value) / operator_scale,
                        lambda value: adjoint_candidate(value) / operator_scale,
                        scaled_samples,
                        args.iterations,
                        scaled_lambda,
                        f"{trajectory_name}/frame{frame}/{variant_name}",
                        args.debug_cg,
                        rhs_adjoint=(
                            (lambda value: adjoint_reference(value) / operator_scale)
                            if reference_needed
                            else None
                        ),
                        reliable_forward=(
                            (lambda value: forward_reference(value) / operator_scale)
                            if reference_needed
                            else None
                        ),
                        reliable_adjoint=(
                            (lambda value: adjoint_reference(value) / operator_scale)
                            if reference_needed
                            else None
                        ),
                        reliable_update_period=args.reliable_update_period,
                        reliable_update_direction=args.reliable_update_direction,
                        scheduled_reference_period=args.scheduled_fp32_step_period,
                        explicit_reference_steps=explicit_scheduled_steps,
                    )
                    synchronize(device)
                    candidate_seconds = time.perf_counter() - candidate_started

                    candidate_image = torch.abs(candidate_reconstruction[0, 0]).cpu().numpy()
                    data_range = max(
                        float(baseline_image.max() - baseline_image.min()), 1e-12
                    )
                    reconstruction_nrmse = nrmse(candidate_image, baseline_image)
                    candidate_truth_nrmse = nrmse(candidate_image, truth_image)
                    truth_nrmse_delta = candidate_truth_nrmse - baseline_truth_nrmse
                    truth_quality_pass = (
                        args.truth_regression_gate is None
                        or truth_nrmse_delta <= args.truth_regression_gate
                    )
                    completed = (
                        candidate_status["finite"]
                        and candidate_status["positive_curvature"]
                        and candidate_status["completed_iterations"] == args.iterations
                    )
                    row = {
                        "trajectory": trajectory_name,
                        "trajectory_file": str(trajectory_path),
                        "samples": int(trajectory.shape[0]),
                        "frame": frame,
                        "variant": variant_name,
                        "matrix_mode": specification["matrix"],
                        "panel_mode": specification["panel"],
                        "expected_sparse_mma_work": specification["sparse_mma_work"],
                        "matrix_build_seconds": matrix_seconds,
                        "nnz": int(real_matrix._nnz()),
                        "operator_scale": operator_scale,
                        "scaled_lambda": scaled_lambda,
                        "solver_reference": args.solver_reference,
                        "sparse_layout": args.sparse_layout,
                        "reliable_update_period": args.reliable_update_period,
                        "baseline_reliable_update_period": baseline_reliable_period,
                        "reliable_update_direction": args.reliable_update_direction,
                        "scheduled_fp32_step_period": args.scheduled_fp32_step_period,
                        "scheduled_fp32_steps": candidate_status[
                            "scheduled_reference_steps"
                        ],
                        "reliable_update_iterations": [
                            update["after_iteration"]
                            for update in candidate_status["reliable_updates"]
                        ],
                        "ordinary_candidate_normal_evaluations": (
                            args.iterations
                            - len(candidate_status["scheduled_reference_steps"])
                        ),
                        "fp32_rhs_adjoint_evaluations": (
                            1 if reference_needed else 0
                        ),
                        "additional_fp32_normal_evaluations": len(
                            candidate_status["reliable_updates"]
                        ),
                        "scheduled_fp32_normal_evaluations": len(
                            candidate_status["scheduled_reference_steps"]
                        ),
                        "forward_vs_reference_rel_l2": rel_l2(
                            forward_samples, reference_samples
                        ),
                        "adjoint_vs_reference_rel_l2": rel_l2(
                            candidate_adjoint, reference_adjoint
                        ),
                        "forward_vs_table_rel_l2": rel_l2(forward_samples, table_samples),
                        "adjoint_vs_table_rel_l2": rel_l2(candidate_adjoint, table_adjoint),
                        "empirical_route_adjoint_defect": defect,
                        "baseline_truth_nrmse": baseline_truth_nrmse,
                        "candidate_truth_nrmse": candidate_truth_nrmse,
                        "truth_nrmse_delta": truth_nrmse_delta,
                        "truth_quality_pass": bool(truth_quality_pass),
                        "candidate_vs_reference_nrmse": reconstruction_nrmse,
                        "candidate_vs_table_nrmse": (
                            reconstruction_nrmse
                            if args.solver_reference == "table"
                            else None
                        ),
                        "candidate_vs_sparse_fp32_nrmse": (
                            reconstruction_nrmse
                            if args.solver_reference == "sparse-fp32"
                            else None
                        ),
                        "candidate_vs_reference_ssim": float(
                            structural_similarity(
                                baseline_image, candidate_image, data_range=data_range
                            )
                        ),
                        "candidate_vs_reference_psnr_db": float(
                            peak_signal_noise_ratio(
                                baseline_image, candidate_image, data_range=data_range
                            )
                        ),
                        "baseline_residuals": baseline_residuals,
                        "candidate_residuals": candidate_residuals,
                        "baseline_emulation_seconds": baseline_seconds,
                        "candidate_emulation_seconds": candidate_seconds,
                        "cg_status": candidate_status,
                        "quality_pass": bool(
                            completed
                            and reconstruction_nrmse <= args.quality_gate
                            and truth_quality_pass
                        ),
                    }
                    rows.append(row)
                    with raw_path.open("a") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    write_summary(args.output, metadata, rows)
                    print(json.dumps(row, sort_keys=True), flush=True)
                    del forward_candidate, adjoint_candidate, candidate_reconstruction
                    clear_device_cache(device)

        del real_matrix, imag_matrix, forward_op, adjoint_op
        clear_device_cache(device)

    write_summary(args.output, metadata, rows)
    passers = sorted({row["variant"] for row in rows if row["quality_pass"]})
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "rows": len(rows),
                "quality_passers_present_in_rows": passers,
                "summary": str(args.output / "summary.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
