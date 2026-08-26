#!/usr/bin/env python3
"""Export real OCMR coil data for the native CUDA runtime harness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torchkbnufft as tkbn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_array(directory: Path, name: str, array: np.ndarray) -> dict:
    path = directory / name
    contiguous = np.ascontiguousarray(array)
    contiguous.tofile(path)
    return {
        "file": name,
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument(
        "--trajectory", action="append", required=True, help="NAME=PATH"
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--measurement-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.measurement_scale <= 0.0:
        raise ValueError("measurement-scale must be positive")
    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    coil_images = np.load(args.case / "coil_images.npy").astype(np.complex64)
    sensitivity_maps = np.load(args.case / "sensitivity_maps.npy").astype(np.complex64)
    reference = np.load(args.case / "reference_rss.npy").astype(np.float32)
    if coil_images.shape[1:] != (8, 256, 256):
        raise ValueError(f"unsupported coil image shape {coil_images.shape}")
    if not 0 <= args.frame < coil_images.shape[0]:
        raise ValueError(f"frame {args.frame} is outside {coil_images.shape[0]} frames")

    device = torch.device("cuda")
    image = torch.from_numpy(reference[args.frame : args.frame + 1, None]).to(
        device=device, dtype=torch.complex64
    )
    smaps = torch.from_numpy(sensitivity_maps[args.frame : args.frame + 1]).to(device)

    manifests = []
    for specification in args.trajectory:
        name, trajectory_string = specification.split("=", 1)
        trajectory_path = Path(trajectory_string)
        trajectory = np.load(trajectory_path).reshape(-1, 2).astype(np.float32)
        omega = torch.from_numpy((trajectory.T * (2.0 * np.pi)).astype(np.float32)).to(device)
        operator = tkbn.KbNufft(
            im_size=(256, 256),
            grid_size=(512, 512),
            numpoints=6,
            dtype=torch.float32,
        ).to(device)
        with torch.no_grad():
            measurement = operator(image, omega, smaps=smaps)
            torch.cuda.synchronize()

        output = output_root / name
        output.mkdir(parents=True, exist_ok=True)
        runtime_coils = (image * smaps)[0].cpu().numpy().astype(np.complex64)
        measurement_np = np.asarray(
            measurement[0].cpu().numpy().astype(np.complex64)
            * np.complex64(args.measurement_scale),
            dtype=np.complex64,
        )
        scaling_complex = operator.scaling_coef.detach().cpu().numpy().squeeze()
        if np.max(np.abs(np.imag(scaling_complex))) > 1e-7:
            raise ValueError("TorchKbNufft scaling coefficient is unexpectedly complex")
        scaling = np.real(scaling_complex).astype(np.float32)
        density = np.ones(trajectory.shape[0], dtype=np.float32)
        entries = [
            write_array(output, "forward_coils.c64.bin", runtime_coils),
            write_array(output, "measurement.c64.bin", measurement_np),
            write_array(output, "scaling.f32.bin", scaling),
            write_array(output, "density.f32.bin", density),
            write_array(
                output,
                "sensitivity_maps.c64.bin",
                sensitivity_maps[args.frame].astype(np.complex64),
            ),
            write_array(output, "truth.f32.bin", reference[args.frame].astype(np.float32)),
            write_array(
                output,
                "truth.c64.bin",
                reference[args.frame].astype(np.complex64),
            ),
        ]
        manifest = {
            "experiment_id": "data_prep",
            "name": name,
            "frame": args.frame,
            "samples": int(trajectory.shape[0]),
            "coils": 8,
            "image_size": [256, 256],
            "grid_size": [512, 512],
            "trajectory_file": str(trajectory_path),
            "source_case": str(args.case),
            "measurement_reference": "TorchKbNufft table path, FP32",
            "measurement_scale": args.measurement_scale,
            "density_contract": "all ones; matches the frozen quality experiment",
            "entries": entries,
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests.append(manifest)
        print(json.dumps(manifest, sort_keys=True))

    (output_root / "manifest.json").write_text(
        json.dumps(manifests, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
