#!/usr/bin/env python3
"""Generate normalized 2D radial and golden-angle trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def radial(spokes: int, readout: int, golden: bool) -> np.ndarray:
    if golden:
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        angles = np.arange(spokes, dtype=np.float64) * golden_angle
    else:
        angles = np.arange(spokes, dtype=np.float64) * (np.pi / spokes)
    radius = np.linspace(-0.5, 0.5, readout, endpoint=False, dtype=np.float64)
    x = np.cos(angles)[:, None] * radius[None, :]
    y = np.sin(angles)[:, None] * radius[None, :]
    return np.stack((x, y), axis=-1).reshape(-1, 2).astype(np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spiral_npy = args.output / "spiral_standard_256.npy"
    spiral_raw = args.output / "spiral_standard_256.f32xy.bin"
    if not spiral_npy.is_file() or not spiral_raw.is_file():
        raise FileNotFoundError("the frozen spiral trajectory fixture is missing")
    spiral = np.load(spiral_npy)
    spiral_values = spiral.reshape(-1, 2)
    spiral_from_raw = np.fromfile(spiral_raw, dtype=np.float32).reshape(-1, 2)
    if spiral_values.shape != (65536, 2) or spiral.dtype != np.float32:
        raise ValueError("unexpected spiral trajectory shape or dtype")
    if not np.array_equal(spiral_values, spiral_from_raw):
        raise ValueError("spiral NPY and raw fixtures differ")
    rows = [
        {
            "name": "spiral",
            "samples": 65536,
            "file": str(spiral_npy),
            "sha256": sha256(spiral_npy),
            "cufinufft_file": str(spiral_raw),
            "cufinufft_sha256": sha256(spiral_raw),
            "coordinate_contract": "cycles_per_pixel in [-0.5, 0.5); cuFINUFFT harness multiplies by 2*pi",
        }
    ]
    for spokes in (128, 256):
        for name, golden in (("radial", False), ("golden", True)):
            trajectory = radial(spokes, 256, golden)
            path = args.output / f"{name}_{spokes}x256.npy"
            np.save(path, trajectory)
            raw_path = args.output / f"{name}_{spokes}x256.f32xy.bin"
            trajectory.tofile(raw_path)
            rows.append(
                {
                    "name": name,
                    "spokes": spokes,
                    "readout": 256,
                    "samples": int(trajectory.shape[0]),
                    "file": str(path),
                    "sha256": sha256(path),
                    "cufinufft_file": str(raw_path),
                    "cufinufft_sha256": sha256(raw_path),
                    "coordinate_contract": "cycles_per_pixel in [-0.5, 0.5); cuFINUFFT harness multiplies by 2*pi",
                }
            )
    (args.output / "manifest.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
