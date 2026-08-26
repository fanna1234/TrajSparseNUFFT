#!/usr/bin/env python3
"""Export one packed NUFFT view and a same-semantics CSR keeper for CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nufft_layout_oracle import (
    M_TILE,
    PackedTile,
    apply_packed,
    apply_sparse,
    build_interpolation,
    generate_trajectory,
    pack_view,
    packed_checksum,
    sparse_checksum,
    transpose_sparse,
    validate_24,
)


MAGIC = 0x4E55464654323431  # "NUFFT241"
VERSION = 2
N_CHANNELS = 16
K_TILE = 32
K_COMP = K_TILE // 2


def pair_metadata(row_meta: np.ndarray) -> np.ndarray:
    result = np.zeros(M_TILE, dtype=np.uint32)
    for group in range(8):
        m0 = int(row_meta[group])
        m1 = int(row_meta[group + 8])
        result[group * 2] = (m0 & 0xFFFF) | ((m1 & 0xFFFF) << 16)
        result[group * 2 + 1] = ((m0 >> 16) & 0xFFFF) | (m1 & 0xFFFF0000)
    return result


def compress_tile(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rounded = values.astype(np.float16)
    compressed = np.zeros((M_TILE, K_COMP), dtype=np.float16)
    row_meta = np.zeros(M_TILE, dtype=np.uint32)
    for row in range(M_TILE):
        meta = 0
        for chunk in range(K_TILE // 4):
            four = rounded[row, chunk * 4 : chunk * 4 + 4]
            nonzero = list(map(int, np.flatnonzero(four)))
            if len(nonzero) > 2:
                raise ValueError("tile is not legal 2:4 after FP16 rounding")
            selected = list(nonzero)
            for position in range(4):
                if len(selected) == 2:
                    break
                if position not in selected:
                    selected.append(position)
            selected.sort()
            p0, p1 = selected
            compressed[row, chunk * 2] = four[p0]
            compressed[row, chunk * 2 + 1] = four[p1]
            meta |= (p0 | (p1 << 2)) << (chunk * 4)
        row_meta[row] = meta
    return compressed, pair_metadata(row_meta)


def split_illegal_tile(tile: PackedTile) -> tuple[PackedTile, PackedTile]:
    layer0 = np.zeros_like(tile.values)
    layer1 = np.zeros_like(tile.values)
    for row in range(M_TILE):
        for offset in range(0, K_TILE, 4):
            positions = list(
                map(int, np.flatnonzero(tile.values[row, offset : offset + 4]))
            )
            for position in positions[:2]:
                layer0[row, offset + position] = tile.values[row, offset + position]
            for position in positions[2:]:
                layer1[row, offset + position] = tile.values[row, offset + position]
    first = PackedTile(
        row_ids=tile.row_ids.copy(),
        col_ids=tile.col_ids.copy(),
        values=layer0,
        legal_24=True,
        violation_count=0,
    )
    second = PackedTile(
        row_ids=tile.row_ids.copy(),
        col_ids=tile.col_ids.copy(),
        values=layer1,
        legal_24=True,
        violation_count=0,
    )
    if not validate_24(first)[0] or not validate_24(second)[0]:
        raise RuntimeError("split24 failed to legalize a tile")
    if not np.array_equal(first.values + second.values, tile.values):
        raise RuntimeError("split24 changed tile coefficients")
    return first, second


def split_legal_tiles_and_residual(view, packed, illegal_mode: str):
    offsets = [0]
    rows: list[np.ndarray] = []
    legal_tiles = []
    residual_cols: list[list[int]] = [[] for _ in range(view.output_size)]
    residual_vals: list[list[float]] = [[] for _ in range(view.output_size)]
    previous: tuple[int, ...] | None = None
    for tile in packed.tiles:
        current = tuple(map(int, tile.row_ids))
        if current != previous:
            if previous is not None:
                offsets.append(len(legal_tiles))
            rows.append(tile.row_ids.astype(np.int32, copy=True))
            previous = current
        if validate_24(tile)[0]:
            legal_tiles.append(tile)
        elif illegal_mode == "split24":
            legal_tiles.extend(split_illegal_tile(tile))
        else:
            for row_slot, logical_row in enumerate(tile.row_ids):
                if logical_row < 0:
                    continue
                for col_slot, logical_col in enumerate(tile.col_ids):
                    value = float(tile.values[row_slot, col_slot])
                    if logical_col >= 0 and value != 0.0:
                        residual_cols[int(logical_row)].append(int(logical_col))
                        residual_vals[int(logical_row)].append(value)
    offsets.append(len(legal_tiles))
    residual_row_ptr = np.zeros(view.output_size + 1, dtype=np.uint32)
    residual_row_ptr[1:] = np.cumsum(
        [len(cols) for cols in residual_cols], dtype=np.uint64
    )
    residual_col_array = np.asarray(
        [col for row in residual_cols for col in row], dtype=np.int32
    )
    residual_val_array = np.asarray(
        [value for row in residual_vals for value in row], dtype=np.float16
    )
    return (
        np.asarray(offsets, dtype=np.uint32),
        np.stack(rows).astype(np.int32),
        legal_tiles,
        residual_row_ptr,
        residual_col_array,
        residual_val_array,
    )


def write_array(path: Path, array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    path.write_bytes(contiguous.tobytes(order="C"))
    payload = path.read_bytes()
    return {
        "file": path.name,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def export_case(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    trajectory_file_sha256 = None
    if getattr(args, "trajectory_file", None):
        source_path = Path(args.trajectory_file)
        raw_trajectory = np.load(source_path).reshape(-1, 2).astype(np.float64)
        if not np.all(np.isfinite(raw_trajectory)):
            raise ValueError("trajectory file contains nonfinite coordinates")
        if np.max(np.abs(raw_trajectory)) > 0.500001:
            raise ValueError("trajectory file must use normalized [-0.5, 0.5] coordinates")
        trajectory = np.mod(raw_trajectory * args.grid + args.grid / 2.0, args.grid)
        trajectory_file_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        args.samples = len(trajectory)
    else:
        trajectory = generate_trajectory(args.trajectory, args.grid, args.samples)
    forward = build_interpolation(
        trajectory, args.grid, args.width, periodic=getattr(args, "periodic", False)
    )
    view = transpose_sparse(forward, args.grid) if args.view == "G_T" else forward
    packed = pack_view(view, args.policy, K_TILE, args.seed)
    if sparse_checksum(view) != packed_checksum(packed):
        raise RuntimeError("packed coefficient checksum mismatch")

    (
        group_offsets,
        group_row_ids,
        legal_tiles,
        residual_row_ptr,
        residual_cols,
        residual_vals,
    ) = split_legal_tiles_and_residual(view, packed, args.illegal_mode)
    if not legal_tiles:
        raise RuntimeError("export produced no legal Sparse-MMA tiles")
    tile_col_ids = np.stack([tile.col_ids for tile in legal_tiles]).astype(np.int32)
    tile_col_ids[tile_col_ids < 0] = 0
    a_comp: list[np.ndarray] = []
    pair_meta: list[np.ndarray] = []
    for tile in legal_tiles:
        comp, meta = compress_tile(tile.values)
        a_comp.append(comp)
        pair_meta.append(meta)
    tile_a_comp = np.stack(a_comp).astype(np.float16)
    tile_pair_meta = np.stack(pair_meta).astype(np.uint32)

    csr_row_ptr = np.zeros(view.n_active_rows + 1, dtype=np.uint32)
    csr_row_ptr[1:] = np.cumsum([len(cols) for cols in view.cols], dtype=np.uint64)
    csr_row_ids = view.row_ids.astype(np.int32, copy=True)
    csr_cols = np.concatenate(view.cols).astype(np.int32)
    csr_vals = np.concatenate(view.vals).astype(np.float16)

    rng = np.random.default_rng(args.seed)
    dense_input = np.clip(
        rng.standard_normal((view.n_cols, N_CHANNELS)), -3.0, 3.0
    ).astype(np.float16)
    reference = np.zeros((view.output_size, N_CHANNELS), dtype=np.float32)
    dense_f32 = dense_input.astype(np.float32)
    for local_row, (cols, vals) in enumerate(zip(view.cols, view.vals, strict=True)):
        reference[int(view.row_ids[local_row])] = (
            vals.astype(np.float16).astype(np.float32) @ dense_f32[cols]
        )

    # Independently confirm that the unrounded packed layout preserves the
    # operator before exporting its FP16 realization.
    probe = rng.standard_normal((view.n_cols, 2)) + 1j * rng.standard_normal(
        (view.n_cols, 2)
    )
    structural_error = float(
        np.max(np.abs(apply_sparse(view, probe) - apply_packed(packed, probe)))
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<Q11I",
        MAGIC,
        VERSION,
        N_CHANNELS,
        view.output_size,
        view.n_cols,
        view.n_active_rows,
        view.nnz,
        len(group_offsets) - 1,
        len(legal_tiles),
        len(residual_cols),
        M_TILE,
        K_TILE,
    )
    (args.output_dir / "header.bin").write_bytes(header)
    files = [
        {
            "file": "header.bin",
            "dtype": "packed_header_v2",
            "shape": [1],
            "bytes": len(header),
            "sha256": hashlib.sha256(header).hexdigest(),
        },
        write_array(args.output_dir / "group_offsets.bin", group_offsets),
        write_array(args.output_dir / "group_row_ids.bin", group_row_ids),
        write_array(args.output_dir / "tile_col_ids.bin", tile_col_ids),
        write_array(args.output_dir / "tile_a_comp.bin", tile_a_comp),
        write_array(args.output_dir / "tile_pair_meta.bin", tile_pair_meta),
        write_array(args.output_dir / "csr_row_ptr.bin", csr_row_ptr),
        write_array(args.output_dir / "csr_row_ids.bin", csr_row_ids),
        write_array(args.output_dir / "csr_cols.bin", csr_cols),
        write_array(args.output_dir / "csr_vals.bin", csr_vals),
        write_array(args.output_dir / "residual_row_ptr.bin", residual_row_ptr),
        write_array(args.output_dir / "residual_cols.bin", residual_cols),
        write_array(args.output_dir / "residual_vals.bin", residual_vals),
        write_array(args.output_dir / "dense_input.bin", dense_input),
        write_array(args.output_dir / "reference.bin", reference),
    ]
    half_nnz = int(np.count_nonzero(csr_vals))
    slot_count = len(legal_tiles) * M_TILE * K_TILE // 2
    residual_nonzero_fp16 = int(np.count_nonzero(residual_vals))
    legal_nonzero_fp16 = half_nnz - residual_nonzero_fp16
    manifest = {
        "experiment_id": "trajsparsenufft_structural_oracle",
        "trajectory": args.trajectory,
        "trajectory_file": str(args.trajectory_file) if args.trajectory_file else None,
        "trajectory_file_sha256": trajectory_file_sha256,
        "periodic_grid": bool(args.periodic),
        "view": args.view,
        "grid": args.grid,
        "width": args.width,
        "samples": args.samples,
        "policy": args.policy,
        "illegal_mode": args.illegal_mode,
        "seed": args.seed,
        "n_channels": N_CHANNELS,
        "m_tile": M_TILE,
        "k_tile": K_TILE,
        "output_size": view.output_size,
        "logical_k": view.n_cols,
        "active_rows": view.n_active_rows,
        "nnz_fp64": view.nnz,
        "nnz_nonzero_fp16": half_nnz,
        "groups": len(group_offsets) - 1,
        "tiles_total": len(packed.tiles),
        "tiles_legal": len(legal_tiles),
        "residual_nnz_fp64": len(residual_cols),
        "residual_nnz_nonzero_fp16": residual_nonzero_fp16,
        "residual_fraction_fp16": residual_nonzero_fp16 / half_nnz,
        "sparse_value_slots": slot_count,
        "fp16_legal_useful_slot_utilization": legal_nonzero_fp16 / slot_count,
        "structural_apply_max_abs_error": structural_error,
        "export_seconds": time.perf_counter() - start,
        "files": files,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", choices=("radial", "spiral"), default="spiral")
    parser.add_argument("--trajectory-file", type=Path)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--view", choices=("G", "G_T"), default="G_T")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--policy", default="overlap")
    parser.add_argument(
        "--illegal-mode", choices=("csr_residual", "split24"), default="csr_residual"
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduced-results/oracle-export"),
    )
    args = parser.parse_args()
    export_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
