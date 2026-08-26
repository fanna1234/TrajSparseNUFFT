#!/usr/bin/env python3
"""Export TorchKbNufft real/imag sparse interpolation components for CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchkbnufft as tkbn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.export_gpu_case import (
    K_TILE,
    MAGIC,
    N_CHANNELS,
    VERSION,
    compress_tile,
    split_illegal_tile,
    split_legal_tiles_and_residual,
    write_array,
)
from src.nufft_layout_oracle import (
    M_TILE,
    PackedView,
    PackedTile,
    SparseRows,
    pack_view,
    row_order,
    transpose_sparse,
    validate_24,
)


def fast_split24_pack(view: SparseRows, policy: str, seed: int) -> PackedView:
    """Pack geometry directly; Split24 removes the need for graph search."""

    started = time.perf_counter()
    order = row_order(view, policy, K_TILE)
    tiles: list[PackedTile] = []
    for offset in range(0, len(order), M_TILE):
        local_rows = order[offset : offset + M_TILE]
        local_rows += [-1] * (M_TILE - len(local_rows))
        row_ids = np.asarray(
            [int(view.row_ids[row]) if row >= 0 else -1 for row in local_rows],
            dtype=np.int32,
        )
        row_maps = [
            dict(zip(map(int, view.cols[row]), map(float, view.vals[row]), strict=True))
            if row >= 0
            else {}
            for row in local_rows
        ]
        union = sorted({column for row_map in row_maps for column in row_map})
        for column_offset in range(0, len(union), K_TILE):
            columns = union[column_offset : column_offset + K_TILE]
            columns += [-1] * (K_TILE - len(columns))
            col_ids = np.asarray(columns, dtype=np.int32)
            values = np.zeros((M_TILE, K_TILE), dtype=np.float64)
            for row_slot, row_map in enumerate(row_maps):
                values[row_slot] = np.fromiter(
                    (row_map.get(column, 0.0) if column >= 0 else 0.0 for column in columns),
                    dtype=np.float64,
                    count=K_TILE,
                )
            tile = PackedTile(
                row_ids=row_ids.copy(),
                col_ids=col_ids,
                values=values,
                legal_24=validate_24(
                    PackedTile(row_ids, col_ids, values, True, 0)
                )[0],
                violation_count=0,
            )
            if tile.legal_24:
                tiles.append(tile)
            else:
                tiles.extend(split_illegal_tile(tile))
    return PackedView(
        source_name=view.name,
        output_size=view.output_size,
        n_cols=view.n_cols,
        m_tile=M_TILE,
        k_tile=K_TILE,
        policy=f"fast-{policy}",
        tiles=tiles,
        pack_seconds=time.perf_counter() - started,
    )


def optimized_assign_columns(
    column_masks: dict[int, int], seed: int
) -> list[list[int]]:
    columns = list(column_masks)
    if not columns:
        return [[]]
    bits_by_column = {
        column: np.fromiter(
            ((mask >> row) & 1 for row in range(M_TILE)),
            dtype=np.int16,
            count=M_TILE,
        )
        for column, mask in column_masks.items()
    }
    max_row_degree = max(
        sum(int(bits[row]) for bits in bits_by_column.values())
        for row in range(M_TILE)
    )
    minimum_tiles = max(
        math.ceil(len(columns) / K_TILE),
        math.ceil(max_row_degree / (K_TILE // 2)),
        1,
    )
    rng = random.Random(seed)
    for tile_count in range(minimum_tiles, minimum_tiles + 9):
        for _ in range(8):
            tie_break = {column: rng.random() for column in columns}
            ordered = sorted(
                columns,
                key=lambda column: (
                    -column_masks[column].bit_count(), tie_break[column]
                ),
            )
            tiles: list[list[int]] = [[] for _ in range(tile_count)]
            sizes = np.zeros(tile_count, dtype=np.int16)
            counts = np.zeros((tile_count, M_TILE), dtype=np.int16)
            success = True
            for column in ordered:
                bits = bits_by_column[column]
                updated = counts + bits[None, :]
                valid = (sizes < K_TILE) & (
                    np.max(updated, axis=1) <= K_TILE // 2
                )
                if not np.any(valid):
                    success = False
                    break
                score = np.sum(
                    updated * updated - counts * counts, axis=1, dtype=np.int64
                ).astype(np.float64)
                score += sizes.astype(np.float64) / K_TILE
                score += np.asarray(
                    [rng.random() * 1e-6 for _ in range(tile_count)]
                )
                score[~valid] = np.inf
                selected = int(np.argmin(score))
                tiles[selected].append(column)
                sizes[selected] += 1
                counts[selected] = updated[selected]
            if success:
                return [tile for tile in tiles if tile]
    raise RuntimeError("optimized column assignment failed")


def optimized_arrange_24(
    columns: list[int], column_masks: dict[int, int], seed: int
) -> tuple[list[int], bool]:
    group_count = K_TILE // 4
    bits_by_column = {
        column: np.fromiter(
            ((mask >> row) & 1 for row in range(M_TILE)),
            dtype=np.int8,
            count=M_TILE,
        )
        for column, mask in column_masks.items()
        if column in columns
    }
    rng = random.Random(seed)
    for _ in range(16):
        tie_break = {column: rng.random() for column in columns}
        ordered = sorted(
            columns,
            key=lambda column: (
                -column_masks[column].bit_count(), tie_break[column]
            ),
        )
        groups: list[list[int]] = [[] for _ in range(group_count)]
        sizes = np.zeros(group_count, dtype=np.int8)
        counts = np.zeros((group_count, M_TILE), dtype=np.int8)
        success = True
        for column in ordered:
            bits = bits_by_column[column]
            updated = counts + bits[None, :]
            valid = (sizes < 4) & (np.max(updated, axis=1) <= 2)
            if not np.any(valid):
                success = False
                break
            score = (counts @ bits).astype(np.float64) * 100.0
            score += sizes.astype(np.float64)
            score += np.asarray(
                [rng.random() * 1e-6 for _ in range(group_count)]
            )
            score[~valid] = np.inf
            selected = int(np.argmin(score))
            groups[selected].append(column)
            sizes[selected] += 1
            counts[selected] = updated[selected]
        if success:
            for group in sorted(groups, key=len):
                while len(group) < 4:
                    group.append(-1)
            return [column for group in groups for column in group], True
    padded = columns + [-1] * (K_TILE - len(columns))
    return padded, False


_parallel_view: SparseRows | None = None


def _optimized_pack_group(task: tuple[int, list[int], int]) -> list[PackedTile]:
    group_index, local_rows, seed = task
    if _parallel_view is None:
        raise RuntimeError("parallel planner view is not initialized")
    view = _parallel_view
    row_ids = np.asarray(
        [int(view.row_ids[row]) if row >= 0 else -1 for row in local_rows],
        dtype=np.int32,
    )
    row_maps = [
        dict(zip(map(int, view.cols[row]), map(float, view.vals[row]), strict=True))
        if row >= 0
        else {}
        for row in local_rows
    ]
    masks: dict[int, int] = {}
    for row_slot, row_map in enumerate(row_maps):
        for column in row_map:
            masks[column] = masks.get(column, 0) | (1 << row_slot)
    column_tiles = optimized_assign_columns(masks, seed + group_index * 1009)
    tiles: list[PackedTile] = []
    for tile_index, columns in enumerate(column_tiles):
        arranged, legal = optimized_arrange_24(
            columns, masks, seed + group_index * 1009 + tile_index * 9176
        )
        col_ids = np.asarray(arranged, dtype=np.int32)
        values = np.zeros((M_TILE, K_TILE), dtype=np.float64)
        for row_slot, row_map in enumerate(row_maps):
            values[row_slot] = np.fromiter(
                (
                    row_map.get(column, 0.0) if column >= 0 else 0.0
                    for column in arranged
                ),
                dtype=np.float64,
                count=K_TILE,
            )
        tile = PackedTile(
            row_ids=row_ids.copy(),
            col_ids=col_ids,
            values=values,
            legal_24=legal,
            violation_count=0,
        )
        if legal:
            tiles.append(tile)
        else:
            tiles.extend(split_illegal_tile(tile))
    return tiles


def optimized_split24_pack(
    view: SparseRows, policy: str, seed: int, workers: int
) -> PackedView:
    started = time.perf_counter()
    order = row_order(view, policy, K_TILE)
    tasks = []
    for group_index, offset in enumerate(range(0, len(order), M_TILE)):
        local_rows = order[offset : offset + M_TILE]
        local_rows += [-1] * (M_TILE - len(local_rows))
        tasks.append((group_index, local_rows, seed))
    global _parallel_view
    _parallel_view = view
    if workers > 1:
        context = mp.get_context("fork")
        with context.Pool(processes=workers) as pool:
            grouped_tiles = pool.map(_optimized_pack_group, tasks, chunksize=8)
    else:
        grouped_tiles = [_optimized_pack_group(task) for task in tasks]
    tiles = [tile for group in grouped_tiles for tile in group]
    _parallel_view = None
    return PackedView(
        source_name=view.name,
        output_size=view.output_size,
        n_cols=view.n_cols,
        m_tile=M_TILE,
        k_tile=K_TILE,
        policy=f"optimized-{policy}",
        tiles=tiles,
        pack_seconds=time.perf_counter() - started,
    )


def compress_tiles_vectorized(
    tiles: list[PackedTile],
) -> tuple[np.ndarray, np.ndarray]:
    rounded = np.stack([tile.values for tile in tiles]).astype(np.float16)
    groups = rounded.reshape(len(tiles), M_TILE, K_TILE // 4, 4)
    positions = np.arange(4, dtype=np.uint32)
    priority = np.where(groups != 0, positions, positions + 4)
    selected = np.sort(np.argsort(priority, axis=-1)[..., :2], axis=-1).astype(
        np.uint32
    )
    compressed = np.take_along_axis(groups, selected, axis=-1).reshape(
        len(tiles), M_TILE, K_TILE // 2
    )
    codes = selected[..., 0] | (selected[..., 1] << np.uint32(2))
    shifts = (np.arange(K_TILE // 4, dtype=np.uint32) * np.uint32(4))[None, None, :]
    row_meta = np.bitwise_or.reduce(codes << shifts, axis=2)
    pair_meta = np.zeros((len(tiles), M_TILE), dtype=np.uint32)
    for group in range(8):
        first = row_meta[:, group]
        second = row_meta[:, group + 8]
        pair_meta[:, group * 2] = (first & 0xFFFF) | ((second & 0xFFFF) << 16)
        pair_meta[:, group * 2 + 1] = (first >> 16) | (second & 0xFFFF0000)
    return compressed, pair_meta


def sparse_rows_from_coalesced(
    name: str,
    indices: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
    coords: np.ndarray,
) -> SparseRows:
    row_counts = np.bincount(indices[0], minlength=shape[0])
    row_ptr = np.zeros(shape[0] + 1, dtype=np.int64)
    row_ptr[1:] = np.cumsum(row_counts)
    rows_cols = tuple(
        indices[1, row_ptr[row] : row_ptr[row + 1]].astype(np.int32, copy=True)
        for row in range(shape[0])
    )
    rows_vals = tuple(
        values[row_ptr[row] : row_ptr[row + 1]].astype(np.float64, copy=True)
        for row in range(shape[0])
    )
    return SparseRows(
        name=name,
        output_size=shape[0],
        n_cols=shape[1],
        row_ids=np.arange(shape[0], dtype=np.int32),
        coords=coords.astype(np.float64, copy=True),
        cols=rows_cols,
        vals=rows_vals,
    )


def materialize_component_tiles(
    structure_tiles: list[PackedTile], component: SparseRows
) -> list[PackedTile]:
    row_lookup = {int(row_id): local for local, row_id in enumerate(component.row_ids)}
    row_maps = [
        dict(zip(map(int, cols), map(float, vals), strict=True))
        for cols, vals in zip(component.cols, component.vals, strict=True)
    ]
    result: list[PackedTile] = []
    for structure_tile in structure_tiles:
        values = np.zeros_like(structure_tile.values)
        for row_slot, logical_row in enumerate(structure_tile.row_ids):
            if logical_row < 0:
                continue
            row_map = row_maps[row_lookup[int(logical_row)]]
            for col_slot, logical_col in enumerate(structure_tile.col_ids):
                if logical_col >= 0 and structure_tile.values[row_slot, col_slot] != 0.0:
                    values[row_slot, col_slot] = row_map.get(int(logical_col), 0.0)
        tile = PackedTile(
            row_ids=structure_tile.row_ids.copy(),
            col_ids=structure_tile.col_ids.copy(),
            values=values,
            legal_24=True,
            violation_count=0,
        )
        if not validate_24(tile)[0]:
            raise RuntimeError("materialized component is not legal 2:4")
        result.append(tile)
    return result


def export_component(
    output_dir: Path,
    component_name: str,
    component: SparseRows,
    group_offsets: np.ndarray,
    group_row_ids: np.ndarray,
    structure_tiles: list[PackedTile],
    dense_input: np.ndarray,
    common_manifest: dict[str, object],
    production_only: bool,
) -> dict[str, object]:
    start = time.perf_counter()
    tiles = materialize_component_tiles(structure_tiles, component)
    tile_col_ids = np.stack([tile.col_ids for tile in tiles]).astype(np.int32)
    tile_col_ids[tile_col_ids < 0] = 0
    tile_a_comp, tile_pair_meta = compress_tiles_vectorized(tiles)

    csr_vals = np.concatenate(component.vals).astype(np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<Q11I",
        MAGIC,
        VERSION,
        N_CHANNELS,
        component.output_size,
        component.n_cols,
        component.n_active_rows,
        component.nnz,
        len(group_offsets) - 1,
        len(tiles),
        0,
        M_TILE,
        K_TILE,
    )
    (output_dir / "header.bin").write_bytes(header)
    files = [
        {
            "file": "header.bin",
            "dtype": "packed_header_v2",
            "shape": [1],
            "bytes": len(header),
            "sha256": hashlib.sha256(header).hexdigest(),
        },
        write_array(output_dir / "group_offsets.bin", group_offsets),
        write_array(output_dir / "group_row_ids.bin", group_row_ids),
        write_array(output_dir / "tile_col_ids.bin", tile_col_ids),
        write_array(output_dir / "tile_a_comp.bin", tile_a_comp),
        write_array(output_dir / "tile_pair_meta.bin", tile_pair_meta),
    ]
    if not production_only:
        csr_row_ptr = np.zeros(component.n_active_rows + 1, dtype=np.uint32)
        csr_row_ptr[1:] = np.cumsum(
            [len(cols) for cols in component.cols], dtype=np.uint64
        )
        csr_row_ids = component.row_ids.astype(np.int32, copy=True)
        csr_cols = np.concatenate(component.cols).astype(np.int32)
        reference = np.zeros((component.output_size, N_CHANNELS), dtype=np.float32)
        dense_f32 = dense_input.astype(np.float32)
        for local_row, (cols, vals) in enumerate(
            zip(component.cols, component.vals, strict=True)
        ):
            reference[int(component.row_ids[local_row])] = (
                vals.astype(np.float16).astype(np.float32) @ dense_f32[cols]
            )
        empty_cols = np.empty(0, dtype=np.int32)
        empty_vals = np.empty(0, dtype=np.float16)
        residual_row_ptr = np.zeros(component.output_size + 1, dtype=np.uint32)
        files.extend(
            [
                write_array(output_dir / "csr_row_ptr.bin", csr_row_ptr),
                write_array(output_dir / "csr_row_ids.bin", csr_row_ids),
                write_array(output_dir / "csr_cols.bin", csr_cols),
                write_array(output_dir / "csr_vals.bin", csr_vals),
                write_array(output_dir / "residual_row_ptr.bin", residual_row_ptr),
                write_array(output_dir / "residual_cols.bin", empty_cols),
                write_array(output_dir / "residual_vals.bin", empty_vals),
                write_array(output_dir / "dense_input.bin", dense_input),
                write_array(output_dir / "reference.bin", reference),
            ]
        )
    nonzero_fp16 = int(np.count_nonzero(csr_vals))
    slots = len(tiles) * M_TILE * K_TILE // 2
    manifest = {
        **common_manifest,
        "component": component_name,
        "output_size": component.output_size,
        "logical_k": component.n_cols,
        "active_rows": component.n_active_rows,
        "nnz_structural": component.nnz,
        "nnz_nonzero_fp16": nonzero_fp16,
        "groups": len(group_offsets) - 1,
        "tiles": len(tiles),
        "sparse_value_slots": slots,
        "fp16_useful_slot_utilization": nonzero_fp16 / slots,
        "component_export_seconds": time.perf_counter() - start,
        "production_only": production_only,
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


_component_export_context: dict[str, object] | None = None


def _export_component_task(task: tuple[str, str]) -> dict[str, object]:
    component_name, output_string = task
    if _component_export_context is None:
        raise RuntimeError("component export context is not initialized")
    components = _component_export_context["components"]
    return export_component(
        Path(output_string),
        component_name,
        components[component_name],
        _component_export_context["group_offsets"],
        _component_export_context["group_row_ids"],
        _component_export_context["structure_tiles"],
        _component_export_context["dense_input"],
        _component_export_context["common"],
        bool(_component_export_context["production_only"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-file", type=Path, required=True)
    parser.add_argument("--view", choices=("G", "G_T"), required=True)
    parser.add_argument("--im-size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--numpoints", type=int, default=6)
    parser.add_argument("--policy", default="overlap")
    parser.add_argument("--fast-split24", action="store_true")
    parser.add_argument("--optimized-pack", action="store_true")
    parser.add_argument("--planner-workers", type=int, default=1)
    parser.add_argument("--component-workers", type=int, default=1)
    parser.add_argument("--production-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--experiment-id",
        default="trajsparsenufft_gpu_export",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    trajectory = np.load(args.trajectory_file).reshape(-1, 2).astype(np.float32)
    omega = torch.from_numpy((trajectory.T * 2.0 * np.pi).astype(np.float32))
    real_tensor, imag_tensor = tkbn.calc_tensor_spmatrix(
        omega,
        (args.im_size, args.im_size),
        grid_size=(args.grid_size, args.grid_size),
        numpoints=args.numpoints,
    )
    real_tensor = real_tensor.coalesce()
    imag_tensor = imag_tensor.coalesce()
    if not torch.equal(real_tensor.indices(), imag_tensor.indices()):
        raise RuntimeError("TorchKbNufft real/imag matrices have different structure")
    indices = real_tensor.indices().cpu().numpy()
    real_values = real_tensor.values().cpu().numpy()
    imag_values = imag_tensor.values().cpu().numpy()
    shape = tuple(map(int, real_tensor.shape))
    coords = np.mod(
        trajectory.astype(np.float64) * args.grid_size + args.grid_size / 2.0,
        args.grid_size,
    )
    real_forward = sparse_rows_from_coalesced(
        "TKBN_REAL_G", indices, real_values, shape, coords
    )
    imag_forward = sparse_rows_from_coalesced(
        "TKBN_IMAG_G", indices, imag_values, shape, coords
    )
    structure_forward = sparse_rows_from_coalesced(
        "TKBN_STRUCTURE_G",
        indices,
        np.ones_like(real_values, dtype=np.float32),
        shape,
        coords,
    )
    if args.view == "G_T":
        real_view = transpose_sparse(real_forward, args.grid_size)
        imag_view = transpose_sparse(imag_forward, args.grid_size)
        structure_view = transpose_sparse(structure_forward, args.grid_size)
    else:
        real_view, imag_view, structure_view = (
            real_forward,
            imag_forward,
            structure_forward,
        )

    if args.fast_split24 and args.optimized_pack:
        raise ValueError("select only one fast packing path")
    if args.fast_split24:
        packed_structure = fast_split24_pack(structure_view, args.policy, args.seed)
    elif args.optimized_pack:
        packed_structure = optimized_split24_pack(
            structure_view, args.policy, args.seed, args.planner_workers
        )
    else:
        packed_structure = pack_view(structure_view, args.policy, K_TILE, args.seed)
    (
        group_offsets,
        group_row_ids,
        structure_tiles,
        _,
        residual_cols,
        _,
    ) = split_legal_tiles_and_residual(structure_view, packed_structure, "split24")
    if len(residual_cols):
        raise RuntimeError("split24 structure export left residual entries")

    rng = np.random.default_rng(args.seed)
    dense_input = np.clip(
        rng.standard_normal((structure_view.n_cols, N_CHANNELS)), -3.0, 3.0
    ).astype(np.float16)
    asset_sha = hashlib.sha256(args.trajectory_file.read_bytes()).hexdigest()
    common = {
        "experiment_id": args.experiment_id,
        "source": "TorchKbNufft calc_tensor_spmatrix",
        "torchkbnufft_version": getattr(tkbn, "__version__", "unknown"),
        "trajectory_file": str(args.trajectory_file),
        "trajectory_file_sha256": asset_sha,
        "view": args.view,
        "im_size": args.im_size,
        "grid_size": args.grid_size,
        "numpoints": args.numpoints,
        "samples": shape[0],
        "policy": args.policy,
        "illegal_mode": "split24",
        "seed": args.seed,
        "n_channels": N_CHANNELS,
        "m_tile": M_TILE,
        "k_tile": K_TILE,
        "shared_structure_pack_seconds": time.perf_counter() - started,
    }
    global _component_export_context
    _component_export_context = {
        "components": {"real": real_view, "imag": imag_view},
        "group_offsets": group_offsets,
        "group_row_ids": group_row_ids,
        "structure_tiles": structure_tiles,
        "dense_input": dense_input,
        "common": common,
        "production_only": args.production_only,
    }
    component_tasks = [
        ("real", str(args.output_root / "real")),
        ("imag", str(args.output_root / "imag")),
    ]
    if args.component_workers > 1:
        context = mp.get_context("fork")
        with context.Pool(processes=min(2, args.component_workers)) as pool:
            manifests = pool.map(_export_component_task, component_tasks)
    else:
        manifests = [_export_component_task(task) for task in component_tasks]
    _component_export_context = None
    print(json.dumps(manifests, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
