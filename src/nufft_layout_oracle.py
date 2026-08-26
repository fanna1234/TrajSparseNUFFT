#!/usr/bin/env python3
"""Lossless dual-view block-2:4 packing oracle for NUFFT interpolation.

This program is a structural CPU oracle. It does not measure GPU performance.
It generates compact-support interpolation matrices for synthetic radial and
spiral trajectories, packs both G and G^T into outer sparse / inner 2:4 tiles,
and checks exact coefficient preservation and adjointness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


M_TILE = 16
DEFAULT_K_TILES = (32, 64)
ROW_GROUPING_POLICIES = ("trajectory", "morton", "overlap", "packing256")


@dataclass(frozen=True)
class SparseRows:
    """Sparse row collection with a mapping to logical output rows."""

    name: str
    output_size: int
    n_cols: int
    row_ids: np.ndarray
    coords: np.ndarray
    cols: tuple[np.ndarray, ...]
    vals: tuple[np.ndarray, ...]

    @property
    def n_active_rows(self) -> int:
        return len(self.cols)

    @property
    def nnz(self) -> int:
        return int(sum(len(row) for row in self.cols))


@dataclass
class PackedTile:
    row_ids: np.ndarray
    col_ids: np.ndarray
    values: np.ndarray
    legal_24: bool
    violation_count: int

    @property
    def nnz(self) -> int:
        return int(np.count_nonzero(self.values))


@dataclass
class PackedView:
    source_name: str
    output_size: int
    n_cols: int
    m_tile: int
    k_tile: int
    policy: str
    tiles: list[PackedTile]
    pack_seconds: float


@dataclass(frozen=True)
class ViewMetrics:
    case_id: str
    view: str
    policy: str
    m_tile: int
    k_tile: int
    active_rows: int
    logical_rows: int
    logical_cols: int
    nnz: int
    tile_count: int
    legal_tile_count: int
    residual_tile_count: int
    residual_nnz: int
    residual_fraction: float
    sparse_value_slots: int
    useful_slot_utilization: float
    useful_dense_equiv_efficiency: float
    packed_storage_bytes: int
    reference_csr_bytes: int
    packed_to_csr_ratio: float
    whole_matrix_half_dense_slots: int
    whole_matrix_slot_amplification: float
    max_tile_violation: int
    pack_seconds: float
    apply_max_abs_error: float
    checksum_match: bool


def morton2d(x: int, y: int) -> int:
    """Interleave two nonnegative 16-bit integers."""

    def spread(value: int) -> int:
        value &= 0xFFFF
        value = (value | (value << 8)) & 0x00FF00FF
        value = (value | (value << 4)) & 0x0F0F0F0F
        value = (value | (value << 2)) & 0x33333333
        value = (value | (value << 1)) & 0x55555555
        return value

    return spread(x) | (spread(y) << 1)


def generate_trajectory(kind: str, grid_size: int, n_samples: int) -> np.ndarray:
    """Generate deterministic sample coordinates in Cartesian-grid units."""

    center = (grid_size - 1) / 2.0
    radius_max = 0.43 * grid_size
    if kind == "radial":
        n_spokes = max(16, int(round(math.sqrt(n_samples))))
        per_spoke = int(math.ceil(n_samples / n_spokes))
        golden = math.pi * (3.0 - math.sqrt(5.0))
        coords: list[tuple[float, float]] = []
        radii = np.linspace(-radius_max, radius_max, per_spoke, endpoint=True)
        for spoke in range(n_spokes):
            angle = spoke * golden
            cosine, sine = math.cos(angle), math.sin(angle)
            for radius in radii:
                coords.append((center + radius * cosine, center + radius * sine))
        return np.asarray(coords[:n_samples], dtype=np.float64)
    if kind == "spiral":
        t = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples
        turns = max(6.0, grid_size / 8.0)
        radius = radius_max * np.sqrt(t)
        angle = 2.0 * math.pi * turns * t
        return np.column_stack(
            (center + radius * np.cos(angle), center + radius * np.sin(angle))
        )
    raise ValueError(f"unsupported trajectory kind: {kind}")


def kaiser_bessel_weight(distance: float, width: int, beta: float) -> float:
    scaled = 2.0 * distance / width
    if abs(scaled) >= 1.0:
        return 0.0
    return float(np.i0(beta * math.sqrt(max(0.0, 1.0 - scaled * scaled))) / np.i0(beta))


def build_interpolation(
    trajectory: np.ndarray,
    grid_size: int,
    width: int,
    oversampling: float = 2.0,
    periodic: bool = False,
) -> SparseRows:
    """Build a normalized separable compact-support interpolation matrix."""

    beta_term = (width / oversampling * (oversampling - 0.5)) ** 2 - 0.8
    beta = math.pi * math.sqrt(max(beta_term, 0.01))
    rows_cols: list[np.ndarray] = []
    rows_vals: list[np.ndarray] = []
    half = width // 2
    for x_coord, y_coord in trajectory:
        x_start = int(math.floor(x_coord)) - half + 1
        y_start = int(math.floor(y_coord)) - half + 1
        entries: list[tuple[int, float]] = []
        for gy in range(y_start, y_start + width):
            if not periodic and (gy < 0 or gy >= grid_size):
                continue
            wy = kaiser_bessel_weight(y_coord - gy, width, beta)
            for gx in range(x_start, x_start + width):
                if not periodic and (gx < 0 or gx >= grid_size):
                    continue
                wx = kaiser_bessel_weight(x_coord - gx, width, beta)
                value = wx * wy
                if value != 0.0:
                    mapped_y = gy % grid_size if periodic else gy
                    mapped_x = gx % grid_size if periodic else gx
                    entries.append((mapped_y * grid_size + mapped_x, value))
        norm = sum(value for _, value in entries)
        if norm == 0.0:
            raise RuntimeError("empty interpolation row")
        entries.sort(key=lambda item: item[0])
        rows_cols.append(np.asarray([col for col, _ in entries], dtype=np.int32))
        rows_vals.append(
            np.asarray([value / norm for _, value in entries], dtype=np.float64)
        )
    return SparseRows(
        name="G",
        output_size=len(trajectory),
        n_cols=grid_size * grid_size,
        row_ids=np.arange(len(trajectory), dtype=np.int32),
        coords=np.asarray(trajectory, dtype=np.float64),
        cols=tuple(rows_cols),
        vals=tuple(rows_vals),
    )


def transpose_sparse(view: SparseRows, grid_size: int) -> SparseRows:
    """Build the exact active-row transpose of a sparse view."""

    col_rows: list[list[int]] = [[] for _ in range(view.n_cols)]
    col_vals: list[list[float]] = [[] for _ in range(view.n_cols)]
    for local_row, (cols, vals) in enumerate(zip(view.cols, view.vals, strict=True)):
        logical_row = int(view.row_ids[local_row])
        for col, value in zip(cols, vals, strict=True):
            col_rows[int(col)].append(logical_row)
            col_vals[int(col)].append(float(value))
    active = [index for index, rows in enumerate(col_rows) if rows]
    coords = np.asarray(
        [(index % grid_size, index // grid_size) for index in active], dtype=np.float64
    )
    return SparseRows(
        name="G_T",
        output_size=view.n_cols,
        n_cols=view.output_size,
        row_ids=np.asarray(active, dtype=np.int32),
        coords=coords,
        cols=tuple(np.asarray(col_rows[index], dtype=np.int32) for index in active),
        vals=tuple(np.asarray(col_vals[index], dtype=np.float64) for index in active),
    )


def apply_sparse(view: SparseRows, dense: np.ndarray) -> np.ndarray:
    if dense.shape[0] != view.n_cols:
        raise ValueError("dense input has incompatible K dimension")
    result = np.zeros((view.output_size, dense.shape[1]), dtype=np.complex128)
    for local_row, (cols, vals) in enumerate(zip(view.cols, view.vals, strict=True)):
        result[int(view.row_ids[local_row])] = vals @ dense[cols]
    return result


def row_order(view: SparseRows, policy: str, k_tile: int = 32) -> list[int]:
    if policy == "trajectory":
        return list(range(view.n_active_rows))
    morton_order = sorted(
        range(view.n_active_rows),
        key=lambda row: morton2d(
            max(0, int(round(view.coords[row, 0]))),
            max(0, int(round(view.coords[row, 1]))),
        ),
    )
    if policy == "morton":
        return morton_order
    if policy not in ("overlap", "packing256"):
        raise ValueError(f"unknown row grouping policy: {policy}")

    # Restrict the incidence search to Morton-local windows. This keeps the
    # planner deterministic and sub-quadratic while favoring common supports.
    supports = [set(map(int, cols)) for cols in view.cols]
    ordered: list[int] = []
    window_size = 4 * M_TILE if policy == "overlap" else 16 * M_TILE
    for start in range(0, len(morton_order), window_size):
        remaining = list(morton_order[start : start + window_size])
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            union = set(supports[seed])
            while remaining and len(group) < M_TILE:
                if policy == "overlap":
                    best_pos = max(
                        range(len(remaining)),
                        key=lambda pos: (
                            len(supports[remaining[pos]] & union),
                            -len(supports[remaining[pos]] - union),
                            -pos,
                        ),
                    )
                else:
                    current_nnz = sum(len(supports[row]) for row in group)
                    current_max_degree = max(len(supports[row]) for row in group)

                    def packing_score(pos: int) -> tuple[float, int, int, int]:
                        candidate = remaining[pos]
                        candidate_support = supports[candidate]
                        combined_union = union | candidate_support
                        maximum_degree = max(current_max_degree, len(candidate_support))
                        estimated_tiles = max(
                            math.ceil(len(combined_union) / k_tile),
                            math.ceil(maximum_degree / (k_tile // 2)),
                            1,
                        )
                        issued_slots = estimated_tiles * M_TILE * k_tile // 2
                        predicted_utilization = (
                            current_nnz + len(candidate_support)
                        ) / issued_slots
                        return (
                            predicted_utilization,
                            len(candidate_support & union),
                            -len(combined_union),
                            -pos,
                        )

                    best_pos = max(range(len(remaining)), key=packing_score)
                chosen = remaining.pop(best_pos)
                group.append(chosen)
                union.update(supports[chosen])
            ordered.extend(group)
    return ordered


def _column_masks(view: SparseRows, local_rows: Sequence[int]) -> dict[int, int]:
    masks: dict[int, int] = {}
    for slot, local_row in enumerate(local_rows):
        if local_row < 0:
            continue
        for col in view.cols[local_row]:
            col_int = int(col)
            masks[col_int] = masks.get(col_int, 0) | (1 << slot)
    return masks


def _assign_columns_to_tiles(
    column_masks: dict[int, int], k_tile: int, seed: int
) -> list[list[int]]:
    """Partition columns while respecting per-row half-density capacity."""

    columns = list(column_masks)
    if not columns:
        return [[]]
    max_row_degree = max(
        sum(1 for mask in column_masks.values() if mask & (1 << row))
        for row in range(M_TILE)
    )
    minimum_tiles = max(
        math.ceil(len(columns) / k_tile),
        math.ceil(max_row_degree / (k_tile // 2)),
        1,
    )
    rng = random.Random(seed)
    for tile_count in range(minimum_tiles, minimum_tiles + 9):
        for restart in range(24):
            tie_break = {col: rng.random() for col in columns}
            ordered = sorted(
                columns,
                key=lambda col: (-column_masks[col].bit_count(), tie_break[col]),
            )
            tiles: list[list[int]] = [[] for _ in range(tile_count)]
            row_counts = np.zeros((tile_count, M_TILE), dtype=np.int16)
            success = True
            for col in ordered:
                mask = column_masks[col]
                bits = np.asarray([(mask >> row) & 1 for row in range(M_TILE)])
                candidates: list[tuple[float, int]] = []
                for tile_index in range(tile_count):
                    if len(tiles[tile_index]) >= k_tile:
                        continue
                    updated = row_counts[tile_index] + bits
                    if int(updated.max(initial=0)) > k_tile // 2:
                        continue
                    delta_square = float(np.sum(updated * updated - row_counts[tile_index] ** 2))
                    capacity_pressure = len(tiles[tile_index]) / k_tile
                    candidates.append((delta_square + capacity_pressure + rng.random() * 1e-6, tile_index))
                if not candidates:
                    success = False
                    break
                _, selected = min(candidates)
                tiles[selected].append(col)
                row_counts[selected] += bits
            if success:
                return [tile for tile in tiles if tile]
    raise RuntimeError("failed to partition columns into half-density tiles")


def _arrange_24(
    columns: Sequence[int], column_masks: dict[int, int], k_tile: int, seed: int
) -> tuple[list[int], bool, int]:
    """Arrange columns into quartets with at most two nonzeros per row."""

    if len(columns) > k_tile:
        raise ValueError("too many columns for tile")
    group_count = k_tile // 4
    rng = random.Random(seed)
    best_columns: list[int] | None = None
    best_violation = sys.maxsize
    for restart in range(64):
        tie_break = {col: rng.random() for col in columns}
        ordered = sorted(
            columns,
            key=lambda col: (-column_masks[col].bit_count(), tie_break[col]),
        )
        groups: list[list[int]] = [[] for _ in range(group_count)]
        counts = np.zeros((group_count, M_TILE), dtype=np.int8)
        success = True
        for col in ordered:
            mask = column_masks[col]
            bits = np.asarray([(mask >> row) & 1 for row in range(M_TILE)], dtype=np.int8)
            choices: list[tuple[float, int]] = []
            for group_index in range(group_count):
                if len(groups[group_index]) >= 4:
                    continue
                updated = counts[group_index] + bits
                if int(updated.max(initial=0)) > 2:
                    continue
                overlap = int(np.dot(counts[group_index], bits))
                choices.append(
                    (
                        overlap * 100.0
                        + len(groups[group_index])
                        + rng.random() * 1e-6,
                        group_index,
                    )
                )
            if not choices:
                success = False
                break
            _, selected = min(choices)
            groups[selected].append(col)
            counts[selected] += bits
        if not success:
            continue
        for group in sorted(groups, key=len):
            while len(group) < 4:
                group.append(-1)
        flattened = [col for group in groups for col in group]
        return flattened, True, 0

    # Preserve a deterministic illegal layout as residual evidence.
    padded = list(columns) + [-1] * (k_tile - len(columns))
    for offset in range(0, k_tile, 4):
        quartet = padded[offset : offset + 4]
        violation = 0
        for row in range(M_TILE):
            count = sum(
                1
                for col in quartet
                if col >= 0 and (column_masks[col] & (1 << row))
            )
            violation += max(0, count - 2)
        best_violation = min(best_violation, violation)
    best_columns = padded
    return best_columns, False, int(best_violation)


def pack_view(view: SparseRows, policy: str, k_tile: int, seed: int) -> PackedView:
    start_time = time.perf_counter()
    order = row_order(view, policy, k_tile)
    tiles: list[PackedTile] = []
    for group_index, offset in enumerate(range(0, len(order), M_TILE)):
        local_rows = order[offset : offset + M_TILE]
        local_rows += [-1] * (M_TILE - len(local_rows))
        masks = _column_masks(view, local_rows)
        column_tiles = _assign_columns_to_tiles(
            masks, k_tile, seed + group_index * 1009
        )
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
        for tile_index, columns in enumerate(column_tiles):
            arranged, legal, violation = _arrange_24(
                columns,
                masks,
                k_tile,
                seed + group_index * 1009 + tile_index * 9176,
            )
            col_ids = np.asarray(arranged, dtype=np.int32)
            values = np.zeros((M_TILE, k_tile), dtype=np.float64)
            for row_slot, row_map in enumerate(row_maps):
                for col_slot, col in enumerate(arranged):
                    if col >= 0:
                        values[row_slot, col_slot] = row_map.get(col, 0.0)
            tiles.append(
                PackedTile(
                    row_ids=row_ids.copy(),
                    col_ids=col_ids,
                    values=values,
                    legal_24=legal,
                    violation_count=violation,
                )
            )
    return PackedView(
        source_name=view.name,
        output_size=view.output_size,
        n_cols=view.n_cols,
        m_tile=M_TILE,
        k_tile=k_tile,
        policy=policy,
        tiles=tiles,
        pack_seconds=time.perf_counter() - start_time,
    )


def apply_packed(packed: PackedView, dense: np.ndarray) -> np.ndarray:
    if dense.shape[0] != packed.n_cols:
        raise ValueError("dense input has incompatible K dimension")
    result = np.zeros((packed.output_size, dense.shape[1]), dtype=np.complex128)
    for tile in packed.tiles:
        valid_cols = np.flatnonzero(tile.col_ids >= 0)
        if not len(valid_cols):
            continue
        cols = tile.col_ids[valid_cols]
        partial = tile.values[:, valid_cols] @ dense[cols]
        for row_slot, logical_row in enumerate(tile.row_ids):
            if logical_row >= 0:
                result[int(logical_row)] += partial[row_slot]
    return result


def _hash_records(records: Iterable[tuple[int, int, float]]) -> str:
    digest = hashlib.sha256()
    for row, col, value in sorted(records):
        digest.update(struct.pack("<qqd", row, col, value))
    return digest.hexdigest()


def sparse_checksum(view: SparseRows) -> str:
    records = (
        (int(view.row_ids[local_row]), int(col), float(value))
        for local_row, (cols, vals) in enumerate(zip(view.cols, view.vals, strict=True))
        for col, value in zip(cols, vals, strict=True)
    )
    return _hash_records(records)


def packed_checksum(packed: PackedView) -> str:
    records = (
        (int(tile.row_ids[row]), int(tile.col_ids[col]), float(tile.values[row, col]))
        for tile in packed.tiles
        for row in range(tile.values.shape[0])
        if tile.row_ids[row] >= 0
        for col in range(tile.values.shape[1])
        if tile.col_ids[col] >= 0 and tile.values[row, col] != 0.0
    )
    return _hash_records(records)


def canonical_operator_checksum(view: SparseRows, transposed: bool) -> str:
    if not transposed:
        return sparse_checksum(view)
    records = (
        (int(col), int(view.row_ids[local_row]), float(value))
        for local_row, (cols, vals) in enumerate(zip(view.cols, view.vals, strict=True))
        for col, value in zip(cols, vals, strict=True)
    )
    return _hash_records(records)


def reference_csr_bytes(view: SparseRows) -> int:
    # FP16 values + int32 columns + int32 row pointer.
    return view.nnz * (2 + 4) + (view.output_size + 1) * 4


def packed_storage_bytes(packed: PackedView) -> int:
    total = 0
    for tile in packed.tiles:
        if tile.legal_24:
            value_bytes = packed.m_tile * packed.k_tile // 2 * 2
            metadata_bytes = packed.m_tile * packed.k_tile // 8
            map_bytes = packed.k_tile * 4 + packed.m_tile * 4
            total += value_bytes + metadata_bytes + map_bytes + 32
        else:
            total += tile.nnz * (2 + 4) + (packed.m_tile + 1) * 4 + 32
    return total


def validate_24(tile: PackedTile) -> tuple[bool, int]:
    violations = 0
    for offset in range(0, tile.values.shape[1], 4):
        counts = np.count_nonzero(tile.values[:, offset : offset + 4], axis=1)
        violations += int(np.maximum(counts - 2, 0).sum())
    return violations == 0, violations


def measure_view(
    case_id: str,
    view: SparseRows,
    packed: PackedView,
    dense: np.ndarray,
) -> ViewMetrics:
    reference = apply_sparse(view, dense)
    candidate = apply_packed(packed, dense)
    error = float(np.max(np.abs(reference - candidate), initial=0.0))
    residual_nnz = 0
    legal_count = 0
    maximum_violation = 0
    for tile in packed.tiles:
        legal, violations = validate_24(tile)
        if legal != tile.legal_24:
            raise AssertionError("stored and recomputed 2:4 legality disagree")
        if legal:
            legal_count += 1
        else:
            residual_nnz += tile.nnz
        maximum_violation = max(maximum_violation, violations)
    slots = legal_count * packed.m_tile * packed.k_tile // 2
    legal_nnz = view.nnz - residual_nnz
    utilization = legal_nnz / slots if slots else 0.0
    packed_bytes = packed_storage_bytes(packed)
    csr_bytes = reference_csr_bytes(view)
    whole_slots = view.output_size * view.n_cols // 2
    return ViewMetrics(
        case_id=case_id,
        view=view.name,
        policy=packed.policy,
        m_tile=packed.m_tile,
        k_tile=packed.k_tile,
        active_rows=view.n_active_rows,
        logical_rows=view.output_size,
        logical_cols=view.n_cols,
        nnz=view.nnz,
        tile_count=len(packed.tiles),
        legal_tile_count=legal_count,
        residual_tile_count=len(packed.tiles) - legal_count,
        residual_nnz=residual_nnz,
        residual_fraction=residual_nnz / view.nnz if view.nnz else 0.0,
        sparse_value_slots=slots,
        useful_slot_utilization=utilization,
        useful_dense_equiv_efficiency=2.0 * utilization,
        packed_storage_bytes=packed_bytes,
        reference_csr_bytes=csr_bytes,
        packed_to_csr_ratio=packed_bytes / csr_bytes,
        whole_matrix_half_dense_slots=whole_slots,
        whole_matrix_slot_amplification=whole_slots / view.nnz,
        max_tile_violation=maximum_violation,
        pack_seconds=packed.pack_seconds,
        apply_max_abs_error=error,
        checksum_match=sparse_checksum(view) == packed_checksum(packed),
    )


def metric_rank(metric: ViewMetrics) -> tuple[int, int, float, float, float]:
    return (
        int(metric.checksum_match and metric.apply_max_abs_error <= 1e-12),
        int(metric.residual_fraction <= 0.10),
        metric.useful_slot_utilization,
        -metric.packed_to_csr_ratio,
        -metric.pack_seconds,
    )


def run_case(
    trajectory_kind: str,
    grid_size: int,
    width: int,
    n_samples: int,
    seed: int,
) -> tuple[list[ViewMetrics], dict[str, object]]:
    case_id = f"{trajectory_kind}_g{grid_size}_j{width}_m{n_samples}"
    trajectory = generate_trajectory(trajectory_kind, grid_size, n_samples)
    forward = build_interpolation(trajectory, grid_size, width)
    adjoint = transpose_sparse(forward, grid_size)
    if canonical_operator_checksum(forward, False) != canonical_operator_checksum(adjoint, True):
        raise AssertionError("forward and transpose coefficient sets differ")

    rng = np.random.default_rng(seed)
    grid_dense = rng.standard_normal((forward.n_cols, 3)) + 1j * rng.standard_normal(
        (forward.n_cols, 3)
    )
    sample_dense = rng.standard_normal((forward.output_size, 3)) + 1j * rng.standard_normal(
        (forward.output_size, 3)
    )

    all_metrics: list[ViewMetrics] = []
    best_forward: tuple[ViewMetrics, PackedView] | None = None
    best_adjoint: tuple[ViewMetrics, PackedView] | None = None
    for view, dense in ((forward, grid_dense), (adjoint, sample_dense)):
        for policy in ROW_GROUPING_POLICIES:
            for k_tile in DEFAULT_K_TILES:
                packed = pack_view(
                    view,
                    policy,
                    k_tile,
                    seed + width * 100000 + grid_size * 1000 + k_tile,
                )
                metric = measure_view(case_id, view, packed, dense)
                all_metrics.append(metric)
                incumbent = best_forward if view.name == "G" else best_adjoint
                if incumbent is None or metric_rank(metric) > metric_rank(incumbent[0]):
                    if view.name == "G":
                        best_forward = (metric, packed)
                    else:
                        best_adjoint = (metric, packed)

    if best_forward is None or best_adjoint is None:
        raise AssertionError("missing best packed view")
    forward_metric, forward_packed = best_forward
    adjoint_metric, adjoint_packed = best_adjoint
    packed_forward_output = apply_packed(forward_packed, grid_dense)
    packed_adjoint_output = apply_packed(adjoint_packed, sample_dense)
    left = np.vdot(packed_forward_output, sample_dense)
    right = np.vdot(grid_dense, packed_adjoint_output)
    denominator = max(abs(left), abs(right), 1e-30)
    adjoint_defect = float(abs(left - right) / denominator)
    dual_storage = forward_metric.packed_storage_bytes + adjoint_metric.packed_storage_bytes
    csr_reference = reference_csr_bytes(forward)
    dual_ratio = dual_storage / csr_reference
    minimum_utilization = min(
        forward_metric.useful_slot_utilization,
        adjoint_metric.useful_slot_utilization,
    )
    maximum_residual = max(
        forward_metric.residual_fraction, adjoint_metric.residual_fraction
    )
    minimum_useful_efficiency = min(
        forward_metric.useful_dense_equiv_efficiency,
        adjoint_metric.useful_dense_equiv_efficiency,
    )
    correctness_pass = (
        forward_metric.checksum_match
        and adjoint_metric.checksum_match
        and forward_metric.apply_max_abs_error <= 1e-12
        and adjoint_metric.apply_max_abs_error <= 1e-12
        and adjoint_defect <= 1e-12
    )
    phase1_pass = (
        correctness_pass
        and minimum_utilization >= 0.70
        and maximum_residual <= 0.10
        and dual_ratio <= 2.0
        and minimum_useful_efficiency >= 1.50
    )
    summary: dict[str, object] = {
        "case_id": case_id,
        "trajectory": trajectory_kind,
        "grid_size": grid_size,
        "kernel_width": width,
        "samples": n_samples,
        "nnz": forward.nnz,
        "active_adjoint_rows": adjoint.n_active_rows,
        "best_forward": asdict(forward_metric),
        "best_adjoint": asdict(adjoint_metric),
        "adjoint_relative_defect": adjoint_defect,
        "dual_view_storage_bytes": dual_storage,
        "forward_csr_reference_bytes": csr_reference,
        "dual_view_to_forward_csr_ratio": dual_ratio,
        "minimum_useful_slot_utilization": minimum_utilization,
        "maximum_residual_fraction": maximum_residual,
        "minimum_useful_dense_equiv_efficiency": minimum_useful_efficiency,
        "correctness_pass": correctness_pass,
        "phase1_case_pass": phase1_pass,
    }
    return all_metrics, summary


def write_csv(path: Path, metrics: Sequence[ViewMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(metrics[0]).keys()) if metrics else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(asdict(metric))


def render_summary(cases: Sequence[dict[str, object]], overall: dict[str, object]) -> str:
    lines = [
        "# Phase-1 Layout Oracle Summary",
        "",
        "> Structural CPU evidence only. This is not a GPU performance result.",
        "",
        "| Case | Best G | Best G^T | Min utilization | Max residual | Dual/CSR | Adjoint defect | Gate |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        forward = case["best_forward"]
        adjoint = case["best_adjoint"]
        assert isinstance(forward, dict) and isinstance(adjoint, dict)
        lines.append(
            "| {case} | {fp}/{fk} | {ap}/{ak} | {util:.3f} | {resid:.3f} | "
            "{storage:.3f} | {defect:.3e} | {gate} |".format(
                case=case["case_id"],
                fp=forward["policy"],
                fk=forward["k_tile"],
                ap=adjoint["policy"],
                ak=adjoint["k_tile"],
                util=case["minimum_useful_slot_utilization"],
                resid=case["maximum_residual_fraction"],
                storage=case["dual_view_to_forward_csr_ratio"],
                defect=case["adjoint_relative_defect"],
                gate="PASS" if case["phase1_case_pass"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Phase-1 admission: **{overall['phase1_admission']}**",
            f"- Cases passed: {overall['case_pass_count']}/{overall['case_count']}",
            f"- Radial has an admitted width: {overall['radial_has_admitted_width']}",
            f"- Spiral has an admitted width: {overall['spiral_has_admitted_width']}",
            "- GPU speed remains `unknown`; no CUDA kernel or cuFINUFFT timing was run.",
            "",
            "The raw per-policy results are retained in `raw/view_results.csv` and "
            "`raw/case_summaries.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", default="64,96")
    parser.add_argument("--widths", default="4,6")
    parser.add_argument("--trajectories", default="radial,spiral")
    parser.add_argument("--samples-per-grid", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratchpad/layout_oracle/raw"),
    )
    args = parser.parse_args()

    grids = parse_int_list(args.grids)
    widths = parse_int_list(args.widths)
    trajectories = [item.strip() for item in args.trajectories.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_id": "layout_oracle_scratch",
        "evidence_class": "partial",
        "performance_claim": "none",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "grids": grids,
        "widths": widths,
        "trajectories": trajectories,
        "samples_per_grid": args.samples_per_grid,
        "seed": args.seed,
        "m_tile": M_TILE,
        "k_tiles": list(DEFAULT_K_TILES),
        "policies": list(ROW_GROUPING_POLICIES),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    all_metrics: list[ViewMetrics] = []
    case_summaries: list[dict[str, object]] = []
    for trajectory in trajectories:
        for grid_size in grids:
            n_samples = grid_size * args.samples_per_grid
            for width in widths:
                metrics, summary = run_case(
                    trajectory, grid_size, width, n_samples, args.seed
                )
                all_metrics.extend(metrics)
                case_summaries.append(summary)
                print(
                    summary["case_id"],
                    "PASS" if summary["phase1_case_pass"] else "FAIL",
                    f"util={summary['minimum_useful_slot_utilization']:.3f}",
                    f"residual={summary['maximum_residual_fraction']:.3f}",
                    f"dual/csr={summary['dual_view_to_forward_csr_ratio']:.3f}",
                    flush=True,
                )

    radial_admitted = any(
        case["trajectory"] == "radial" and case["phase1_case_pass"]
        for case in case_summaries
    )
    spiral_admitted = any(
        case["trajectory"] == "spiral" and case["phase1_case_pass"]
        for case in case_summaries
    )
    overall = {
        "case_count": len(case_summaries),
        "case_pass_count": sum(bool(case["phase1_case_pass"]) for case in case_summaries),
        "radial_has_admitted_width": radial_admitted,
        "spiral_has_admitted_width": spiral_admitted,
        "phase1_admission": "PASS" if radial_admitted and spiral_admitted else "FAIL",
    }
    write_csv(args.output_dir / "view_results.csv", all_metrics)
    (args.output_dir / "case_summaries.json").write_text(
        json.dumps(case_summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "overall.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path = args.output_dir.parent / "SUMMARY.md"
    summary_path.write_text(render_summary(case_summaries, overall), encoding="utf-8")
    return 0 if overall["phase1_admission"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
