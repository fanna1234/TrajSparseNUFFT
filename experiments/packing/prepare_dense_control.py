#!/usr/bin/env python3
"""Expand Split24 compressed A layers into exact dense M16xK32 tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


M = 16
K = 32
K_COMP = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_row_metadata(pair_meta: np.ndarray) -> np.ndarray:
    if pair_meta.shape[1] != M:
        raise ValueError(f"unexpected pair metadata shape {pair_meta.shape}")
    result = np.zeros((pair_meta.shape[0], M), dtype=np.uint32)
    for group in range(8):
        first = pair_meta[:, group * 2]
        second = pair_meta[:, group * 2 + 1]
        result[:, group] = (first & 0xFFFF) | ((second & 0xFFFF) << 16)
        result[:, group + 8] = (first >> 16) | (second & 0xFFFF0000)
    return result


def expand_component(directory: Path, output: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    tiles = int(manifest["tiles"])
    compressed = np.fromfile(directory / "tile_a_comp.bin", dtype=np.float16).reshape(
        tiles, M, K_COMP
    )
    pair_meta = np.fromfile(
        directory / "tile_pair_meta.bin", dtype=np.uint32
    ).reshape(tiles, M)
    row_meta = decode_row_metadata(pair_meta)
    dense = np.zeros((tiles, M, K), dtype=np.float16)
    for chunk in range(K // 4):
        code = (row_meta >> np.uint32(chunk * 4)) & np.uint32(0xF)
        p0 = code & np.uint32(0x3)
        p1 = (code >> np.uint32(2)) & np.uint32(0x3)
        rows = np.arange(M, dtype=np.int64)[None, :]
        tile_ids = np.arange(tiles, dtype=np.int64)[:, None]
        dense[tile_ids, rows, chunk * 4 + p0] = compressed[:, :, chunk * 2]
        dense[tile_ids, rows, chunk * 4 + p1] = compressed[:, :, chunk * 2 + 1]

    recovered = np.empty_like(compressed)
    for chunk in range(K // 4):
        code = (row_meta >> np.uint32(chunk * 4)) & np.uint32(0xF)
        p0 = code & np.uint32(0x3)
        p1 = (code >> np.uint32(2)) & np.uint32(0x3)
        rows = np.arange(M, dtype=np.int64)[None, :]
        tile_ids = np.arange(tiles, dtype=np.int64)[:, None]
        recovered[:, :, chunk * 2] = dense[tile_ids, rows, chunk * 4 + p0]
        recovered[:, :, chunk * 2 + 1] = dense[tile_ids, rows, chunk * 4 + p1]
    if not np.array_equal(recovered.view(np.uint16), compressed.view(np.uint16)):
        raise RuntimeError("dense expansion does not reconstruct compressed values")
    nonzeros = np.count_nonzero(dense, axis=2)
    if np.max(nonzeros) > 16:
        raise RuntimeError("expanded 2:4 layer exceeds 16 nonzeros per K32 row")

    output.parent.mkdir(parents=True, exist_ok=True)
    dense.tofile(output)
    return {
        "source": str(directory),
        "output": str(output),
        "shape": list(dense.shape),
        "dtype": str(dense.dtype),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "source_compressed_sha256": sha256(directory / "tile_a_comp.bin"),
        "source_metadata_sha256": sha256(directory / "tile_pair_meta.bin"),
        "bitwise_roundtrip": True,
        "max_nonzeros_per_row": int(np.max(nonzeros)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--imag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "real": expand_component(args.real.resolve(), args.output / "real.f16.bin"),
        "imag": expand_component(args.imag.resolve(), args.output / "imag.f16.bin"),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
