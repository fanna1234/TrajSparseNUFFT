#!/usr/bin/env python3
"""Prepare three 256x256 eight-coil OCMR frames for NUFFT evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import ismrmrd
import ismrmrd.xsd
import numpy as np
from skimage.io import imsave


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def center_copy_kspace(source: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Center crop/pad [frame, coil, ky, kx] k-space to a square grid."""
    frames, coils, src_y, src_x = source.shape
    target = np.zeros((frames, coils, target_size, target_size), np.complex64)
    copy_y = min(src_y, target_size)
    copy_x = min(src_x, target_size)
    src_y0 = (src_y - copy_y) // 2
    src_x0 = (src_x - copy_x) // 2
    dst_y0 = (target_size - copy_y) // 2
    dst_x0 = (target_size - copy_x) // 2
    target[:, :, dst_y0 : dst_y0 + copy_y, dst_x0 : dst_x0 + copy_x] = source[
        :, :, src_y0 : src_y0 + copy_y, src_x0 : src_x0 + copy_x
    ]
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", default="0,6,13")
    args = parser.parse_args()

    selected = [int(value) for value in args.frames.split(",")]
    output_root = args.output.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output_root}")
    source_digest = sha256(args.input)
    dataset = ismrmrd.Dataset(str(args.input), "dataset", create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    encoding = header.encoding[0]
    encoded_x = int(encoding.encodedSpace.matrixSize.x)
    encoded_y = int(encoding.encodingLimits.kspace_encoding_step_1.maximum + 1)
    phases = int(encoding.encodingLimits.phase.maximum + 1)
    coils = int(header.acquisitionSystemInformation.receiverChannels)
    if any(frame < 0 or frame >= phases for frame in selected):
        raise ValueError(f"selected frames {selected} outside [0,{phases})")

    kspace = np.zeros((len(selected), coils, encoded_y, encoded_x), np.complex64)
    line_counts = np.zeros((len(selected), encoded_y), dtype=np.int32)
    frame_lookup = {frame: index for index, frame in enumerate(selected)}
    imaging_acquisitions = 0
    for acquisition_index in range(dataset.number_of_acquisitions()):
        acquisition = dataset.read_acquisition(acquisition_index)
        if acquisition.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue
        phase = int(acquisition.idx.phase)
        if phase not in frame_lookup:
            continue
        ky = int(acquisition.idx.kspace_encode_step_1)
        samples = int(acquisition.number_of_samples)
        source = np.asarray(acquisition.data, np.complex64)
        if source.shape != (coils, samples):
            raise RuntimeError(f"unexpected acquisition shape {source.shape}")
        x0 = max(0, encoded_x - samples) if acquisition.center_sample * 2 < encoded_x else 0
        frame_index = frame_lookup[phase]
        kspace[frame_index, :, ky, x0 : x0 + samples] += source
        line_counts[frame_index, ky] += 1
        imaging_acquisitions += 1

    if np.any(line_counts == 0):
        missing = np.argwhere(line_counts == 0)
        raise RuntimeError(
            f"missing {len(missing)} selected phase/ky lines; first entries "
            f"{missing[:8].tolist()}"
        )
    kspace /= line_counts[:, None, :, None]

    kspace_256 = center_copy_kspace(kspace)
    coil_images = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace_256, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    ).astype(np.complex64)

    # One compression matrix is shared across all selected real frames.
    flattened = coil_images.transpose(1, 0, 2, 3).reshape(coils, -1)
    covariance = flattened @ flattened.conj().T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    compression = eigenvectors[:, order[:8]].astype(np.complex64)
    compressed = np.einsum(
        "ca,fcyx->fayx", compression.conj(), coil_images, optimize=True
    ).astype(np.complex64)
    rss = np.sqrt(np.sum(np.abs(compressed) ** 2, axis=1)).astype(np.float32)
    scales = np.maximum(rss.reshape(len(selected), -1).max(axis=1), 1e-12)
    compressed /= scales[:, None, None, None]
    rss /= scales[:, None, None]
    sensitivity = compressed / np.maximum(rss[:, None], 1e-6)
    captured_energy = float(
        np.maximum(eigenvalues[order[:8]], 0).sum()
        / np.maximum(np.maximum(eigenvalues, 0).sum(), 1e-30)
    )

    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / "coil_images.npy", compressed)
    np.save(output_root / "sensitivity_maps.npy", sensitivity.astype(np.complex64))
    np.save(output_root / "reference_rss.npy", rss)
    np.save(output_root / "compression_matrix.npy", compression)
    for index, frame in enumerate(selected):
        image = np.clip(rss[index] / np.maximum(rss[index].max(), 1e-12), 0, 1)
        imsave(output_root / f"reference_frame{frame:02d}.png", (image * 65535).astype(np.uint16))

    manifest = {
        "experiment_id": "data_prep",
        "source": "OCMR fully sampled cardiac cine",
        "source_file": str(args.input),
        "source_sha256": source_digest,
        "source_license": "CC BY-NC 4.0",
        "selected_phases": selected,
        "source_encoded_shape": [encoded_y, encoded_x],
        "source_coils": coils,
        "output_shape": [256, 256],
        "output_virtual_coils": 8,
        "coil_compression_energy_fraction": captured_energy,
        "normalization_scales": scales.tolist(),
        "selected_imaging_acquisitions": imaging_acquisitions,
        "line_repeats_min": int(line_counts.min()),
        "line_repeats_max": int(line_counts.max()),
        "retrospective_noncartesian": True,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
