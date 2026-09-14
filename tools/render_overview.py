#!/usr/bin/env python3
"""Render the README overview from the fixed trajectory and schematic dataflow."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np


def render(repo: Path) -> None:
    fixture = repo / "experiments/data_prep/trajectories/spiral_standard_256.npy"
    trajectory = np.load(fixture)
    coordinates = trajectory.reshape(-1, 2)
    # Show one complete interleaf; joining interleaves would draw spurious edges.
    points = trajectory[0]
    span = max(float(np.ptp(coordinates, axis=0).max()), 1e-12)
    center = (coordinates.max(axis=0) + coordinates.min(axis=0)) / 2
    xy = (points - center) * (94 / span)
    path = " ".join(f"{x + 88:.2f},{106 - y:.2f}" for x, y in xy)
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="340" viewBox="0 0 1160 340" role="img" aria-labelledby="title desc">',
        '<title id="title">TrajTC: reusable geometry and per-call precision</title>',
        '<desc id="desc">The fixed spiral trajectory is packed offline into independent forward and adjoint 2:4 coefficient layouts. Each dynamic FP32 input is split into FP16 high and residual panels, evaluated with the same sparse pack, and accumulated in FP32.</desc>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#202124"/></marker></defs>',
        '<rect width="1160" height="340" rx="10" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#111111">',
    ]

    def text(x, y, content, size=19, weight="400", anchor="middle"):
        pieces.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(content)}</text>')

    def rect(x, y, width, height, fill="#f3f5f7"):
        pieces.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="{fill}"/>')

    def arrow(x1, y1, x2, y2):
        pieces.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="#202124" stroke-width="1.6" marker-end="url(#arrow)"/>')

    text(30, 29, "OFFLINE · FIXED GEOMETRY", 14, "600", "start")
    pieces.append(f'<polyline points="{path}" fill="none" stroke="#6e94b3" stroke-width="1.1"/>')
    text(192, 97, "Fixed", 20)
    text(192, 122, "trajectory", 20)
    text(88, 162, "One interleaf", 13)
    arrow(257, 106, 308, 106)
    rect(331, 60, 270, 90)
    text(466, 96, "Pack coefficients", 21, "600")
    text(466, 124, "Reorder + legal layers", 17)
    arrow(622, 106, 672, 106)
    rect(695, 60, 422, 90, "#e8f0f7")
    text(906, 96, "Reusable 2:4 coefficient packs", 21, "600")
    text(906, 124, "Forward / adjoint coefficients and maps", 17)
    pieces.append('<path d="M30 173 H1130" stroke="#b7bec5" stroke-dasharray="5 5" fill="none"/>')
    text(30, 201, "PER CALL · CHANGING INPUTS", 14, "600", "start")
    # The reuse arrow feeds only the sparse interpolation operation.
    arrow(755, 151, 755, 223)
    rect(30, 228, 213, 75)
    text(136.5, 260, "FP32 input panel", 21, "600")
    text(136.5, 286, "FFT or sample endpoint", 16)
    arrow(253, 266, 294, 266)
    rect(308, 228, 106, 75, "#dceaf5")
    rect(424, 228, 120, 75, "#edf2f6")
    text(361, 259, "High", 21, "600")
    text(361, 286, "FP16", 17)
    text(484, 259, "Residual", 21, "600")
    text(484, 286, "FP16", 17)
    arrow(557, 266, 600, 266)
    rect(614, 228, 276, 75, "#e8f0f7")
    text(752, 260, "Sparse complex products", 20, "600")
    text(752, 286, "High + residual passes", 17)
    arrow(904, 266, 944, 266)
    rect(958, 228, 159, 75)
    text(1037.5, 260, "FP32 output", 20, "600")
    text(1037.5, 286, "Solver continues", 16)
    text(30, 330, "Interpolation dataflow; FFT, SENSE and CG remain GPU-resident.", 15, anchor="start")
    pieces.extend(["</g>", "</svg>"])
    destination = repo / "assets"
    destination.mkdir(exist_ok=True)
    (destination / "overview.svg").write_text("\n".join(pieces) + "\n", encoding="utf-8")
    manifest = {
        "kind": "schematic interpolation dataflow with a real trajectory fixture",
        "trajectory": str(fixture.relative_to(repo)),
        "trajectory_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "curve_sampling": "first complete interleaf, all 256 stored samples",
        "scope": "offline coefficient packs versus per-call input precision; no timing or matrix occupancy is encoded",
    }
    (destination / "overview.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    render(parser.parse_args().repo_root.resolve())
