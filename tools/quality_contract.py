"""Validate complete quality matrices from row-level records."""

import itertools
import math


TRAJECTORIES = ("spiral", "radial", "golden")


def validate_rows(rows: list[dict], acquisitions: tuple[str, ...] = ()) -> None:
    if acquisitions:
        expected = set(itertools.product(acquisitions, TRAJECTORIES, range(3)))
        keys = [(r["acquisition"], r["trajectory"], r["frame"]) for r in rows]
    else:
        expected = set(itertools.product(TRAJECTORIES, range(3)))
        keys = [(r["trajectory"], r["frame"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise RuntimeError("quality matrix has missing, duplicate, or unexpected rows")
    for row in rows:
        if row["cg_iterations"] != 10 or row["operator_scale"] != 512:
            raise RuntimeError("quality row does not use ten steps and raw scale 512")
        if row["native_operator_scale"] != 1:
            raise RuntimeError("quality row does not use native scale 1")
        if not all(row[key] is True for key in (
            "finite", "application_pass", "application_vs_fp32_pass",
            "eager_graph_reconstruction_bitwise", "eager_graph_residuals_bitwise",
        )):
            raise RuntimeError("row-level quality or eager/Graph determinism failed")
        metrics = [float(row[key]) for key in (
            "native_vs_fp32_magnitude_nrmse", "native_vs_fp32_ssim",
            "native_vs_fp32_psnr_db", "native_truth_nrmse", "fp32_truth_nrmse",
        )]
        if not all(math.isfinite(value) for value in metrics):
            raise RuntimeError("nonfinite reconstruction metric")
        _, ssim, psnr, native_truth, reference_truth = metrics
        if ssim < 0.9999 or psnr < 60 or native_truth - reference_truth > 0.002:
            raise RuntimeError("row metrics fail the frozen application-quality gate")
