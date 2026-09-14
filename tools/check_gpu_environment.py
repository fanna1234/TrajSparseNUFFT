#!/usr/bin/env python3
"""Check GPU validation dependencies before downloading or preparing inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


PACKAGES = {
    "numpy": "2.2.6",
    "torch": "2.8.0+cu129",
    "torchkbnufft": "1.5.2",
    "h5py": "3.16.0",
    "ismrmrd": "1.15.0",
    "scikit-image": "0.26.0",
}


def inspect() -> dict:
    observed = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": {},
    }
    errors = []
    for package in PACKAGES:
        try:
            observed["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing dependency: {package}")
    if errors:
        raise RuntimeError("; ".join(errors) + "; see docs/REPRODUCE.md")

    import h5py  # noqa: F401 -- also check binary-extension imports
    import ismrmrd  # noqa: F401
    import skimage  # noqa: F401
    import torch
    import torchkbnufft  # noqa: F401

    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("GPU reproduction requires Linux x86-64")
    if not torch.cuda.is_available():
        raise RuntimeError("the selected Python environment cannot access CUDA")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 0):
        raise RuntimeError(f"this implementation requires SM120a, observed {capability}")
    observed.update(
        torch_cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        compute_capability=list(capability),
    )
    matched = (
        sys.version_info[:2] == (3, 13)
        and observed["packages"] == PACKAGES
        and torch.version.cuda == "12.9"
    )
    allow_drift = os.environ.get("TRAJTC_ALLOW_VERSION_DRIFT", os.environ.get(
        "TRAJSPARSE_ALLOW_VERSION_DRIFT", "0"
    )) == "1"
    observed["contract"] = "matched-python" if matched else "portability"
    if not matched and not allow_drift:
        raise RuntimeError(
            "Python dependencies differ from the measured environment: "
            + json.dumps(observed, sort_keys=True)
            + "; use requirements-gpu.txt or set TRAJTC_ALLOW_VERSION_DRIFT=1 "
            "for an explicitly labeled portability run"
        )
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    observed = inspect()
    text = json.dumps(observed, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
