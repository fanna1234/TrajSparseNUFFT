# Reproduction Guide

## Environment

The measured configuration is Linux x86-64, NVIDIA SM120a, CUDA 13.2, and
driver 610.43.02. Production and dense-control builds accept CUDA 12.8--13.3
but results outside CUDA 13.2 are portability checks, not matched
reproductions. `uv.lock` freezes the dependency-light CPU oracle. The measured
GPU validation environment used Python 3.13.9, NumPy 2.2.6, PyTorch
2.8.0+cu129, and TorchKbNufft 1.5.2; data preparation additionally requires
`h5py`, `ismrmrd`, and `scikit-image`. Install the PyTorch build appropriate
for the reviewer's CUDA driver rather than using the CPU oracle environment.
`prepare-data` rejects a different core Python environment by default. Set
`TRAJSPARSE_ALLOW_VERSION_DRIFT=1` only for a result explicitly treated as a
portability run rather than a matched reproduction.

```bash
./reproduce.sh smoke
./reproduce.sh evidence
```

The CPU smoke test takes under one second after dependency synchronization on
the development Mac.
Cold GPU build and full-run wall times have not been remeasured from a fresh
clone; the drivers retain timestamps and complete raw logs.

## Data preparation

The repository does not contain raw k-space, derived coil arrays, sensitivity
maps, packed matrices, or runtime measurement buffers. The cold-start path is:

```bash
./reproduce.sh get-data
export TRAJSPARSE_GPU_PYTHON=/path/to/gpu-environment/bin/python
./reproduce.sh prepare-data
source reproduced-data/layout.env
```

`get-data` downloads the three exact public S3 objects and validates size plus
SHA-256. `prepare-data` then performs phase selection, virtual-coil
compression, trajectory generation, TorchKbNufft measurement export with the
common raw scale of 512, paired `G/G^H` packing, and dense-control expansion.
It refuses to reuse a nonempty `TRAJSPARSE_PREPARED_ROOT`.

The fixed spiral fixture is versioned because it defines the evaluated sample
locations. Its SHA-256 values are:

```text
fe1955cbcd0e343b7724357a833ea83b6d47e3ffbc749ac3208393a879483eba  spiral_standard_256.npy
faa3f82b6f55ea26bb562ea2ebd501047db0bb184ac55ca8db37db11070a97e  spiral_standard_256.f32xy.bin
```

The resulting layout is:

```text
${DEVELOPMENT_CASE}/
${HELDOUT_CASES_ROOT}/fs0005_v2/
${HELDOUT_CASES_ROOT}/fs0016_v2/
${DEVELOPMENT_RUNTIME_ROOT}/frame{00,01,02}/{spiral,radial,golden}65/
${HELDOUT_RUNTIME_ROOT}/{fs0005,fs0016}/frame{00,01,02}/{spiral,radial,golden}65/
${PACKED_ROOT}/{spiral,radial,golden}65/{G,G_T}/{real,imag}/
${DENSE_ROOT}/{spiral,radial,golden}65/{G,G_T}/
```

Every preparation stage emits hashes. `layout.env` exports the seven paths
consumed by all subsequent groups.

## Build groups

Production and matched dense control:

```bash
./reproduce.sh build
```

Native cuFINUFFT baseline:

```bash
./reproduce.sh build-baseline
```

The group clones the URL and checks out the exact revision recorded in
`experiments/baselines/baselines.lock.json`, configures a static CUDA-only
build, and then links the native full-CG harness. Existing dirty third-party
checkouts are rejected. CMake 3.25 or newer is required; the build disables
CMake's architecture inference and passes the exact `compute_120a/sm_120a`
code-generation flag to NVCC. `FINUFFT_ROOT` and `CUFINUFFT_BUILD` remain
optional location overrides.
Build scripts stop on an unsupported CUDA version, missing dependency, failed
compiler invocation, or unexpected runtime cuFFT dependency in production.

## Quality group

```bash
source reproduced-data/layout.env
./reproduce.sh quality --dry-run
./reproduce.sh quality
```

The group executes nine development rows and eighteen held-out rows with one
truth-free raw scale of 512. It fails unless all 27 rows pass application
quality and eager/Graph determinism.

## Main-performance group

```bash
source reproduced-data/layout.env
./reproduce.sh baseline-quality
./reproduce.sh main-performance --dry-run
./reproduce.sh main-performance
```

`baseline-quality` first admits every output of the exact cuFINUFFT
configuration. Cross-run nondeterminism is retained as a separate diagnostic
and does not override per-output application quality. The timing group then
uses method 1 for the forward call, method 2 for the adjoint call,
sorting disabled, six alternating process pairs, five warmups, twenty timed
reconstructions, and the common raw scale of 512. A result below 97% of the
2.514x anchor is labeled `LOW` and returns a failure status. Set
`TRAJSPARSE_CUFINUFFT_QUALITY_SUMMARY` only when intentionally supplying an
equivalent fresh admission record.

## Design-evidence group

```bash
source reproduced-data/layout.env
./reproduce.sh design-evidence --dry-run
./reproduce.sh design-evidence
```

The group first runs a fresh nine-row dense-control quality and determinism
gate and binds its binary SHA and dense-pack root to the timing run. Sparse and
dense binaries share FFT, SENSE, CG, maps, reductions, and timing scope. Only
the interpolation implementation differs. A result below 97% of the
corresponding eager or Graph anchor returns a failure status.

## Failure policy

All measurement groups are fail-fast and refuse nonempty output directories.
Missing inputs, partial process pairs, failed quality gates, foreign timing
interference, or low performance remain visible in logs and produce a nonzero
exit code. Fresh outputs are written under `reproduced-results/` by default;
set `TRAJSPARSE_RESULTS_ROOT` to redirect them.
