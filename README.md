# TrajSparseNUFFT

TrajSparseNUFFT is a pruning-free non-Cartesian MRI reconstruction path for
NVIDIA 2:4 Sparse Tensor Cores. It represents every interpolation panel as an
FP16 high component plus an FP16 residual component, executes both with native
sparse MMA, and accumulates in FP32. The coefficient set and the original
ten-step CG recurrence are preserved.

## Implementation

The repository contains the complete measured path:

- trajectory-aware packing for the forward and adjoint interpolation views;
- native complex `HMMA.SP` interpolation with FP32 accumulation;
- self-developed forward and inverse FFT endpoints without a runtime cuFFT
  dependency in the production binary;
- SENSE, deterministic reductions, CUDA Graph execution, and ten-step CG;
- an equal-quality dense Tensor Core control and a native cuFINUFFT full-CG
  baseline;
- CPU structural oracles, real-data preparation, validation, and paired-run
  drivers.

The production configuration is fixed to 256x256 images, a 512x512 grid,
kernel width 6, eight complex coils, and SM120a. CUDA 13.2 is the measured
environment; the build scripts accept CUDA 12.8 through 13.3.

## Quick verification

The two commands below require no GPU or licensed MRI data:

```bash
./reproduce.sh smoke
./reproduce.sh evidence
```

`smoke` runs four unit tests over the lossless dual-view packing oracle.
`evidence` validates the schema, row counts, process-pair counts, and frozen
result anchors in the committed evidence record.

On an SM120a system:

```bash
./reproduce.sh build
```

The build fails if the CUDA version or architecture is outside the supported
contract, or if either production binary links against cuFFT.

## Reproduction groups

| Group | Purpose | Data requirement | Output |
|---|---|---|---|
| `smoke` | CPU packing correctness and adjointness | none | console verdicts |
| `evidence` | Recompute committed results from hashed source summaries | none | console verdicts |
| `prepare-trajectories` | Generate radial and golden-angle fixtures | none | `experiments/.../trajectories/` |
| `get-data` | Download and SHA-256-check the three OCMR acquisitions | network access | `data/ocmr/` |
| `prepare-data` | Build cases, runtime inputs, sparse packs, and dense controls | OCMR data, GPU Python environment | `reproduced-data/` |
| `build` | Build production and matched dense-control binaries | SM120a, CUDA | experiment-local `build/` directories |
| `build-baseline` | Checkout the locked FINUFFT revision and build native cuFINUFFT full CG | SM120a, CUDA, network access | baseline `build/` directory |
| `baseline-quality` | Admit the exact native cuFINUFFT configuration before timing | development OCMR case and runtime inputs | `reproduced-results/baseline-quality/` |
| `quality` | Re-run the 27-row truth-free reconstruction gate | prepared OCMR cases, packs, runtime inputs | `reproduced-results/quality/` |
| `main-performance` | Six paired full-CG comparisons with native cuFINUFFT | prepared 65k packs and runtime inputs | `reproduced-results/main-performance/` |
| `design-evidence` | Six paired sparse-versus-dense full-CG comparisons | sparse and dense packs | `reproduced-results/design-evidence/` |
| `hardening` | memcheck plus fixed-input stress for all three paths | prepared inputs, SM120a | `reproduced-results/hardening/` |
| `long-cg` | Re-run the 10/20/30/50-step stability endpoints | prepared held-out input, SM120a | `reproduced-results/long-cg/` |

Every group supports `--dry-run`. Measurement drivers refuse to overwrite
nonempty output directories, propagate failed subprocesses, and retain low or
failed results. Data download validates existing files before reuse, while
`prepare-data` refuses a nonempty prepared-data root.
See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the required data layout and
exact environment variables.

## Reference results

The frozen SM120a evidence closes three independent gates:

| Gate | Result |
|---|---|
| Reconstruction quality | 27/27 rows pass across three OCMR acquisitions; worst magnitude NRMSE is 0.1857% and minimum SSIM is 0.9999628 |
| Native external baseline | complete CG is 2.514x faster than quality-admitted native cuFINUFFT; 95% CI [2.484, 2.542], 6/6 process-pair wins |
| Native 2:4 attribution | complete CG is 1.408x faster than the equal-quality dense control in eager mode and 1.414x in Graph mode; 6/6 wins in each mode |

The machine-readable record is `evidence/frozen_results.json`; its input hashes
bind it to the complete source summaries under `evidence/source/`.
Fresh runs are classified against these anchors as `OK >= anchor`,
`WITHIN 3%`, or `LOW`; a low result returns a failing exit code and is not
discarded.

## Measurement contract

- Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, SM120a.
- Workload: three 65k retrospective trajectory families and three cardiac
  phases per acquisition.
- Quality: one truth-free raw scale of 512 is shared by every row; FP32 sparse
  reconstruction is the numerical reference.
- Timing: complete device-resident reconstruction, excluding data transfer,
  setup, trajectory packing, and baseline planning.
- Estimator: six fresh process pairs with alternating execution order; eager
  and Graph modes are reported separately.
- Admission: external and dense denominators must first pass their own
  reconstruction-quality gates.

cuFINUFFT is not bitwise deterministic under this configuration. All 36
frozen outputs pass application quality, while 16 of 18 repeat pairs meet the
separate 0.2% repeat diagnostic; the repository preserves both facts.

## Scope boundary

These results establish application-level agreement, not numerical or clinical
equivalence. They cover three retrospective OCMR acquisitions on one GPU
generation and one shape. Prospective non-Cartesian scans, other image sizes,
other coil counts, and other architectures remain unmeasured. Packing takes
seconds and must be shipped with or amortized across repeated fixed-trajectory
reconstructions.

## Repository map

- `src/`, `tests/`: CPU structural oracle, exporters, and unit tests.
- `experiments/production/`: production
  CUDA implementation and complete-CG drivers.
- `experiments/dense_control/`:
  matched dense Tensor Core control.
- `experiments/cufinufft_baseline/`:
  native cuFINUFFT baseline.
- `experiments/quality/` and
  `experiments/frozen_evidence/`:
  multi-acquisition and long-CG reproduction drivers.
- `experiments/data_prep/`: OCMR preparation and
  trajectory fixtures.
- `evidence/`: sanitized source summaries, hashes, and the derived result
  record.
- `tools/`, `reproduce.sh`: fail-fast environment and result gates.

## Data and licensing

OCMR data is not redistributed. Download it from the upstream registry under
CC BY-NC 4.0 and verify the hashes in [docs/DATASET.md](docs/DATASET.md).
Project-authored code is available under the included MIT license; third-party
baselines retain their upstream licenses.
