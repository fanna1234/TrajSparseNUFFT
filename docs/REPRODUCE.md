# Reproduction

Source tests, frozen-result checks, and new GPU measurements are separate
activities. The first two need no MRI data; the third requires the measured
hardware configuration and prepared inputs.

## Environment

### CPU checks

Install Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The CPU dependencies are recorded in `uv.lock`; no PyTorch installation is
needed for these checks.

```bash
./reproduce.sh smoke
./reproduce.sh evidence
```

`smoke` exercises the structural oracle and failure paths. `evidence` checks
the saved JSON sources and derived record, not fresh GPU execution.
`./reproduce.sh help` lists the runnable groups.

### GPU checks and reconstruction

The measured configuration is Linux x86-64, RTX PRO 6000 Blackwell Workstation
Edition (SM120a), CUDA toolkit 13.2, and driver 610.43.02. Use Python 3.13 and
the direct dependency versions in [requirements-gpu.txt](../requirements-gpu.txt).
Create a separate environment so the dependency-light CPU checks stay separate:

```bash
uv venv --python 3.13 .venv-gpu
uv pip install --python .venv-gpu/bin/python \
  torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv-gpu/bin/python -r requirements-gpu.txt
export TRAJTC_GPU_PYTHON="$PWD/.venv-gpu/bin/python"
export NVCC=/usr/local/cuda-13.2/bin/nvcc
./reproduce.sh doctor
```

The PyTorch CUDA 12.9 wheel follows the
[official PyTorch 2.8 installation instructions](https://pytorch.org/get-started/previous-versions/).
Its bundled runtime is distinct from the CUDA 13.2 toolkit used to compile the
custom kernels. `requirements-gpu.txt` pins direct dependencies, not every
transitive package; save the installed package list with fresh results.

`doctor` checks the compiler, SM120a target, Python packages, CUDA access, and
visible device capability. The selected Python is reused by preparation,
quality validation, and paired-run drivers. Builds accept CUDA 12.8--13.3;
versions other than 13.2 are portability checks, not matched timings. Set
`TRAJTC_ALLOW_VERSION_DRIFT=1` only for a labeled portability run. It does not
enable a different GPU architecture.

Native builds require a C++ toolchain and standard Linux binary tools.
The cuFINUFFT build additionally requires Git, CMake 3.25+, and network access.
Sanitizer checks require Compute Sanitizer. Timing requires exclusive GPU
access; the drivers stop if they observe another compute process.

## Prepare the data

```bash
./reproduce.sh get-data
./reproduce.sh prepare-data
source reproduced-data/layout.env
```

The [dataset manifest](DATASET.md) lists three public OCMR objects, approximately
710 MB in total, with exact sizes and SHA-256 checksums. Existing downloads
are validated before reuse. Corrupt files are not silently replaced.

Preparation selects phases 0, 6, and 13, compresses each acquisition to eight
virtual coils, generates retrospective measurements, and builds forward,
adjoint, and dense-control packs. Operator normalization uses raw scale 512
(native scale 1) for every row; exported measurements remain unscaled.
Multiplying measurements by 512 is not equivalent and causes FP16 overflow.
All derived arrays stay outside Git. A nonempty preparation directory is never
overwritten; choose a new `TRAJTC_PREPARED_ROOT` for another attempt.

The generated `layout.env` records the interpreter and paths used by later
groups; `environment.json` records the observed Python and device environment.
The main layout is:

```text
reproduced-data/
  cases/{fs0152,fs0005_v2,fs0016_v2}/
  runtime/development/frame{00,01,02}/{spiral,radial,golden}65/
  runtime/heldout/{fs0005,fs0016}/frame{00,01,02}/{spiral,radial,golden}65/
  packed/{spiral,radial,golden}65/{G,G_T}/{real,imag}/
  dense/{spiral,radial,golden}65/{g,gt}/
  environment.json
  layout.env
```

The planner also retains the 32k radial/golden support cases; headline timing
uses only the three 65k cases. The fixed spiral fixture is shipped with the
source; its checksum and raw-data provenance are in [Data](DATASET.md).

## Reproduce quality and the external comparison

```bash
source reproduced-data/layout.env
./reproduce.sh quality --dry-run
./reproduce.sh quality
./reproduce.sh baseline-quality
./reproduce.sh main-performance
```

`quality` builds the production binary and evaluates nine development and
eighteen held-out rows. The checker requires the exact acquisition/trajectory/
phase matrix, finite application-quality metrics, and eager/Graph equality.
The numerical reference and native solver use the same fixed scale. A summary
flag alone cannot conceal a missing, duplicated, or failed row.

`baseline-quality` fetches the locked FINUFFT revision, builds the native
full-CG baseline, and checks two outputs for each of nine development rows.
Cross-run nondeterminism is retained separately from per-output admission.
The admission record binds the binary hash, scale, methods, and sort setting
to the timing command.

`main-performance` measures complete ten-step device-resident CG with six
fresh process pairs, alternating order, five warmups, and twenty timed solves.
The admitted baseline uses forward method 1, adjoint method 2, sorting off,
and raw scale 512. Packing, transfers, setup, and baseline planning are excluded.
This is a warm reconstruction comparison, not cold-start end-to-end latency.

Each run retains process output, paired latency records, GPU state snapshots,
and the derived summary. The checker recomputes speedup from six distinct
process pairs; it rejects invalid latencies and incomplete records.

## Attribute sparse execution

```bash
./reproduce.sh design-evidence
```

The group builds the matched dense control, runs its nine-row quality and
determinism checks, then compares sparse and dense complete CG in eager and
Graph modes separately. Both paths share the FFT, SENSE, CG, input precision,
maps, and reductions. Their interpolation implementations differ.

## Command map

| Group | Prerequisite | Result |
|---|---|---|
| `smoke` | uv, Python | CPU correctness and failure-path tests |
| `evidence` | Python | Frozen-source integrity and result checks |
| `doctor` | GPU Python, CUDA | Environment and device readiness |
| `prepare-trajectories` | uv, Python | Deterministic radial/golden fixtures |
| `get-data` | Network, curl | Hash-verified OCMR downloads |
| `prepare-data` | GPU Python, OCMR | Cases, runtime inputs, packs, `layout.env` |
| `build` | CUDA | Production and dense-control binaries |
| `build-baseline` | CUDA, Git, CMake | Pinned native cuFINUFFT binary |
| `quality` | Prepared data | Complete 27-row quality matrix |
| `baseline-quality` | Prepared data | cuFINUFFT admission record |
| `main-performance` | Fresh baseline admission | External paired complete-CG result |
| `design-evidence` | Prepared sparse/dense packs | Matched sparse/dense paired result |
| `hardening` | Prepared data, sanitizer | memcheck and fixed-input stress |
| `long-cg` | Held-out data | Fixed 10/20/30/50-step endpoints |

Every group accepts `--dry-run`. It prints commands without building,
downloading, allocating GPU work, or changing result directories. Optional
checks do not change the headline measurement contract.

## Outputs and troubleshooting

Fresh results default to `reproduced-results/`; use a new
`TRAJTC_RESULTS_ROOT` to preserve a failed or completed campaign before a rerun.
Performance is classified as `OK >= anchor`, `WITHIN 3%`, or `LOW` against the
matching frozen result. `LOW` returns a nonzero exit code and keeps all output.
It means the timing anchor was not reproduced, not that the logs should be
deleted or a different baseline substituted.

| Symptom | Action |
|---|---|
| Missing Python packages | Set `TRAJTC_GPU_PYTHON` to the GPU environment and run `doctor` |
| Wrong CUDA/architecture | Set `NVCC`; retain SM120a rather than relaxing the hardware check |
| Nonempty output directory | Choose a new prepared/results root; keep the prior attempt |
| Missing baseline admission | Run `baseline-quality` before `main-performance` |
| Foreign GPU process | Wait for exclusive access; do not terminate unrelated work |
| Lower speedup | Inspect clocks, environment, timing scope, and retained paired logs |

Path overrides use the `TRAJTC_` prefix: `DATA_ROOT`, `PREPARED_ROOT`,
`RESULTS_ROOT`, `GPU_PYTHON`, and the seven paths generated in `layout.env`.
Existing `TRAJSPARSE_` settings and old layout files remain accepted; conflicting
old/new values fail explicitly. `FINUFFT_ROOT`, `CUFINUFFT_BUILD`, `BUILD_JOBS`,
`NVCC`, and `COMPUTE_SANITIZER` remain supported tool overrides.

See [Validation status](VALIDATION.md) for what was actually exercised from
the delivered source. Frozen results do not imply that every fresh-run group
was rerun during each documentation update.
