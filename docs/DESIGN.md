# Design

TrajTC separates two properties with different lifetimes: interpolation
coefficients depend on a fixed sampling trajectory, while each solver call
produces new input values. The first determines the sparse layout; the second
determines the input precision needed during reconstruction.

## Pack the geometry once

Nearby trajectory samples share much of their interpolation support. The
planner groups nearby rows, collects their occupied columns, and permutes
columns into legal 2:4 groups. The input gather uses the same column map, so
the permutation does not change the represented product. If a group remains
overfull, disjoint legal layers retain its entries instead of pruning them.

Forward and adjoint interpolation receive independent packs because
transposition changes row degrees and conflicts. Both represent the stored
coefficient matrix and its conjugate transpose; packing does not claim to
undo the earlier coefficient quantization.

Implementation: [planner](../experiments/packing/run_fast_planner_matrix.py),
[exporter](../src/export_torchkbnufft_gpu_case.py), and
[structural oracle](../src/nufft_layout_oracle.py).

## Recover input precision on every call

A legal sparse layout does not recover information lost when a new FP32 input
panel is rounded to FP16. TrajTC forms a high component and a residual:

```text
B_hi  = round_fp16(B)
B_res = round_fp16(B - float32(B_hi))
C     ≈ G B_hi + G B_res
```

Both products use the same sparse coefficient pack and FP32 accumulation.
The residual is input-dependent and must be recomputed for every call, not
cached with the trajectory. It adds a second sparse product; the dense
control therefore uses the same two-component input representation.

Implementation: [integrated operator](../experiments/production/src/nufft_integrated_bench.cu)
and [reference arithmetic](../experiments/production/reference_sparse_solver.py).

## Keep the complete reconstruction on the GPU

Custom FFT endpoints, complex interpolation, SENSE, and deterministic
reductions implement the existing CG update sequence. The production build
contains no runtime cuFFT calls. Eager execution and CUDA Graph execution
are separate modes of the same reconstruction, not different quality targets.

Keeping the updates does not make the low-precision operator exactly linear
or Hermitian: per-call rounding depends on the input. Reconstruction quality,
eager/Graph equality, and longer-run stability are tested separately.

Implementation: [CG driver](../experiments/production/src/nufft_fp16x2_cg_bench.cu)
and [FFT endpoints](../experiments/production/src/nufft_custom_fft_forward_bench.cu).

## What each comparison establishes

| Path | Purpose | Interpolation | Comparison scope |
|---|---|---|---|
| TrajTC | Production method | Two sparse Tensor Core passes | Complete device-resident CG |
| Native cuFINUFFT | External system comparison | Tuned, quality-admitted NUFFT | Same input, reconstruction task, and quality criterion |
| Dense control | Sparse-execution attribution | Two dense Tensor Core passes | Same FFT, SENSE, CG, maps, and reductions |
| FP32 sparse reference | Numerical reference | TorchKbNufft sparse arithmetic | Same reconstruction setup; not a timing baseline |

## Reuse and limits

The coefficient pack is reusable across images and coil sensitivities with
the same trajectory and gridding parameters. Changing coordinates, grid
dimensions, kernel width, or panel dimensions requires repacking. Preparation
is separate from the timed reconstruction and is not amortized by one solve.

The shipped CUDA path is specialized to 256×256 images, a 512×512 grid,
kernel width six, eight coils, and SM120a. CPU oracle tests cover structural
invariants at small sizes; they do not imply a general-shape GPU API.

## Regenerate the overview

```bash
uv run --frozen python tools/render_overview.py
```

The vector diagram uses the first complete interleaf of the fixed spiral
fixture. [Its manifest](../assets/overview.json) records the input hash and
selection rule. Boxes describe the dataflow, not measured matrix occupancy
or a timing breakdown.
