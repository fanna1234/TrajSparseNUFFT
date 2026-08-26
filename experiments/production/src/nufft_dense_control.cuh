#pragma once

// Equal-quality dense Tensor Core control for the production residual-panel
// path. This header is included only after PackedDirection is defined.

__device__ __forceinline__ void mma_dense_f16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
        "r"(b[1]));
}

template <int kWarps, bool kConjugate, bool kAccumulate>
__global__ __launch_bounds__(128, 10) void nufft_gt_dense_complex_control(
    const uint32_t* __restrict__ group_offsets,
    const int32_t* __restrict__ group_row_ids,
    const int32_t* __restrict__ tile_col_ids,
    const half* __restrict__ tile_a_real_dense,
    const half* __restrict__ tile_a_imag_dense,
    const half* __restrict__ dense,
    uint32_t output_rows,
    float2* __restrict__ output,
    float output_scale) {
  extern __shared__ float partial[];
  float* partial_real = partial;
  float* partial_imag = partial + kWarps * kM * kN;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group4 = lane >> 2;
  const int tid4 = lane & 3;
  const int row0 = group4;
  const int row1 = group4 + 8;
  const int col0 = tid4 * 2;
  const int col1 = col0 + 1;
  const int group_index = blockIdx.x;

  float real_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float real_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float imag_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float imag_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const uint32_t tile_begin = group_offsets[group_index];
  const uint32_t tile_end = group_offsets[group_index + 1];

  for (uint32_t tile = tile_begin + warp; tile < tile_end; tile += kWarps) {
    const int32_t* map = tile_col_ids + static_cast<size_t>(tile) * kK;
#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
      const half* real_base =
          tile_a_real_dense + static_cast<size_t>(tile) * kM * kK + phase * 16;
      const half* imag_base =
          tile_a_imag_dense + static_cast<size_t>(tile) * kM * kK + phase * 16;
      const int a_k0 = tid4 * 2;
      const int a_k1 = a_k0 + 8;
      uint32_t real_a[4];
      uint32_t imag_a[4];
      real_a[0] = load_half2_aligned(real_base + row0 * kK + a_k0);
      real_a[1] = load_half2_aligned(real_base + row1 * kK + a_k0);
      real_a[2] = load_half2_aligned(real_base + row0 * kK + a_k1);
      real_a[3] = load_half2_aligned(real_base + row1 * kK + a_k1);
      imag_a[0] = load_half2_aligned(imag_base + row0 * kK + a_k0);
      imag_a[1] = load_half2_aligned(imag_base + row1 * kK + a_k0);
      imag_a[2] = load_half2_aligned(imag_base + row0 * kK + a_k1);
      imag_a[3] = load_half2_aligned(imag_base + row1 * kK + a_k1);

#pragma unroll
      for (int n_block = 0; n_block < 2; ++n_block) {
        const int n = group4 + n_block * 8;
        const int k_base = phase * 16 + tid4 * 2;
        uint32_t b_regs[2];
        b_regs[0] = pack_half2(dense[map[k_base] * kN + n],
                               dense[map[k_base + 1] * kN + n]);
        b_regs[1] = pack_half2(dense[map[k_base + 8] * kN + n],
                               dense[map[k_base + 9] * kN + n]);
        if (n_block == 0) {
          mma_dense_f16(real_lo, real_a, b_regs);
          mma_dense_f16(imag_lo, imag_a, b_regs);
        } else {
          mma_dense_f16(real_hi, real_a, b_regs);
          mma_dense_f16(imag_hi, imag_a, b_regs);
        }
      }
    }
  }

  float* warp_real = partial_real + warp * kM * kN;
  float* warp_imag = partial_imag + warp * kM * kN;
  warp_real[row0 * kN + col0] = real_lo[0];
  warp_real[row0 * kN + col1] = real_lo[1];
  warp_real[row1 * kN + col0] = real_lo[2];
  warp_real[row1 * kN + col1] = real_lo[3];
  warp_real[row0 * kN + col0 + 8] = real_hi[0];
  warp_real[row0 * kN + col1 + 8] = real_hi[1];
  warp_real[row1 * kN + col0 + 8] = real_hi[2];
  warp_real[row1 * kN + col1 + 8] = real_hi[3];
  warp_imag[row0 * kN + col0] = imag_lo[0];
  warp_imag[row0 * kN + col1] = imag_lo[1];
  warp_imag[row1 * kN + col0] = imag_lo[2];
  warp_imag[row1 * kN + col1] = imag_lo[3];
  warp_imag[row0 * kN + col0 + 8] = imag_hi[0];
  warp_imag[row0 * kN + col1 + 8] = imag_hi[1];
  warp_imag[row1 * kN + col0 + 8] = imag_hi[2];
  warp_imag[row1 * kN + col1 + 8] = imag_hi[3];
  __syncthreads();

  if (warp == 0) {
    float real_sum_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float real_sum_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float imag_sum_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float imag_sum_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int source_warp = 0; source_warp < kWarps; ++source_warp) {
      const float* source_real = partial_real + source_warp * kM * kN;
      const float* source_imag = partial_imag + source_warp * kM * kN;
      real_sum_lo[0] += source_real[row0 * kN + col0];
      real_sum_lo[1] += source_real[row0 * kN + col1];
      real_sum_lo[2] += source_real[row1 * kN + col0];
      real_sum_lo[3] += source_real[row1 * kN + col1];
      real_sum_hi[0] += source_real[row0 * kN + col0 + 8];
      real_sum_hi[1] += source_real[row0 * kN + col1 + 8];
      real_sum_hi[2] += source_real[row1 * kN + col0 + 8];
      real_sum_hi[3] += source_real[row1 * kN + col1 + 8];
      imag_sum_lo[0] += source_imag[row0 * kN + col0];
      imag_sum_lo[1] += source_imag[row0 * kN + col1];
      imag_sum_lo[2] += source_imag[row1 * kN + col0];
      imag_sum_lo[3] += source_imag[row1 * kN + col1];
      imag_sum_hi[0] += source_imag[row0 * kN + col0 + 8];
      imag_sum_hi[1] += source_imag[row0 * kN + col1 + 8];
      imag_sum_hi[2] += source_imag[row1 * kN + col0 + 8];
      imag_sum_hi[3] += source_imag[row1 * kN + col1 + 8];
    }

    const int logical_row0 = group_row_ids[group_index * kM + row0];
    const int logical_row1 = group_row_ids[group_index * kM + row1];
    if (logical_row0 >= 0) {
      store_complex_output<kAccumulate>(
          output, static_cast<size_t>(col0) * output_rows + logical_row0,
          output_scale *
              combine_complex_real<kConjugate>(real_sum_lo[0], imag_sum_hi[0]),
          output_scale *
              combine_complex_imag<kConjugate>(real_sum_hi[0], imag_sum_lo[0]));
      store_complex_output<kAccumulate>(
          output, static_cast<size_t>(col1) * output_rows + logical_row0,
          output_scale *
              combine_complex_real<kConjugate>(real_sum_lo[1], imag_sum_hi[1]),
          output_scale *
              combine_complex_imag<kConjugate>(real_sum_hi[1], imag_sum_lo[1]));
    }
    if (logical_row1 >= 0) {
      store_complex_output<kAccumulate>(
          output, static_cast<size_t>(col0) * output_rows + logical_row1,
          output_scale *
              combine_complex_real<kConjugate>(real_sum_lo[2], imag_sum_hi[2]),
          output_scale *
              combine_complex_imag<kConjugate>(real_sum_hi[2], imag_sum_lo[2]));
      store_complex_output<kAccumulate>(
          output, static_cast<size_t>(col1) * output_rows + logical_row1,
          output_scale *
              combine_complex_real<kConjugate>(real_sum_lo[3], imag_sum_hi[3]),
          output_scale *
              combine_complex_imag<kConjugate>(real_sum_hi[3], imag_sum_lo[3]));
    }
  }
}

template <bool kConjugate, bool kAccumulate>
void launch_dense_control(PackedDirection& packed,
                          const half* dense_panel,
                          float2* output,
                          float output_scale,
                          cudaStream_t stream) {
  constexpr int kWarps = 4;
  nufft_gt_dense_complex_control<kWarps, kConjugate, kAccumulate>
      <<<packed.h.groups, kWarps * 32,
         2 * kWarps * kM * kN * sizeof(float), stream>>>(
          packed.d.group_offsets, packed.d.group_row_ids,
          packed.d.tile_col_ids, packed.dense_a_real,
          packed.dense_a_imag, dense_panel, packed.h.output_size,
          output, output_scale);
}

inline void launch_dense_control_dispatch(PackedDirection& packed,
                                          const half* dense_panel,
                                          float2* output,
                                          bool conjugate,
                                          bool accumulate,
                                          float output_scale,
                                          cudaStream_t stream) {
  if (conjugate) {
    if (accumulate) {
      launch_dense_control<true, true>(packed, dense_panel, output,
                                       output_scale, stream);
    } else {
      launch_dense_control<true, false>(packed, dense_panel, output,
                                        output_scale, stream);
    }
  } else if (accumulate) {
    launch_dense_control<false, true>(packed, dense_panel, output,
                                      output_scale, stream);
  } else {
    launch_dense_control<false, false>(packed, dense_panel, output,
                                       output_scale, stream);
  }
}
