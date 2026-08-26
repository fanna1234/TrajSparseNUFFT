// Shape-specialized exploratory benchmark for a packed NUFFT interpolation view.
// Build twice with JKF_VARIANT=0 (CSR keeper) and JKF_VARIANT=1 (Sparse MMA).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#ifndef JKF_VARIANT
#define JKF_VARIANT 0
#endif

#ifndef JKF_FUSED_A32
#define JKF_FUSED_A32 1
#endif

#ifndef JKF_FUSED_LB10
#define JKF_FUSED_LB10 1
#endif

#ifndef JKF_FUSED_DIRECT_COMPLEX_OUTPUT
#define JKF_FUSED_DIRECT_COMPLEX_OUTPUT 0
#endif

namespace {

constexpr uint64_t kMagic = 0x4E55464654323431ull;
constexpr int kM = 16;
constexpr int kK = 32;
constexpr int kKComp = 16;
constexpr int kN = 16;

#pragma pack(push, 1)
struct Header {
  uint64_t magic;
  uint32_t version;
  uint32_t n_channels;
  uint32_t output_size;
  uint32_t n_cols;
  uint32_t active_rows;
  uint32_t nnz;
  uint32_t groups;
  uint32_t tiles;
  uint32_t residual_nnz;
  uint32_t m_tile;
  uint32_t k_tile;
};
#pragma pack(pop)

static_assert(sizeof(Header) == 52, "packed header size mismatch");

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t status_ = (expr);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status_));                                \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

template <typename T>
std::vector<T> read_array(const std::string& path, size_t count) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<T> data(count);
  stream.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(count * sizeof(T)));
  if (!stream || stream.gcount() != static_cast<std::streamsize>(count * sizeof(T))) {
    std::fprintf(stderr, "short read %s\n", path.c_str());
    std::exit(1);
  }
  return data;
}

Header read_header(const std::string& directory) {
  auto bytes = read_array<uint8_t>(directory + "/header.bin", sizeof(Header));
  Header header{};
  std::memcpy(&header, bytes.data(), sizeof(Header));
  if (header.magic != kMagic || header.version != 2 ||
      header.n_channels != kN || header.m_tile != kM || header.k_tile != kK) {
    std::fprintf(stderr, "unsupported input header\n");
    std::exit(1);
  }
  return header;
}

template <typename T>
T* device_copy(const std::vector<T>& host) {
  T* device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return device;
}

__device__ __forceinline__ uint32_t pack_half2(half lo, half hi) {
  return static_cast<uint32_t>(__half_as_ushort(lo)) |
         (static_cast<uint32_t>(__half_as_ushort(hi)) << 16);
}

__device__ __forceinline__ uint32_t load_half2_aligned(const half* pointer) {
  return *reinterpret_cast<const uint32_t*>(pointer);
}

template <int Selector>
__device__ __forceinline__ void mma_sp_f16(float (&d)[4],
                                            const uint32_t (&a)[4],
                                            const uint32_t (&b)[4],
                                            uint32_t metadata) {
  asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9, %10, %11}, "
      "{%0, %1, %2, %3}, "
      "%12, %13;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
        "r"(metadata), "n"(Selector));
}

__global__ void nufft_gt_csr_f16_f32(
    const uint32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ row_ids,
    const int32_t* __restrict__ cols,
    const half* __restrict__ values,
    const half* __restrict__ dense,
    uint32_t active_rows,
    float* __restrict__ output) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t local_row = blockIdx.x * 4u + static_cast<uint32_t>(warp_in_block);
  if (local_row >= active_rows || lane >= 8) return;

  float accum0 = 0.0f;
  float accum1 = 0.0f;
  const uint32_t begin = row_ptr[local_row];
  const uint32_t end = row_ptr[local_row + 1];
  const int channel = lane * 2;
  for (uint32_t index = begin; index < end; ++index) {
    const int col = cols[index];
    const float weight = __half2float(values[index]);
    const half2 pair = *reinterpret_cast<const half2*>(dense + col * kN + channel);
    accum0 = fmaf(weight, __half2float(__low2half(pair)), accum0);
    accum1 = fmaf(weight, __half2float(__high2half(pair)), accum1);
  }
  const int logical_row = row_ids[local_row];
  output[logical_row * kN + channel] = accum0;
  output[logical_row * kN + channel + 1] = accum1;
}

__global__ void nufft_gt_spmma_f16_f32(
    const uint32_t* __restrict__ group_offsets,
    const int32_t* __restrict__ group_row_ids,
    const int32_t* __restrict__ tile_col_ids,
    const half* __restrict__ tile_a_comp,
    const uint32_t* __restrict__ tile_pair_meta,
    const uint32_t* __restrict__ residual_row_ptr,
    const int32_t* __restrict__ residual_cols,
    const half* __restrict__ residual_values,
    const half* __restrict__ dense,
    float* __restrict__ output) {
  const int lane = threadIdx.x & 31;
  const int group4 = lane >> 2;
  const int tid4 = lane & 3;
  const int row0 = group4;
  const int row1 = group4 + 8;
  const int chunk0 = tid4;
  const int chunk1 = tid4 + 4;
  const int group_index = blockIdx.x;

  float accum_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float accum_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const uint32_t tile_begin = group_offsets[group_index];
  const uint32_t tile_end = group_offsets[group_index + 1];

  for (uint32_t tile = tile_begin; tile < tile_end; ++tile) {
    const half* a_base = tile_a_comp + static_cast<size_t>(tile) * kM * kKComp;
    uint32_t a_regs[4];
    a_regs[0] = pack_half2(a_base[row0 * kKComp + chunk0 * 2],
                           a_base[row0 * kKComp + chunk0 * 2 + 1]);
    a_regs[1] = pack_half2(a_base[row1 * kKComp + chunk0 * 2],
                           a_base[row1 * kKComp + chunk0 * 2 + 1]);
    a_regs[2] = pack_half2(a_base[row0 * kKComp + chunk1 * 2],
                           a_base[row0 * kKComp + chunk1 * 2 + 1]);
    a_regs[3] = pack_half2(a_base[row1 * kKComp + chunk1 * 2],
                           a_base[row1 * kKComp + chunk1 * 2 + 1]);

    const int k0 = tid4 * 2;
    const int32_t* map = tile_col_ids + static_cast<size_t>(tile) * kK;
    const uint32_t metadata = tile_pair_meta[static_cast<size_t>(tile) * kM +
                                              group4 * 2 + (tid4 & 1)];

#pragma unroll
    for (int n_block = 0; n_block < 2; ++n_block) {
      const int n = group4 + n_block * 8;
      uint32_t b_regs[4];
      b_regs[0] = pack_half2(dense[map[k0] * kN + n],
                             dense[map[k0 + 1] * kN + n]);
      b_regs[1] = pack_half2(dense[map[k0 + 8] * kN + n],
                             dense[map[k0 + 9] * kN + n]);
      b_regs[2] = pack_half2(dense[map[k0 + 16] * kN + n],
                             dense[map[k0 + 17] * kN + n]);
      b_regs[3] = pack_half2(dense[map[k0 + 24] * kN + n],
                             dense[map[k0 + 25] * kN + n]);
      if (n_block == 0) {
        mma_sp_f16<0>(accum_lo, a_regs, b_regs, metadata);
      } else {
        mma_sp_f16<0>(accum_hi, a_regs, b_regs, metadata);
      }
    }
  }

  const int col0 = tid4 * 2;
  const int col1 = col0 + 1;
  const int logical_row0 = group_row_ids[group_index * kM + row0];
  const int logical_row1 = group_row_ids[group_index * kM + row1];
  if (logical_row0 >= 0) {
    const uint32_t begin = residual_row_ptr[logical_row0];
    const uint32_t end = residual_row_ptr[logical_row0 + 1];
    for (uint32_t index = begin; index < end; ++index) {
      const int col = residual_cols[index];
      const float weight = __half2float(residual_values[index]);
      accum_lo[0] = fmaf(weight, __half2float(dense[col * kN + col0]), accum_lo[0]);
      accum_lo[1] = fmaf(weight, __half2float(dense[col * kN + col1]), accum_lo[1]);
      accum_hi[0] = fmaf(weight, __half2float(dense[col * kN + col0 + 8]), accum_hi[0]);
      accum_hi[1] = fmaf(weight, __half2float(dense[col * kN + col1 + 8]), accum_hi[1]);
    }
  }
  if (logical_row1 >= 0) {
    const uint32_t begin = residual_row_ptr[logical_row1];
    const uint32_t end = residual_row_ptr[logical_row1 + 1];
    for (uint32_t index = begin; index < end; ++index) {
      const int col = residual_cols[index];
      const float weight = __half2float(residual_values[index]);
      accum_lo[2] = fmaf(weight, __half2float(dense[col * kN + col0]), accum_lo[2]);
      accum_lo[3] = fmaf(weight, __half2float(dense[col * kN + col1]), accum_lo[3]);
      accum_hi[2] = fmaf(weight, __half2float(dense[col * kN + col0 + 8]), accum_hi[2]);
      accum_hi[3] = fmaf(weight, __half2float(dense[col * kN + col1 + 8]), accum_hi[3]);
    }
  }
  if (logical_row0 >= 0) {
    output[logical_row0 * kN + col0] = accum_lo[0];
    output[logical_row0 * kN + col1] = accum_lo[1];
    output[logical_row0 * kN + col0 + 8] = accum_hi[0];
    output[logical_row0 * kN + col1 + 8] = accum_hi[1];
  }
  if (logical_row1 >= 0) {
    output[logical_row1 * kN + col0] = accum_lo[2];
    output[logical_row1 * kN + col1] = accum_lo[3];
    output[logical_row1 * kN + col0 + 8] = accum_hi[2];
    output[logical_row1 * kN + col1 + 8] = accum_hi[3];
  }
}

template <int kWarps>
__global__ void nufft_gt_spmma_split_f16_f32(
    const uint32_t* __restrict__ group_offsets,
    const int32_t* __restrict__ group_row_ids,
    const int32_t* __restrict__ tile_col_ids,
    const half* __restrict__ tile_a_comp,
    const uint32_t* __restrict__ tile_pair_meta,
    const half* __restrict__ dense,
    float* __restrict__ output) {
  extern __shared__ float partial[];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group4 = lane >> 2;
  const int tid4 = lane & 3;
  const int row0 = group4;
  const int row1 = group4 + 8;
  const int chunk0 = tid4;
  const int chunk1 = tid4 + 4;
  const int group_index = blockIdx.x;

  float accum_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float accum_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const uint32_t tile_begin = group_offsets[group_index];
  const uint32_t tile_end = group_offsets[group_index + 1];

  for (uint32_t tile = tile_begin + warp; tile < tile_end; tile += kWarps) {
    const half* a_base = tile_a_comp + static_cast<size_t>(tile) * kM * kKComp;
    uint32_t a_regs[4];
    a_regs[0] = pack_half2(a_base[row0 * kKComp + chunk0 * 2],
                           a_base[row0 * kKComp + chunk0 * 2 + 1]);
    a_regs[1] = pack_half2(a_base[row1 * kKComp + chunk0 * 2],
                           a_base[row1 * kKComp + chunk0 * 2 + 1]);
    a_regs[2] = pack_half2(a_base[row0 * kKComp + chunk1 * 2],
                           a_base[row0 * kKComp + chunk1 * 2 + 1]);
    a_regs[3] = pack_half2(a_base[row1 * kKComp + chunk1 * 2],
                           a_base[row1 * kKComp + chunk1 * 2 + 1]);

    const int k0 = tid4 * 2;
    const int32_t* map = tile_col_ids + static_cast<size_t>(tile) * kK;
    const uint32_t metadata = tile_pair_meta[static_cast<size_t>(tile) * kM +
                                              group4 * 2 + (tid4 & 1)];
#pragma unroll
    for (int n_block = 0; n_block < 2; ++n_block) {
      const int n = group4 + n_block * 8;
      uint32_t b_regs[4];
      b_regs[0] = pack_half2(dense[map[k0] * kN + n],
                             dense[map[k0 + 1] * kN + n]);
      b_regs[1] = pack_half2(dense[map[k0 + 8] * kN + n],
                             dense[map[k0 + 9] * kN + n]);
      b_regs[2] = pack_half2(dense[map[k0 + 16] * kN + n],
                             dense[map[k0 + 17] * kN + n]);
      b_regs[3] = pack_half2(dense[map[k0 + 24] * kN + n],
                             dense[map[k0 + 25] * kN + n]);
      if (n_block == 0) {
        mma_sp_f16<0>(accum_lo, a_regs, b_regs, metadata);
      } else {
        mma_sp_f16<0>(accum_hi, a_regs, b_regs, metadata);
      }
    }
  }

  const int col0 = tid4 * 2;
  const int col1 = col0 + 1;
  float* warp_partial = partial + warp * kM * kN;
  warp_partial[row0 * kN + col0] = accum_lo[0];
  warp_partial[row0 * kN + col1] = accum_lo[1];
  warp_partial[row1 * kN + col0] = accum_lo[2];
  warp_partial[row1 * kN + col1] = accum_lo[3];
  warp_partial[row0 * kN + col0 + 8] = accum_hi[0];
  warp_partial[row0 * kN + col1 + 8] = accum_hi[1];
  warp_partial[row1 * kN + col0 + 8] = accum_hi[2];
  warp_partial[row1 * kN + col1 + 8] = accum_hi[3];
  __syncthreads();

  if (warp == 0) {
    float sum_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int source_warp = 0; source_warp < kWarps; ++source_warp) {
      const float* source = partial + source_warp * kM * kN;
      sum_lo[0] += source[row0 * kN + col0];
      sum_lo[1] += source[row0 * kN + col1];
      sum_lo[2] += source[row1 * kN + col0];
      sum_lo[3] += source[row1 * kN + col1];
      sum_hi[0] += source[row0 * kN + col0 + 8];
      sum_hi[1] += source[row0 * kN + col1 + 8];
      sum_hi[2] += source[row1 * kN + col0 + 8];
      sum_hi[3] += source[row1 * kN + col1 + 8];
    }
    const int logical_row0 = group_row_ids[group_index * kM + row0];
    const int logical_row1 = group_row_ids[group_index * kM + row1];
    if (logical_row0 >= 0) {
      output[logical_row0 * kN + col0] = sum_lo[0];
      output[logical_row0 * kN + col1] = sum_lo[1];
      output[logical_row0 * kN + col0 + 8] = sum_hi[0];
      output[logical_row0 * kN + col1 + 8] = sum_hi[1];
    }
    if (logical_row1 >= 0) {
      output[logical_row1 * kN + col0] = sum_lo[2];
      output[logical_row1 * kN + col1] = sum_lo[3];
      output[logical_row1 * kN + col0 + 8] = sum_hi[2];
      output[logical_row1 * kN + col1 + 8] = sum_hi[3];
    }
  }
}

template <bool kConjugate>
__device__ __forceinline__ float combine_complex_real(float real_term,
                                                       float imag_term) {
  if constexpr (kConjugate) return real_term + imag_term;
  return real_term - imag_term;
}

template <bool kConjugate>
__device__ __forceinline__ float combine_complex_imag(float real_term,
                                                       float imag_term) {
  if constexpr (kConjugate) return real_term - imag_term;
  return real_term + imag_term;
}

template <bool kAccumulate>
__device__ __forceinline__ void store_complex_output(
    float2* output, size_t index, float real_value, float imag_value) {
  if constexpr (kAccumulate) {
    const float2 prior = output[index];
    real_value += prior.x;
    imag_value += prior.y;
  }
  output[index] = make_float2(real_value, imag_value);
}

template <int kWarps, bool kDataConsistency = false,
          bool kConjugate = false, bool kAccumulate = false>
#if JKF_FUSED_LB10
__global__ __launch_bounds__(128, 10)
#else
__global__ void nufft_gt_spmma_complex_fused_f16_f32(
#endif
#if JKF_FUSED_LB10
void nufft_gt_spmma_complex_fused_f16_f32(
#endif
    const uint32_t* __restrict__ group_offsets,
    const int32_t* __restrict__ group_row_ids,
    const int32_t* __restrict__ tile_col_ids,
    const half* __restrict__ tile_a_real_comp,
    const uint32_t* __restrict__ tile_real_pair_meta,
    const half* __restrict__ tile_a_imag_comp,
    const uint32_t* __restrict__ tile_imag_pair_meta,
    const half* __restrict__ dense,
    uint32_t output_rows,
    const float2* __restrict__ dc_measurements,
    const float* __restrict__ dc_density,
    half* __restrict__ dc_output,
    float* __restrict__ output,
    float output_scale) {
  static_assert(!(kDataConsistency && kConjugate));
  extern __shared__ float partial[];
  float* partial_real = partial;
  float* partial_imag = partial + kWarps * kM * kN;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group4 = lane >> 2;
  const int tid4 = lane & 3;
  const int row0 = group4;
  const int row1 = group4 + 8;
  const int chunk0 = tid4;
  const int chunk1 = tid4 + 4;
  const int group_index = blockIdx.x;

  float real_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float real_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float imag_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float imag_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const uint32_t tile_begin = group_offsets[group_index];
  const uint32_t tile_end = group_offsets[group_index + 1];

  for (uint32_t tile = tile_begin + warp; tile < tile_end; tile += kWarps) {
    const half* real_base =
        tile_a_real_comp + static_cast<size_t>(tile) * kM * kKComp;
    const half* imag_base =
        tile_a_imag_comp + static_cast<size_t>(tile) * kM * kKComp;
    uint32_t real_a[4];
    uint32_t imag_a[4];
#if JKF_FUSED_A32
    real_a[0] = load_half2_aligned(real_base + row0 * kKComp + chunk0 * 2);
    real_a[1] = load_half2_aligned(real_base + row1 * kKComp + chunk0 * 2);
    real_a[2] = load_half2_aligned(real_base + row0 * kKComp + chunk1 * 2);
    real_a[3] = load_half2_aligned(real_base + row1 * kKComp + chunk1 * 2);
    imag_a[0] = load_half2_aligned(imag_base + row0 * kKComp + chunk0 * 2);
    imag_a[1] = load_half2_aligned(imag_base + row1 * kKComp + chunk0 * 2);
    imag_a[2] = load_half2_aligned(imag_base + row0 * kKComp + chunk1 * 2);
    imag_a[3] = load_half2_aligned(imag_base + row1 * kKComp + chunk1 * 2);
#else
    real_a[0] = pack_half2(real_base[row0 * kKComp + chunk0 * 2],
                           real_base[row0 * kKComp + chunk0 * 2 + 1]);
    real_a[1] = pack_half2(real_base[row1 * kKComp + chunk0 * 2],
                           real_base[row1 * kKComp + chunk0 * 2 + 1]);
    real_a[2] = pack_half2(real_base[row0 * kKComp + chunk1 * 2],
                           real_base[row0 * kKComp + chunk1 * 2 + 1]);
    real_a[3] = pack_half2(real_base[row1 * kKComp + chunk1 * 2],
                           real_base[row1 * kKComp + chunk1 * 2 + 1]);
    imag_a[0] = pack_half2(imag_base[row0 * kKComp + chunk0 * 2],
                           imag_base[row0 * kKComp + chunk0 * 2 + 1]);
    imag_a[1] = pack_half2(imag_base[row1 * kKComp + chunk0 * 2],
                           imag_base[row1 * kKComp + chunk0 * 2 + 1]);
    imag_a[2] = pack_half2(imag_base[row0 * kKComp + chunk1 * 2],
                           imag_base[row0 * kKComp + chunk1 * 2 + 1]);
    imag_a[3] = pack_half2(imag_base[row1 * kKComp + chunk1 * 2],
                           imag_base[row1 * kKComp + chunk1 * 2 + 1]);
#endif

    const int k0 = tid4 * 2;
    const int32_t* map = tile_col_ids + static_cast<size_t>(tile) * kK;
    const size_t meta_index =
        static_cast<size_t>(tile) * kM + group4 * 2 + (tid4 & 1);
    const uint32_t real_metadata = tile_real_pair_meta[meta_index];
    const uint32_t imag_metadata = tile_imag_pair_meta[meta_index];
#pragma unroll
    for (int n_block = 0; n_block < 2; ++n_block) {
      const int n = group4 + n_block * 8;
      uint32_t b_regs[4];
      b_regs[0] = pack_half2(dense[map[k0] * kN + n],
                             dense[map[k0 + 1] * kN + n]);
      b_regs[1] = pack_half2(dense[map[k0 + 8] * kN + n],
                             dense[map[k0 + 9] * kN + n]);
      b_regs[2] = pack_half2(dense[map[k0 + 16] * kN + n],
                             dense[map[k0 + 17] * kN + n]);
      b_regs[3] = pack_half2(dense[map[k0 + 24] * kN + n],
                             dense[map[k0 + 25] * kN + n]);
      if (n_block == 0) {
        mma_sp_f16<0>(real_lo, real_a, b_regs, real_metadata);
        mma_sp_f16<0>(imag_lo, imag_a, b_regs, imag_metadata);
      } else {
        mma_sp_f16<0>(real_hi, real_a, b_regs, real_metadata);
        mma_sp_f16<0>(imag_hi, imag_a, b_regs, imag_metadata);
      }
    }
  }

  const int col0 = tid4 * 2;
  const int col1 = col0 + 1;
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
    if constexpr (kDataConsistency) {
      if (logical_row0 >= 0) {
        const float density = dc_density[logical_row0];
        const float2 measurement0 =
            dc_measurements[static_cast<size_t>(col0) * output_rows +
                            logical_row0];
        const float2 measurement1 =
            dc_measurements[static_cast<size_t>(col1) * output_rows +
                            logical_row0];
        const half2 residual_real = __floats2half2_rn(
            (real_sum_lo[0] - imag_sum_hi[0] - measurement0.x) * density,
            (real_sum_lo[1] - imag_sum_hi[1] - measurement1.x) * density);
        const half2 residual_imag = __floats2half2_rn(
            (real_sum_hi[0] + imag_sum_lo[0] - measurement0.y) * density,
            (real_sum_hi[1] + imag_sum_lo[1] - measurement1.y) * density);
        *reinterpret_cast<half2*>(
            dc_output + static_cast<size_t>(logical_row0) * kN + col0) =
            residual_real;
        *reinterpret_cast<half2*>(
            dc_output + static_cast<size_t>(logical_row0) * kN + col0 + 8) =
            residual_imag;
      }
      if (logical_row1 >= 0) {
        const float density = dc_density[logical_row1];
        const float2 measurement0 =
            dc_measurements[static_cast<size_t>(col0) * output_rows +
                            logical_row1];
        const float2 measurement1 =
            dc_measurements[static_cast<size_t>(col1) * output_rows +
                            logical_row1];
        const half2 residual_real = __floats2half2_rn(
            (real_sum_lo[2] - imag_sum_hi[2] - measurement0.x) * density,
            (real_sum_lo[3] - imag_sum_hi[3] - measurement1.x) * density);
        const half2 residual_imag = __floats2half2_rn(
            (real_sum_hi[2] + imag_sum_lo[2] - measurement0.y) * density,
            (real_sum_hi[3] + imag_sum_lo[3] - measurement1.y) * density);
        *reinterpret_cast<half2*>(
            dc_output + static_cast<size_t>(logical_row1) * kN + col0) =
            residual_real;
        *reinterpret_cast<half2*>(
            dc_output + static_cast<size_t>(logical_row1) * kN + col0 + 8) =
            residual_imag;
      }
    } else {
#if JKF_FUSED_DIRECT_COMPLEX_OUTPUT
      float2* complex_output = reinterpret_cast<float2*>(output);
      if (logical_row0 >= 0) {
        store_complex_output<kAccumulate>(
            complex_output,
            static_cast<size_t>(col0) * output_rows + logical_row0,
            output_scale *
                combine_complex_real<kConjugate>(real_sum_lo[0], imag_sum_hi[0]),
            output_scale *
                combine_complex_imag<kConjugate>(real_sum_hi[0], imag_sum_lo[0]));
        store_complex_output<kAccumulate>(
            complex_output,
            static_cast<size_t>(col1) * output_rows + logical_row0,
            output_scale *
                combine_complex_real<kConjugate>(real_sum_lo[1], imag_sum_hi[1]),
            output_scale *
                combine_complex_imag<kConjugate>(real_sum_hi[1], imag_sum_lo[1]));
      }
      if (logical_row1 >= 0) {
        store_complex_output<kAccumulate>(
            complex_output,
            static_cast<size_t>(col0) * output_rows + logical_row1,
            output_scale *
                combine_complex_real<kConjugate>(real_sum_lo[2], imag_sum_hi[2]),
            output_scale *
                combine_complex_imag<kConjugate>(real_sum_hi[2], imag_sum_lo[2]));
        store_complex_output<kAccumulate>(
            complex_output,
            static_cast<size_t>(col1) * output_rows + logical_row1,
            output_scale *
                combine_complex_real<kConjugate>(real_sum_lo[3], imag_sum_hi[3]),
            output_scale *
                combine_complex_imag<kConjugate>(real_sum_hi[3], imag_sum_lo[3]));
      }
#else
      if (logical_row0 >= 0) {
        output[logical_row0 * kN + col0] =
            combine_complex_real<kConjugate>(real_sum_lo[0], imag_sum_hi[0]);
        output[logical_row0 * kN + col1] =
            combine_complex_real<kConjugate>(real_sum_lo[1], imag_sum_hi[1]);
        output[logical_row0 * kN + col0 + 8] =
            combine_complex_imag<kConjugate>(real_sum_hi[0], imag_sum_lo[0]);
        output[logical_row0 * kN + col1 + 8] =
            combine_complex_imag<kConjugate>(real_sum_hi[1], imag_sum_lo[1]);
      }
      if (logical_row1 >= 0) {
        output[logical_row1 * kN + col0] =
            combine_complex_real<kConjugate>(real_sum_lo[2], imag_sum_hi[2]);
        output[logical_row1 * kN + col1] =
            combine_complex_real<kConjugate>(real_sum_lo[3], imag_sum_hi[3]);
        output[logical_row1 * kN + col0 + 8] =
            combine_complex_imag<kConjugate>(real_sum_hi[2], imag_sum_lo[2]);
        output[logical_row1 * kN + col1 + 8] =
            combine_complex_imag<kConjugate>(real_sum_hi[3], imag_sum_lo[3]);
      }
#endif
    }
  }
}

__global__ void nufft_gt_spmma_complex_group_packed_f16_f32(
    const uint32_t* __restrict__ group_offsets,
    const int32_t* __restrict__ group_row_ids,
    const int32_t* __restrict__ tile_col_ids,
    const half* __restrict__ tile_a_real_comp,
    const uint32_t* __restrict__ tile_real_pair_meta,
    const half* __restrict__ tile_a_imag_comp,
    const uint32_t* __restrict__ tile_imag_pair_meta,
    const half* __restrict__ dense,
    uint32_t groups,
    float* __restrict__ output) {
  const int warps_per_block = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const uint32_t group_index =
      blockIdx.x * static_cast<uint32_t>(warps_per_block) + warp;
  if (group_index >= groups) return;

  const int lane = threadIdx.x & 31;
  const int group4 = lane >> 2;
  const int tid4 = lane & 3;
  const int row0 = group4;
  const int row1 = group4 + 8;
  const int chunk0 = tid4;
  const int chunk1 = tid4 + 4;

  float real_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float real_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float imag_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float imag_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const uint32_t tile_begin = group_offsets[group_index];
  const uint32_t tile_end = group_offsets[group_index + 1];

  for (uint32_t tile = tile_begin; tile < tile_end; ++tile) {
    const half* real_base =
        tile_a_real_comp + static_cast<size_t>(tile) * kM * kKComp;
    const half* imag_base =
        tile_a_imag_comp + static_cast<size_t>(tile) * kM * kKComp;
    uint32_t real_a[4];
    uint32_t imag_a[4];
    real_a[0] = pack_half2(real_base[row0 * kKComp + chunk0 * 2],
                           real_base[row0 * kKComp + chunk0 * 2 + 1]);
    real_a[1] = pack_half2(real_base[row1 * kKComp + chunk0 * 2],
                           real_base[row1 * kKComp + chunk0 * 2 + 1]);
    real_a[2] = pack_half2(real_base[row0 * kKComp + chunk1 * 2],
                           real_base[row0 * kKComp + chunk1 * 2 + 1]);
    real_a[3] = pack_half2(real_base[row1 * kKComp + chunk1 * 2],
                           real_base[row1 * kKComp + chunk1 * 2 + 1]);
    imag_a[0] = pack_half2(imag_base[row0 * kKComp + chunk0 * 2],
                           imag_base[row0 * kKComp + chunk0 * 2 + 1]);
    imag_a[1] = pack_half2(imag_base[row1 * kKComp + chunk0 * 2],
                           imag_base[row1 * kKComp + chunk0 * 2 + 1]);
    imag_a[2] = pack_half2(imag_base[row0 * kKComp + chunk1 * 2],
                           imag_base[row0 * kKComp + chunk1 * 2 + 1]);
    imag_a[3] = pack_half2(imag_base[row1 * kKComp + chunk1 * 2],
                           imag_base[row1 * kKComp + chunk1 * 2 + 1]);

    const int k0 = tid4 * 2;
    const int32_t* map = tile_col_ids + static_cast<size_t>(tile) * kK;
    const size_t meta_index =
        static_cast<size_t>(tile) * kM + group4 * 2 + (tid4 & 1);
    const uint32_t real_metadata = tile_real_pair_meta[meta_index];
    const uint32_t imag_metadata = tile_imag_pair_meta[meta_index];
#pragma unroll
    for (int n_block = 0; n_block < 2; ++n_block) {
      const int n = group4 + n_block * 8;
      uint32_t b_regs[4];
      b_regs[0] = pack_half2(dense[map[k0] * kN + n],
                             dense[map[k0 + 1] * kN + n]);
      b_regs[1] = pack_half2(dense[map[k0 + 8] * kN + n],
                             dense[map[k0 + 9] * kN + n]);
      b_regs[2] = pack_half2(dense[map[k0 + 16] * kN + n],
                             dense[map[k0 + 17] * kN + n]);
      b_regs[3] = pack_half2(dense[map[k0 + 24] * kN + n],
                             dense[map[k0 + 25] * kN + n]);
      if (n_block == 0) {
        mma_sp_f16<0>(real_lo, real_a, b_regs, real_metadata);
        mma_sp_f16<0>(imag_lo, imag_a, b_regs, imag_metadata);
      } else {
        mma_sp_f16<0>(real_hi, real_a, b_regs, real_metadata);
        mma_sp_f16<0>(imag_hi, imag_a, b_regs, imag_metadata);
      }
    }
  }

  const int col0 = tid4 * 2;
  const int col1 = col0 + 1;
  const int logical_row0 = group_row_ids[group_index * kM + row0];
  const int logical_row1 = group_row_ids[group_index * kM + row1];
  if (logical_row0 >= 0) {
    output[logical_row0 * kN + col0] = real_lo[0] - imag_hi[0];
    output[logical_row0 * kN + col1] = real_lo[1] - imag_hi[1];
    output[logical_row0 * kN + col0 + 8] = real_hi[0] + imag_lo[0];
    output[logical_row0 * kN + col1 + 8] = real_hi[1] + imag_lo[1];
  }
  if (logical_row1 >= 0) {
    output[logical_row1 * kN + col0] = real_lo[2] - imag_hi[2];
    output[logical_row1 * kN + col1] = real_lo[3] - imag_hi[3];
    output[logical_row1 * kN + col0 + 8] = real_hi[2] + imag_lo[2];
    output[logical_row1 * kN + col1 + 8] = real_hi[3] + imag_lo[3];
  }
}

template <bool kConjugate = false>
__global__ void complex_combine_f32(const float* __restrict__ real_part,
                                    const float* __restrict__ imag_part,
                                    uint32_t rows,
                                    float* __restrict__ output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t count = rows * 8u;
  if (index >= count) return;
  const uint32_t row = index >> 3;
  const uint32_t channel = index & 7u;
  const size_t base = static_cast<size_t>(row) * kN;
  output[base + channel] = combine_complex_real<kConjugate>(
      real_part[base + channel], imag_part[base + channel + 8]);
  output[base + channel + 8] = combine_complex_imag<kConjugate>(
      real_part[base + channel + 8], imag_part[base + channel]);
}

struct DeviceData {
  uint32_t* group_offsets = nullptr;
  int32_t* group_row_ids = nullptr;
  int32_t* tile_col_ids = nullptr;
  half* tile_a_comp = nullptr;
  uint32_t* tile_pair_meta = nullptr;
  half* tile_a_imag_comp = nullptr;
  uint32_t* tile_imag_pair_meta = nullptr;
  uint32_t* csr_row_ptr = nullptr;
  int32_t* csr_row_ids = nullptr;
  int32_t* csr_cols = nullptr;
  half* csr_vals = nullptr;
  uint32_t* residual_row_ptr = nullptr;
  int32_t* residual_cols = nullptr;
  half* residual_vals = nullptr;
  half* dense = nullptr;
  float* output = nullptr;
  float* real_output = nullptr;
  float* imag_output = nullptr;
};

void launch_variant(const Header& h, const DeviceData& d, cudaStream_t stream,
                    bool conjugate = false, bool accumulate = false,
                    float output_scale = 1.0f) {
#if JKF_VARIANT == 0
  const dim3 block(128);
  const dim3 grid((h.active_rows + 3) / 4);
  nufft_gt_csr_f16_f32<<<grid, block, 0, stream>>>(
      d.csr_row_ptr, d.csr_row_ids, d.csr_cols, d.csr_vals, d.dense,
      h.active_rows, d.output);
#elif JKF_VARIANT == 1
  nufft_gt_spmma_f16_f32<<<h.groups, 32, 0, stream>>>(
      d.group_offsets, d.group_row_ids, d.tile_col_ids, d.tile_a_comp,
      d.tile_pair_meta, d.residual_row_ptr, d.residual_cols, d.residual_vals,
      d.dense, d.output);
#elif JKF_VARIANT == 2
  constexpr int kWarps = 4;
  nufft_gt_spmma_split_f16_f32<kWarps>
      <<<h.groups, kWarps * 32, kWarps * kM * kN * sizeof(float), stream>>>(
      d.group_offsets, d.group_row_ids, d.tile_col_ids, d.tile_a_comp,
      d.tile_pair_meta, d.dense, d.output);
#elif JKF_VARIANT == 3
  constexpr int kWarps = 2;
  nufft_gt_spmma_split_f16_f32<kWarps>
      <<<h.groups, kWarps * 32, kWarps * kM * kN * sizeof(float), stream>>>(
      d.group_offsets, d.group_row_ids, d.tile_col_ids, d.tile_a_comp,
      d.tile_pair_meta, d.dense, d.output);
#elif JKF_VARIANT == 4
  constexpr int kWarps = 8;
  nufft_gt_spmma_split_f16_f32<kWarps>
      <<<h.groups, kWarps * 32, kWarps * kM * kN * sizeof(float), stream>>>(
      d.group_offsets, d.group_row_ids, d.tile_col_ids, d.tile_a_comp,
      d.tile_pair_meta, d.dense, d.output);
#elif JKF_VARIANT == 5
  constexpr int kWarps = 4;
  if (conjugate) {
    if (accumulate) {
      nufft_gt_spmma_complex_fused_f16_f32<kWarps, false, true, true>
          <<<h.groups, kWarps * 32,
             2 * kWarps * kM * kN * sizeof(float), stream>>>(
          d.group_offsets, d.group_row_ids, d.tile_col_ids, d.tile_a_comp,
          d.tile_pair_meta, d.tile_a_imag_comp, d.tile_imag_pair_meta,
          d.dense, h.output_size, nullptr, nullptr, nullptr, d.output,
          output_scale);
    } else {
      nufft_gt_spmma_complex_fused_f16_f32<kWarps, false, true>
          <<<h.groups, kWarps * 32,
             2 * kWarps * kM * kN * sizeof(float), stream>>>(
          d.group_offsets, d.group_row_ids, d.tile_col_ids, d.tile_a_comp,
          d.tile_pair_meta, d.tile_a_imag_comp, d.tile_imag_pair_meta,
          d.dense, h.output_size, nullptr, nullptr, nullptr, d.output,
          output_scale);
    }
  } else {
    if (accumulate) {
      nufft_gt_spmma_complex_fused_f16_f32<kWarps, false, false, true>
          <<<h.groups, kWarps * 32,
             2 * kWarps * kM * kN * sizeof(float), stream>>>(
          d.group_offsets, d.group_row_ids, d.tile_col_ids, d.tile_a_comp,
          d.tile_pair_meta, d.tile_a_imag_comp, d.tile_imag_pair_meta,
          d.dense, h.output_size, nullptr, nullptr, nullptr, d.output,
          output_scale);
    } else {
      nufft_gt_spmma_complex_fused_f16_f32<kWarps>
          <<<h.groups, kWarps * 32,
             2 * kWarps * kM * kN * sizeof(float), stream>>>(
          d.group_offsets, d.group_row_ids, d.tile_col_ids, d.tile_a_comp,
          d.tile_pair_meta, d.tile_a_imag_comp, d.tile_imag_pair_meta,
          d.dense, h.output_size, nullptr, nullptr, nullptr, d.output,
          output_scale);
    }
  }
#elif JKF_VARIANT == 6
  constexpr int kWarps = 4;
  constexpr size_t kSharedBytes = kWarps * kM * kN * sizeof(float);
  nufft_gt_spmma_split_f16_f32<kWarps>
      <<<h.groups, kWarps * 32, kSharedBytes, stream>>>(
      d.group_offsets, d.group_row_ids, d.tile_col_ids, d.tile_a_comp,
      d.tile_pair_meta, d.dense, d.real_output);
  nufft_gt_spmma_split_f16_f32<kWarps>
      <<<h.groups, kWarps * 32, kSharedBytes, stream>>>(
      d.group_offsets, d.group_row_ids, d.tile_col_ids, d.tile_a_imag_comp,
      d.tile_imag_pair_meta, d.dense, d.imag_output);
  constexpr int kCombineThreads = 256;
  const uint32_t combine_blocks =
      (h.output_size * 8u + kCombineThreads - 1) / kCombineThreads;
  if (conjugate) {
    complex_combine_f32<true><<<combine_blocks, kCombineThreads, 0, stream>>>(
        d.real_output, d.imag_output, h.output_size, d.output);
  } else {
    complex_combine_f32<false><<<combine_blocks, kCombineThreads, 0, stream>>>(
        d.real_output, d.imag_output, h.output_size, d.output);
  }
#elif JKF_VARIANT >= 7 && JKF_VARIANT <= 9
#if JKF_VARIANT == 7
  constexpr int kGroupsPerBlock = 2;
#elif JKF_VARIANT == 8
  constexpr int kGroupsPerBlock = 4;
#else
  constexpr int kGroupsPerBlock = 8;
#endif
  const uint32_t packed_blocks =
      (h.groups + kGroupsPerBlock - 1) / kGroupsPerBlock;
  nufft_gt_spmma_complex_group_packed_f16_f32
      <<<packed_blocks, kGroupsPerBlock * 32, 0, stream>>>(
      d.group_offsets, d.group_row_ids, d.tile_col_ids, d.tile_a_comp,
      d.tile_pair_meta, d.tile_a_imag_comp, d.tile_imag_pair_meta, d.dense,
      h.groups, d.output);
#else
#error "Unsupported JKF_VARIANT"
#endif
}

const char* variant_name() {
#if JKF_VARIANT == 0
  return "csr_keeper";
#elif JKF_VARIANT == 1
  return "spmma_candidate";
#elif JKF_VARIANT == 2
  return "spmma_w4_candidate";
#elif JKF_VARIANT == 3
  return "spmma_w2_candidate";
#elif JKF_VARIANT == 4
  return "spmma_w8_candidate";
#elif JKF_VARIANT == 5
  return "spmma_complex_fused_w4_candidate";
#elif JKF_VARIANT == 6
  return "spmma_complex_unfused_w4_baseline";
#elif JKF_VARIANT == 7
  return "spmma_complex_group2_candidate";
#elif JKF_VARIANT == 8
  return "spmma_complex_group4_candidate";
#else
  return "spmma_complex_group8_candidate";
#endif
}

}  // namespace

#ifndef JKF_NO_COMPONENT_MAIN
int main(int argc, char** argv) {
#if JKF_VARIANT >= 5 && JKF_VARIANT <= 9
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s REAL_DATA_DIR IMAG_DATA_DIR [warmup=200] "
                 "[iters=2000] [stress=0] [graph=0]\n",
                 argv[0]);
    return 2;
  }
  const std::string directory = argv[1];
  const std::string imag_directory = argv[2];
  constexpr int kOptionBase = 3;
#else
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s DATA_DIR [warmup=200] [iters=2000] [stress=0] "
                 "[graph=0]\n",
                 argv[0]);
    return 2;
  }
  const std::string directory = argv[1];
  constexpr int kOptionBase = 2;
#endif
  const int warmup = argc > kOptionBase ? std::atoi(argv[kOptionBase]) : 200;
  const int iterations =
      argc > kOptionBase + 1 ? std::atoi(argv[kOptionBase + 1]) : 2000;
  const int stress =
      argc > kOptionBase + 2 ? std::atoi(argv[kOptionBase + 2]) : 0;
  const bool graph_mode =
      argc > kOptionBase + 3 ? std::atoi(argv[kOptionBase + 3]) != 0 : false;
  const Header h = read_header(directory);
#if JKF_VARIANT >= 5 && JKF_VARIANT <= 9
  const Header imag_h = read_header(imag_directory);
  if (std::memcmp(&h, &imag_h, sizeof(Header)) != 0) {
    std::fprintf(stderr, "real/imag headers do not match\n");
    return 2;
  }
#endif
#if JKF_VARIANT >= 2 && JKF_VARIANT <= 9
  if (h.residual_nnz != 0) {
    std::fprintf(stderr, "split candidate requires split24 data with no residual\n");
    return 2;
  }
#endif

  auto group_offsets = read_array<uint32_t>(directory + "/group_offsets.bin", h.groups + 1);
  auto group_row_ids = read_array<int32_t>(directory + "/group_row_ids.bin",
                                            static_cast<size_t>(h.groups) * kM);
  auto tile_col_ids = read_array<int32_t>(directory + "/tile_col_ids.bin",
                                           static_cast<size_t>(h.tiles) * kK);
  auto tile_a_bits = read_array<uint16_t>(directory + "/tile_a_comp.bin",
                                           static_cast<size_t>(h.tiles) * kM * kKComp);
  auto tile_pair_meta = read_array<uint32_t>(directory + "/tile_pair_meta.bin",
                                              static_cast<size_t>(h.tiles) * kM);
  auto csr_row_ptr = read_array<uint32_t>(directory + "/csr_row_ptr.bin", h.active_rows + 1);
  auto csr_row_ids = read_array<int32_t>(directory + "/csr_row_ids.bin", h.active_rows);
  auto csr_cols = read_array<int32_t>(directory + "/csr_cols.bin", h.nnz);
  auto csr_val_bits = read_array<uint16_t>(directory + "/csr_vals.bin", h.nnz);
  auto residual_row_ptr = read_array<uint32_t>(directory + "/residual_row_ptr.bin",
                                                h.output_size + 1);
  auto residual_cols = read_array<int32_t>(directory + "/residual_cols.bin",
                                            h.residual_nnz);
  auto residual_val_bits = read_array<uint16_t>(directory + "/residual_vals.bin",
                                                 h.residual_nnz);
  auto dense_bits = read_array<uint16_t>(directory + "/dense_input.bin",
                                          static_cast<size_t>(h.n_cols) * kN);
  auto reference = read_array<float>(directory + "/reference.bin",
                                      static_cast<size_t>(h.output_size) * kN);
#if JKF_VARIANT >= 5 && JKF_VARIANT <= 9
  auto imag_group_offsets =
      read_array<uint32_t>(imag_directory + "/group_offsets.bin", h.groups + 1);
  auto imag_group_row_ids =
      read_array<int32_t>(imag_directory + "/group_row_ids.bin",
                          static_cast<size_t>(h.groups) * kM);
  auto imag_tile_col_ids =
      read_array<int32_t>(imag_directory + "/tile_col_ids.bin",
                          static_cast<size_t>(h.tiles) * kK);
  auto imag_tile_a_bits =
      read_array<uint16_t>(imag_directory + "/tile_a_comp.bin",
                           static_cast<size_t>(h.tiles) * kM * kKComp);
  auto imag_tile_pair_meta =
      read_array<uint32_t>(imag_directory + "/tile_pair_meta.bin",
                           static_cast<size_t>(h.tiles) * kM);
  auto imag_dense_bits =
      read_array<uint16_t>(imag_directory + "/dense_input.bin",
                           static_cast<size_t>(h.n_cols) * kN);
  auto imag_reference =
      read_array<float>(imag_directory + "/reference.bin",
                        static_cast<size_t>(h.output_size) * kN);
  if (group_offsets != imag_group_offsets ||
      group_row_ids != imag_group_row_ids ||
      tile_col_ids != imag_tile_col_ids || dense_bits != imag_dense_bits) {
    std::fprintf(stderr, "real/imag structural data do not match\n");
    return 2;
  }
  for (uint32_t row = 0; row < h.output_size; ++row) {
    const size_t base = static_cast<size_t>(row) * kN;
    for (int channel = 0; channel < 8; ++channel) {
      const float real_output =
          reference[base + channel] - imag_reference[base + channel + 8];
      const float imag_output =
          reference[base + channel + 8] + imag_reference[base + channel];
      reference[base + channel] = real_output;
      reference[base + channel + 8] = imag_output;
    }
  }
#endif

  DeviceData d;
  d.group_offsets = device_copy(group_offsets);
  d.group_row_ids = device_copy(group_row_ids);
  d.tile_col_ids = device_copy(tile_col_ids);
  d.tile_a_comp = reinterpret_cast<half*>(device_copy(tile_a_bits));
  d.tile_pair_meta = device_copy(tile_pair_meta);
#if JKF_VARIANT >= 5 && JKF_VARIANT <= 9
  d.tile_a_imag_comp = reinterpret_cast<half*>(device_copy(imag_tile_a_bits));
  d.tile_imag_pair_meta = device_copy(imag_tile_pair_meta);
#endif
  d.csr_row_ptr = device_copy(csr_row_ptr);
  d.csr_row_ids = device_copy(csr_row_ids);
  d.csr_cols = device_copy(csr_cols);
  d.csr_vals = reinterpret_cast<half*>(device_copy(csr_val_bits));
  d.residual_row_ptr = device_copy(residual_row_ptr);
  d.residual_cols = device_copy(residual_cols);
  d.residual_vals = reinterpret_cast<half*>(device_copy(residual_val_bits));
  d.dense = reinterpret_cast<half*>(device_copy(dense_bits));
  CUDA_CHECK(cudaMalloc(&d.output,
                        static_cast<size_t>(h.output_size) * kN * sizeof(float)));
  CUDA_CHECK(cudaMemset(d.output, 0,
                        static_cast<size_t>(h.output_size) * kN * sizeof(float)));
#if JKF_VARIANT == 6
  CUDA_CHECK(cudaMalloc(&d.real_output,
                        static_cast<size_t>(h.output_size) * kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.imag_output,
                        static_cast<size_t>(h.output_size) * kN * sizeof(float)));
#endif

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  launch_variant(h, d, stream);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<float> output(static_cast<size_t>(h.output_size) * kN);
  CUDA_CHECK(cudaMemcpy(output.data(), d.output, output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double error_square = 0.0;
  double ref_square = 0.0;
  float max_abs = 0.0f;
  size_t nonfinite = 0;
  for (size_t i = 0; i < output.size(); ++i) {
    if (!std::isfinite(output[i])) ++nonfinite;
    const float diff = std::fabs(output[i] - reference[i]);
    max_abs = std::max(max_abs, diff);
    error_square += static_cast<double>(diff) * diff;
    ref_square += static_cast<double>(reference[i]) * reference[i];
  }
  const double rel_l2 = std::sqrt(error_square / std::max(ref_square, 1e-30));
  const bool correct = nonfinite == 0 && max_abs <= 2e-2f && rel_l2 <= 2e-3;
  if (!correct) {
    std::printf(
        "{\"variant\":\"%s\",\"correct\":false,\"max_abs\":%.9g,"
        "\"rel_l2\":%.9g,\"nonfinite\":%zu}\n",
        variant_name(), max_abs, rel_l2, nonfinite);
    return 3;
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  if (graph_mode) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_variant(h, d, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  }
  const auto launch_once = [&]() {
    if (graph_mode) {
      CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    } else {
      launch_variant(h, d, stream);
    }
  };

  for (int i = 0; i < warmup; ++i) launch_once();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) launch_once();
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  for (int i = 0; i < stress; ++i) launch_once();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::printf(
      "{\"variant\":\"%s\",\"correct\":true,\"max_abs\":%.9g,"
      "\"rel_l2\":%.9g,\"nonfinite\":0,\"us\":%.9g,"
      "\"warmup\":%d,\"iters\":%d,\"stress\":%d,"
      "\"groups\":%u,\"tiles\":%u,\"active_rows\":%u,"
      "\"nnz\":%u,\"graph\":%s}\n",
      variant_name(), max_abs, rel_l2,
      static_cast<double>(milliseconds) * 1000.0 / iterations, warmup,
      iterations, stress, h.groups, h.tiles, h.active_rows, h.nnz,
      graph_mode ? "true" : "false");
  std::printf(
      "ALPHA_OPS_RESULT {\"status\":\"pass\","
      "\"correctness_status\":\"pass\",\"variant\":\"%s\","
      "\"alpha_latency_ms\":%.12g,\"max_abs\":%.9g,"
      "\"rel_l2\":%.9g,\"graph\":%s}\n",
      variant_name(), static_cast<double>(milliseconds) / iterations, max_abs,
      rel_l2, graph_mode ? "true" : "false");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (graph_exec) cudaGraphExecDestroy(graph_exec);
  if (graph) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(d.group_offsets);
  cudaFree(d.group_row_ids);
  cudaFree(d.tile_col_ids);
  cudaFree(d.tile_a_comp);
  cudaFree(d.tile_pair_meta);
#if JKF_VARIANT >= 5 && JKF_VARIANT <= 9
  cudaFree(d.tile_a_imag_comp);
  cudaFree(d.tile_imag_pair_meta);
#endif
  cudaFree(d.csr_row_ptr);
  cudaFree(d.csr_row_ids);
  cudaFree(d.csr_cols);
  cudaFree(d.csr_vals);
  cudaFree(d.residual_row_ptr);
  cudaFree(d.residual_cols);
  cudaFree(d.residual_vals);
  cudaFree(d.dense);
  cudaFree(d.output);
#if JKF_VARIANT == 6
  cudaFree(d.real_output);
  cudaFree(d.imag_output);
#endif
  return 0;
}
#endif
