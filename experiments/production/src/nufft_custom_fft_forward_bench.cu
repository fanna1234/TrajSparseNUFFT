// Shape-specialized 8-batch 512x512 forward FFT with an N16 FP16 endpoint.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef JKF_CUSTOM_FFT_CANDIDATE
#define JKF_CUSTOM_FFT_CANDIDATE 0
#endif

#ifndef JKF_CUSTOM_FFT_NO_MAIN
#define JKF_CUSTOM_FFT_NO_MAIN 0
#endif

#ifndef JKF_CUSTOM_FFT_NO_CUFFT_RUNTIME
#define JKF_CUSTOM_FFT_NO_CUFFT_RUNTIME 0
#endif

#if JKF_CUSTOM_FFT_NO_CUFFT_RUNTIME && !JKF_CUSTOM_FFT_NO_MAIN
#error "no-cuFFT runtime mode is only supported by the exported API object"
#endif

#ifndef JKF_CUSTOM_FFT_FUSED_COLUMNS
#define JKF_CUSTOM_FFT_FUSED_COLUMNS 0
#endif

#ifndef JKF_CUSTOM_FFT_ROWS8
#define JKF_CUSTOM_FFT_ROWS8 0
#endif

#ifndef JKF_CUSTOM_FFT_ROWS8_DIRECT_TRANSPOSED
#define JKF_CUSTOM_FFT_ROWS8_DIRECT_TRANSPOSED 0
#endif

#ifndef JKF_CUSTOM_FFT_ROWS8_INTERLEAVED_TRANSPOSED
#define JKF_CUSTOM_FFT_ROWS8_INTERLEAVED_TRANSPOSED 0
#endif

#ifndef JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE
#define JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE 0
#endif

#ifndef JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_FAST
#define JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_FAST 0
#endif

#ifndef JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_ADD
#define JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_ADD 0
#endif

#ifndef JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_RAW_XOR
#define JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_RAW_XOR 0
#endif

#ifndef JKF_CUSTOM_FFT_ROWS8_SKIP_ZERO_ROWS
#define JKF_CUSTOM_FFT_ROWS8_SKIP_ZERO_ROWS 0
#endif

#ifndef JKF_CUSTOM_FFT_ROWS8_GRID2D
#define JKF_CUSTOM_FFT_ROWS8_GRID2D 0
#endif

#ifndef JKF_CUSTOM_FFT_SHARED16_SKIP_ZERO_ROWS
#define JKF_CUSTOM_FFT_SHARED16_SKIP_ZERO_ROWS 0
#endif

#ifndef JKF_CUSTOM_FFT_SHARED16_ZERO_ROWS_FLAT
#define JKF_CUSTOM_FFT_SHARED16_ZERO_ROWS_FLAT 0
#endif

#ifndef JKF_CUSTOM_FFT_VECTOR32
#define JKF_CUSTOM_FFT_VECTOR32 0
#endif

#ifndef JKF_CUSTOM_FFT_MIXED32
#define JKF_CUSTOM_FFT_MIXED32 0
#endif

#ifndef JKF_CUSTOM_FFT_SPECIAL16
#define JKF_CUSTOM_FFT_SPECIAL16 0
#endif

#ifndef JKF_CUSTOM_FFT_SHARED16
#define JKF_CUSTOM_FFT_SHARED16 0
#endif

#ifndef JKF_CUSTOM_FFT_Y_SHARED16
#define JKF_CUSTOM_FFT_Y_SHARED16 0
#endif

#ifndef JKF_CUSTOM_FFT_X_TRANSPOSED
#define JKF_CUSTOM_FFT_X_TRANSPOSED 0
#endif

#ifndef JKF_CUSTOM_FFT_X_DIRECT_TRANSPOSED
#define JKF_CUSTOM_FFT_X_DIRECT_TRANSPOSED 0
#endif

#ifndef JKF_CUSTOM_FFT_PLANAR_SHARED
#define JKF_CUSTOM_FFT_PLANAR_SHARED 0
#endif

#ifndef JKF_CUSTOM_FFT_XOR_SWIZZLE
#define JKF_CUSTOM_FFT_XOR_SWIZZLE 0
#endif

#ifndef JKF_CUSTOM_FFT_ALIGNED_PAIR_SWIZZLE
#define JKF_CUSTOM_FFT_ALIGNED_PAIR_SWIZZLE 0
#endif

#ifndef JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED
#define JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED 0
#endif

#ifndef JKF_CUSTOM_FFT_PHASED_SCALAR_LOAD
#define JKF_CUSTOM_FFT_PHASED_SCALAR_LOAD 0
#endif

#ifndef JKF_CUSTOM_FFT_PHASED_VOLATILE_SCALAR
#define JKF_CUSTOM_FFT_PHASED_VOLATILE_SCALAR 0
#endif

#ifndef JKF_CUSTOM_FFT_Y_N16_FUSED
#define JKF_CUSTOM_FFT_Y_N16_FUSED 0
#endif

#ifndef JKF_CUSTOM_FFT_FUSED_WRITER_ROLLED
#define JKF_CUSTOM_FFT_FUSED_WRITER_ROLLED 0
#endif

#ifndef JKF_CUSTOM_FFT_FUSED_PHASE_ROLLED
#define JKF_CUSTOM_FFT_FUSED_PHASE_ROLLED 0
#endif

#ifndef JKF_CUSTOM_FFT_Y_N16_FUSED_KX1
#define JKF_CUSTOM_FFT_Y_N16_FUSED_KX1 0
#endif

#ifndef JKF_CUSTOM_FFT_FUSED_ALL_KY_STAGE
#define JKF_CUSTOM_FFT_FUSED_ALL_KY_STAGE 0
#endif

#ifndef JKF_CUSTOM_FFT_FUSED_PREPACKED_PANEL
#define JKF_CUSTOM_FFT_FUSED_PREPACKED_PANEL 0
#endif

#ifndef JKF_CUSTOM_FFT_FUSED_PAIR64_WRITER
#define JKF_CUSTOM_FFT_FUSED_PAIR64_WRITER 0
#endif

#ifndef JKF_CUSTOM_FFT_FUSED_WRITER_UNROLL
#define JKF_CUSTOM_FFT_FUSED_WRITER_UNROLL 1
#endif

#ifndef JKF_CUSTOM_FFT_PRUNED_X_HALF_ZERO
#define JKF_CUSTOM_FFT_PRUNED_X_HALF_ZERO 0
#endif

#ifndef JKF_CUSTOM_IFFT_F31
#define JKF_CUSTOM_IFFT_F31 0
#endif

#ifndef JKF_CUSTOM_IFFT_REGULAR_X
#define JKF_CUSTOM_IFFT_REGULAR_X 0
#endif

#ifndef JKF_CUSTOM_IFFT_WARP_LOCAL_CROP
#define JKF_CUSTOM_IFFT_WARP_LOCAL_CROP 0
#endif

#ifndef JKF_CUSTOM_IFFT_DIRECT_REGULAR_LOAD
#define JKF_CUSTOM_IFFT_DIRECT_REGULAR_LOAD 0
#endif

#ifndef JKF_CUSTOM_IFFT_REGULAR_Y_CROP
#define JKF_CUSTOM_IFFT_REGULAR_Y_CROP 0
#endif

#if JKF_CUSTOM_FFT_PLANAR_SHARED && JKF_CUSTOM_FFT_X_TRANSPOSED
#error "planar shared layout is not implemented with the F8 shared reformat"
#endif

#if JKF_CUSTOM_FFT_XOR_SWIZZLE && !JKF_CUSTOM_FFT_PLANAR_SHARED
#error "XOR swizzle requires the planar shared layout"
#endif


#if JKF_CUSTOM_FFT_ALIGNED_PAIR_SWIZZLE && !JKF_CUSTOM_FFT_PLANAR_SHARED
#error "aligned-pair swizzle requires the planar shared layout"
#endif

#if JKF_CUSTOM_FFT_XOR_SWIZZLE && JKF_CUSTOM_FFT_ALIGNED_PAIR_SWIZZLE
#error "select only one shared-memory swizzle"
#endif

#if JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED && \
    (!JKF_CUSTOM_FFT_PLANAR_SHARED || !JKF_CUSTOM_FFT_ALIGNED_PAIR_SWIZZLE)
#error "phased planar shared requires the aligned-pair planar swizzle"
#endif

#if JKF_CUSTOM_FFT_PHASED_SCALAR_LOAD && \
    !JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED
#error "phased scalar loads require phased planar shared"
#endif

#if JKF_CUSTOM_FFT_PHASED_VOLATILE_SCALAR && \
    !JKF_CUSTOM_FFT_PHASED_SCALAR_LOAD
#error "volatile scalar loads require phased scalar loads"
#endif

#if JKF_CUSTOM_FFT_Y_N16_FUSED && \
    (!JKF_CUSTOM_FFT_Y_SHARED16 || !JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED || \
     !JKF_CUSTOM_FFT_X_DIRECT_TRANSPOSED)
#error "fused N16 y requires phased shared16 and direct-transposed x"
#endif

#if JKF_CUSTOM_FFT_FUSED_WRITER_ROLLED && !JKF_CUSTOM_FFT_Y_N16_FUSED
#error "rolled fused writer requires fused N16 y"
#endif

#if JKF_CUSTOM_FFT_FUSED_PHASE_ROLLED && \
    !JKF_CUSTOM_FFT_FUSED_WRITER_ROLLED
#error "rolled fused phase requires the rolled fused writer"
#endif

#if JKF_CUSTOM_FFT_Y_N16_FUSED_KX1 && \
    (!JKF_CUSTOM_FFT_Y_N16_FUSED || !JKF_CUSTOM_FFT_FUSED_PHASE_ROLLED)
#error "one-kx fused y requires the rolled fused N16 path"
#endif

#if JKF_CUSTOM_IFFT_REGULAR_X && \
    (!JKF_CUSTOM_IFFT_F31 || !JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE)
#error "regular inverse-x requires F31 and regular shared swizzle"
#endif

#if JKF_CUSTOM_FFT_ROWS8_DIRECT_TRANSPOSED && !JKF_CUSTOM_FFT_ROWS8
#error "rows8 direct transpose requires the rows8 x kernel"
#endif

#if JKF_CUSTOM_FFT_ROWS8_INTERLEAVED_TRANSPOSED && \
    !JKF_CUSTOM_FFT_ROWS8_DIRECT_TRANSPOSED
#error "interleaved rows8 transpose requires direct-transpose mode"
#endif


#if JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE && \
    !JKF_CUSTOM_FFT_ROWS8_INTERLEAVED_TRANSPOSED
#error "regular shared swizzle requires interleaved rows8 mode"
#endif

#if JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_FAST && \
    !JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE
#error "fast regular swizzle requires regular shared swizzle"
#endif

#if JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_ADD && \
    !JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_FAST
#error "add-folded swizzle requires fast regular swizzle"
#endif

#if JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_RAW_XOR && \
    !JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_FAST
#error "raw-XOR swizzle requires fast regular swizzle"
#endif

#if JKF_CUSTOM_FFT_ROWS8_SKIP_ZERO_ROWS && \
    !JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_FAST
#error "zero-row elision requires the fast regular shared swizzle"
#endif

#if JKF_CUSTOM_FFT_ROWS8_GRID2D && !JKF_CUSTOM_FFT_ROWS8_SKIP_ZERO_ROWS
#error "rows8 2D grid requires zero-row elision"
#endif

#if JKF_CUSTOM_FFT_SHARED16_SKIP_ZERO_ROWS && !JKF_CUSTOM_FFT_SHARED16
#error "shared16 zero-row elision requires the shared16 x kernel"
#endif

#if JKF_CUSTOM_FFT_SHARED16_ZERO_ROWS_FLAT && \
    !JKF_CUSTOM_FFT_SHARED16_SKIP_ZERO_ROWS
#error "flat shared16 zero-row mapping requires zero-row elision"
#endif

namespace {

constexpr int kImage = 256;
constexpr int kGrid = 512;
constexpr int kCoils = 8;
constexpr int kChannels = 16;
constexpr int kFftThreads = 64;
constexpr float kSqrtHalf = 0.7071067811865475244f;
#if JKF_CUSTOM_FFT_XOR_SWIZZLE
constexpr int kSharedPlaneElements = 2 * 32 * 32;
#else
constexpr int kSharedFrequencyStride = 17;
constexpr int kSharedTransformStride = 32 * kSharedFrequencyStride + 16;
#if JKF_CUSTOM_FFT_PLANAR_SHARED
constexpr int kSharedPlaneElements = 4 * kSharedTransformStride;
#endif
#endif
#if JKF_CUSTOM_FFT_PLANAR_SHARED
#if JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED
constexpr size_t kShared16Bytes = kSharedPlaneElements * sizeof(float);
#else
constexpr size_t kShared16Bytes = 2 * kSharedPlaneElements * sizeof(float);
#endif
#else
constexpr size_t kShared16Bytes = 4 * 32 * 16 * sizeof(float2);
#endif

#if JKF_CUSTOM_FFT_Y_N16_FUSED
constexpr int kFusedYTransforms = kCoils * 4;
constexpr int kFusedYPlaneElements = kCoils * kSharedPlaneElements;
constexpr size_t kFusedYSharedBytes =
    kFusedYPlaneElements * sizeof(float);
constexpr int kFusedPanelWords = 256 * 4 * kCoils;
static_assert(kFusedYTransforms == 32);
static_assert(kFusedYSharedBytes >=
              static_cast<size_t>(kFusedPanelWords) * sizeof(uint32_t));

constexpr int kFusedKx1PlaneElements = kCoils * kSharedTransformStride;
constexpr size_t kFusedKx1SharedBytes =
    kFusedKx1PlaneElements * sizeof(float);
constexpr int kFusedKx1PanelWords = 256 * kCoils;
static_assert(kFusedKx1SharedBytes >=
              static_cast<size_t>(kFusedKx1PanelWords) * sizeof(uint32_t));
#if JKF_CUSTOM_FFT_FUSED_ALL_KY_STAGE
constexpr int kFusedKx1AllPanelWords = kGrid * kCoils;
static_assert(kFusedKx1SharedBytes >=
              static_cast<size_t>(kFusedKx1AllPanelWords) * sizeof(uint32_t));
#endif
#endif

__constant__ float2 kForwardTwiddle[kGrid];

#if JKF_CUSTOM_FFT_Y_N16_FUSED
__device__ __forceinline__ int fused_panel_word_index(int ky_local, int row,
                                                       int coil) {
  const int slot =
      (row & 1) |
      ((((coil >> 0) & 1) ^ ((ky_local >> 1) & 1)) << 1) |
      ((((coil >> 1) & 1) ^ ((ky_local >> 2) & 1)) << 2) |
      ((((coil >> 2) & 1) ^ ((ky_local >> 3) & 1)) << 3) |
      ((((row >> 1) & 1) ^ ((ky_local >> 4) & 1)) << 4);
  return ky_local * 32 + slot;
}


__device__ __forceinline__ int fused_kx1_panel_word_index(int ky_local,
                                                           int coil) {
  const int slot =
      ((coil >> 0) & 1) |
      (((ky_local >> 1) & 1) << 1) |
      ((((ky_local >> 2) & 1) ^ ((coil >> 1) & 1)) << 2) |
      ((((ky_local >> 3) & 1) ^ ((coil >> 2) & 1)) << 3) |
      ((((ky_local >> 4) & 1) ^ (ky_local & 1)) << 4);
  return ((ky_local >> 2) << 5) + slot;
}

#if JKF_CUSTOM_FFT_FUSED_PREPACKED_PANEL
__device__ __forceinline__ int fused_kx1_prepacked_word_index(
    int ky, int coil_pair, int imaginary) {
  const int slot =
      imaginary |
      (((((ky >> 4) & 1) ^ (ky & 1) ^ (coil_pair & 1))) << 1) |
      (((((ky >> 1) & 1) ^ ((coil_pair >> 1) & 1))) << 2) |
      (((((ky >> 2) & 1) ^ (coil_pair & 1))) << 3) |
      (((((ky >> 3) & 1) ^ ((coil_pair >> 1) & 1))) << 4);
  return ((ky >> 2) << 5) + slot;
}
#endif
#endif

#if JKF_CUSTOM_FFT_PLANAR_SHARED
__device__ __forceinline__ int planar_shared_index(int row, int frequency,
                                                    int lane) {
#if JKF_CUSTOM_FFT_XOR_SWIZZLE
  const int pair = row >> 1;
  const int half = row & 1;
  const int swizzled_lane = lane ^ (frequency >> 1);
  return (pair * 32 + frequency) * 32 + half * 16 + swizzled_lane;
#elif JKF_CUSTOM_FFT_ALIGNED_PAIR_SWIZZLE
  const int frequency_pair = frequency >> 1;
  const int frequency_in_pair = frequency & 1;
  const int frequency_base =
      frequency_pair * (2 * kSharedFrequencyStride) +
      frequency_in_pair * (kSharedFrequencyStride + 1);
  return row * kSharedTransformStride + frequency_base + lane;
#else
  return row * kSharedTransformStride + frequency * kSharedFrequencyStride +
         lane;
#endif
}


#if JKF_CUSTOM_FFT_PHASED_SCALAR_LOAD
template <bool kImaginary>
__device__ __forceinline__ void phased_scalar_load(
#if JKF_CUSTOM_FFT_PHASED_VOLATILE_SCALAR
    float2 (&values)[32], const volatile float* shared_plane, int row,
    int lane) {
#else
    float2 (&values)[32], const float* shared_plane, int row, int lane) {
#endif
  const int frequency0 = lane * 2;
  const int frequency1 = frequency0 + 1;
#pragma unroll
  for (int input_lane = 0; input_lane < 16; ++input_lane) {
    const int local_index = (input_lane & 3) * 4 + (input_lane >> 2);
    const int shared_index0 =
        planar_shared_index(row, frequency0, input_lane);
    const int shared_index1 =
        planar_shared_index(row, frequency1, input_lane);
    if constexpr (kImaginary) {
      values[local_index].y = shared_plane[shared_index0];
      values[16 + local_index].y = shared_plane[shared_index1];
    } else {
      values[local_index].x = shared_plane[shared_index0];
      values[16 + local_index].x = shared_plane[shared_index1];
    }
  }
}
#endif

__device__ __forceinline__ float2 complex_mul(float2 left, float2 right);
__device__ __forceinline__ uint32_t pack_half_pair(float low, float high);
__device__ __forceinline__ void fft32_mixed_radix(float2 (&values)[32]);
template <int kOffset>
__device__ __forceinline__ void fft16_mixed_radix(float2 (&values)[32]);

#if JKF_CUSTOM_FFT_Y_N16_FUSED_KX1
template <int kK1Base>
__device__ __forceinline__ void stage_fused_kx1_panel_phase(
    float2 (&values)[32], uint32_t* panel_shared, int coil, int frequency0,
    int frequency1, int kx, half* residual_output) {
#pragma unroll
  for (int local_k1 = 0; local_k1 < 8; ++local_k1) {
    constexpr int kPhase = kK1Base / 8;
    const int k1 = kK1Base + local_k1;
    const int ky0 = frequency0 + k1 * 32;
    const int ky1 = frequency1 + k1 * 32;
    const int ky_local0 = ky0 - kPhase * 256;
    const int ky_local1 = ky1 - kPhase * 256;
    const float2 value0 = values[k1];
    const float2 value1 = values[16 + k1];
    panel_shared[fused_kx1_panel_word_index(ky_local0, coil)] =
        pack_half_pair(value0.x, value0.y);
    panel_shared[fused_kx1_panel_word_index(ky_local1, coil)] =
        pack_half_pair(value1.x, value1.y);
    if (residual_output != nullptr) {
      const half value0_real_high = __float2half_rn(value0.x);
      const half value0_imag_high = __float2half_rn(value0.y);
      const half value1_real_high = __float2half_rn(value1.x);
      const half value1_imag_high = __float2half_rn(value1.y);
      const size_t output0 =
          (static_cast<size_t>(ky0) * kGrid + kx) * kChannels;
      const size_t output1 =
          (static_cast<size_t>(ky1) * kGrid + kx) * kChannels;
      residual_output[output0 + coil] = __float2half_rn(
          value0.x - __half2float(value0_real_high));
      residual_output[output0 + coil + kCoils] = __float2half_rn(
          value0.y - __half2float(value0_imag_high));
      residual_output[output1 + coil] = __float2half_rn(
          value1.x - __half2float(value1_real_high));
      residual_output[output1 + coil + kCoils] = __float2half_rn(
          value1.y - __half2float(value1_imag_high));
    }
  }
}

#if JKF_CUSTOM_FFT_FUSED_ALL_KY_STAGE
#if JKF_CUSTOM_FFT_FUSED_PREPACKED_PANEL
__device__ __forceinline__ void stage_fused_kx1_prepacked_all_ky(
    float2 (&values)[32], uint32_t* panel_shared, int coil, int frequency0,
    int frequency1) {
  const int coil_pair = coil >> 1;
#pragma unroll
  for (int k1 = 0; k1 < 16; ++k1) {
    const int ky0 = frequency0 + k1 * 32;
    const int ky1 = frequency1 + k1 * 32;
    const float2 value0 = values[k1];
    const float2 value1 = values[16 + k1];
    const uint32_t self0 = pack_half_pair(value0.x, value0.y);
    const uint32_t self1 = pack_half_pair(value1.x, value1.y);
    const uint32_t peer0 = __shfl_xor_sync(0xffffffffu, self0, 16);
    const uint32_t peer1 = __shfl_xor_sync(0xffffffffu, self1, 16);
    if ((coil & 1) == 0) {
      const uint32_t real0 =
          (self0 & 0xffffu) | ((peer0 & 0xffffu) << 16);
      const uint32_t imaginary0 =
          (self0 >> 16) | (peer0 & 0xffff0000u);
      const uint32_t real1 =
          (self1 & 0xffffu) | ((peer1 & 0xffffu) << 16);
      const uint32_t imaginary1 =
          (self1 >> 16) | (peer1 & 0xffff0000u);
      const uint64_t packed0 =
          static_cast<uint64_t>(real0) |
          (static_cast<uint64_t>(imaginary0) << 32);
      const uint64_t packed1 =
          static_cast<uint64_t>(real1) |
          (static_cast<uint64_t>(imaginary1) << 32);
      *reinterpret_cast<uint64_t*>(
          panel_shared +
          fused_kx1_prepacked_word_index(ky0, coil_pair, 0)) = packed0;
      *reinterpret_cast<uint64_t*>(
          panel_shared +
          fused_kx1_prepacked_word_index(ky1, coil_pair, 0)) = packed1;
    }
  }
}
#else
__device__ __forceinline__ void stage_fused_kx1_all_ky(
    float2 (&values)[32], uint32_t* panel_shared, int coil, int frequency0,
    int frequency1) {
#pragma unroll
  for (int k1 = 0; k1 < 16; ++k1) {
    const int ky0 = frequency0 + k1 * 32;
    const int ky1 = frequency1 + k1 * 32;
    const float2 value0 = values[k1];
    const float2 value1 = values[16 + k1];
    panel_shared[fused_kx1_panel_word_index(ky0, coil)] =
        pack_half_pair(value0.x, value0.y);
    panel_shared[fused_kx1_panel_word_index(ky1, coil)] =
        pack_half_pair(value1.x, value1.y);
  }
}
#endif
#endif

__global__ void fft512_allcoils1_shared16_to_n16(
    const float2* __restrict__ input, half* __restrict__ output,
    half* __restrict__ residual_output,
    float2* __restrict__ fp32_output) {
  extern __shared__ float shared_scalar[];
  const int lane = threadIdx.x;
  const int coil = threadIdx.z;
  const int kx = blockIdx.x;
  const int transform = coil * kGrid + kx;
  float* coil_shared = shared_scalar + coil * kSharedTransformStride;
  float2 values[32];

#pragma unroll
  for (int q = 0; q < 32; ++q) {
    const int y = lane + q * 16;
    values[(q & 3) * 8 + (q >> 2)] =
        input[static_cast<size_t>(transform) * kGrid + y];
  }
  fft32_mixed_radix(values);
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    values[k2] = complex_mul(
        values[k2], kForwardTwiddle[(lane * k2) & (kGrid - 1)]);
  }

#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    const int shared_index = planar_shared_index(0, k2, lane);
    coil_shared[shared_index] = values[k2].x;
  }
  __syncwarp();
  const int frequency0 = lane * 2;
  const int frequency1 = frequency0 + 1;
#pragma unroll
  for (int input_pair = 0; input_pair < 8; ++input_pair) {
    const int input_lane0 = input_pair * 2;
    const int input_lane1 = input_lane0 + 1;
    const int local_index0 = (input_lane0 & 3) * 4 + (input_lane0 >> 2);
    const int local_index1 = (input_lane1 & 3) * 4 + (input_lane1 >> 2);
    const int shared_index0 = planar_shared_index(0, frequency0, input_lane0);
    const int shared_index1 = planar_shared_index(0, frequency1, input_lane0);
    const float2 real0 =
        *reinterpret_cast<const float2*>(coil_shared + shared_index0);
    const float2 real1 =
        *reinterpret_cast<const float2*>(coil_shared + shared_index1);
    values[local_index0].x = real0.x;
    values[local_index1].x = real0.y;
    values[16 + local_index0].x = real1.x;
    values[16 + local_index1].x = real1.y;
  }
  __syncwarp();
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    const int shared_index = planar_shared_index(0, k2, lane);
    coil_shared[shared_index] = values[k2].y;
  }
  __syncwarp();
#pragma unroll
  for (int input_pair = 0; input_pair < 8; ++input_pair) {
    const int input_lane0 = input_pair * 2;
    const int input_lane1 = input_lane0 + 1;
    const int local_index0 = (input_lane0 & 3) * 4 + (input_lane0 >> 2);
    const int local_index1 = (input_lane1 & 3) * 4 + (input_lane1 >> 2);
    const int shared_index0 = planar_shared_index(0, frequency0, input_lane0);
    const int shared_index1 = planar_shared_index(0, frequency1, input_lane0);
    const float2 imag0 =
        *reinterpret_cast<const float2*>(coil_shared + shared_index0);
    const float2 imag1 =
        *reinterpret_cast<const float2*>(coil_shared + shared_index1);
    values[local_index0].y = imag0.x;
    values[local_index1].y = imag0.y;
    values[16 + local_index0].y = imag1.x;
    values[16 + local_index1].y = imag1.y;
  }
  fft16_mixed_radix<0>(values);
  fft16_mixed_radix<16>(values);

  if (fp32_output != nullptr) {
#pragma unroll
    for (int k1 = 0; k1 < 16; ++k1) {
      const int ky0 = frequency0 + k1 * 32;
      const int ky1 = frequency1 + k1 * 32;
      fp32_output[(static_cast<size_t>(coil) * kGrid + ky0) * kGrid + kx] =
          values[k1];
      fp32_output[(static_cast<size_t>(coil) * kGrid + ky1) * kGrid + kx] =
          values[16 + k1];
    }
  }

  __syncthreads();
  uint32_t* panel_shared = reinterpret_cast<uint32_t*>(shared_scalar);
#if JKF_CUSTOM_FFT_FUSED_ALL_KY_STAGE
#if JKF_CUSTOM_FFT_FUSED_PREPACKED_PANEL
  stage_fused_kx1_prepacked_all_ky(values, panel_shared, coil, frequency0,
                                   frequency1);
#else
  stage_fused_kx1_all_ky(values, panel_shared, coil, frequency0, frequency1);
#endif
  __syncthreads();

  uint32_t* output_words = reinterpret_cast<uint32_t*>(output);
  const int linear_thread = lane + 16 * coil;
  const int warp_id = linear_thread >> 5;
  const int warp_lane = linear_thread & 31;
  const int panel_group = warp_lane >> 3;
  const int panel_coil = warp_lane & 7;
#if !JKF_CUSTOM_FFT_FUSED_PREPACKED_PANEL && \
    !JKF_CUSTOM_FFT_FUSED_PAIR64_WRITER
  const int pair = panel_coil & 3;
  const int group_lane_base = panel_group * 8;
#elif JKF_CUSTOM_FFT_FUSED_PAIR64_WRITER
  const int group_lane_base = panel_group * 8;
#endif

#if JKF_CUSTOM_FFT_FUSED_WRITER_UNROLL == 2
#pragma unroll 2
#elif JKF_CUSTOM_FFT_FUSED_WRITER_UNROLL == 4
#pragma unroll 4
#elif JKF_CUSTOM_FFT_FUSED_WRITER_UNROLL == 8
#pragma unroll 8
#else
#pragma unroll 1
#endif
  for (int ky_iteration = 0; ky_iteration < 32; ++ky_iteration) {
    const int ky = warp_id * 4 + panel_group + ky_iteration * 16;
#if JKF_CUSTOM_FFT_FUSED_PREPACKED_PANEL
    const int coil_pair = panel_coil & 3;
    const int imaginary = panel_coil >> 2;
    const uint32_t packed = panel_shared[
        fused_kx1_prepacked_word_index(ky, coil_pair, imaginary)];
#elif JKF_CUSTOM_FFT_FUSED_PAIR64_WRITER
    uint32_t real_pair = 0;
    uint32_t imaginary_pair = 0;
    if (panel_coil < 4) {
      const uint64_t complex_pair = *reinterpret_cast<const uint64_t*>(
          panel_shared +
          fused_kx1_panel_word_index(ky, panel_coil * 2));
      const uint32_t first = static_cast<uint32_t>(complex_pair);
      const uint32_t second = static_cast<uint32_t>(complex_pair >> 32);
      real_pair =
          (first & 0xffffu) | ((second & 0xffffu) << 16);
      imaginary_pair = (first >> 16) | (second & 0xffff0000u);
    }
    const uint32_t shuffled_imaginary = __shfl_sync(
        0xffffffffu, imaginary_pair,
        group_lane_base + (panel_coil & 3));
    const uint32_t packed =
        panel_coil < 4 ? real_pair : shuffled_imaginary;
#else
    const uint32_t complex_word =
        panel_shared[fused_kx1_panel_word_index(ky, panel_coil)];
    const uint32_t first = __shfl_sync(
        0xffffffffu, complex_word, group_lane_base + pair * 2);
    const uint32_t second = __shfl_sync(
        0xffffffffu, complex_word, group_lane_base + pair * 2 + 1);
    const uint32_t packed = panel_coil < 4
                                ? ((first & 0xffffu) |
                                   ((second & 0xffffu) << 16))
                                : ((first >> 16) |
                                   (second & 0xffff0000u));
#endif
    const size_t output_word_base =
        (static_cast<size_t>(ky) * kGrid + kx) * 8;
    output_words[output_word_base + panel_coil] = packed;
  }
#else
  uint32_t* output_words = reinterpret_cast<uint32_t*>(output);
  const int linear_thread = lane + 16 * coil;
  const int warp_id = linear_thread >> 5;
  const int warp_lane = linear_thread & 31;
  const int panel_group = warp_lane >> 3;
  const int panel_coil = warp_lane & 7;
  const int pair = panel_coil & 3;
  const int group_lane_base = panel_group * 8;

#pragma unroll 1
  for (int phase = 0; phase < 2; ++phase) {
    if (phase == 0) {
      stage_fused_kx1_panel_phase<0>(values, panel_shared, coil, frequency0,
                                     frequency1, kx, residual_output);
    } else {
      stage_fused_kx1_panel_phase<8>(values, panel_shared, coil, frequency0,
                                     frequency1, kx, residual_output);
    }
    __syncthreads();

#pragma unroll 1
    for (int ky_iteration = 0; ky_iteration < 16; ++ky_iteration) {
      const int ky_local =
          warp_id * 4 + panel_group + ky_iteration * 16;
      const uint32_t complex_word =
          panel_shared[fused_kx1_panel_word_index(ky_local, panel_coil)];
      const uint32_t first = __shfl_sync(
          0xffffffffu, complex_word, group_lane_base + pair * 2);
      const uint32_t second = __shfl_sync(
          0xffffffffu, complex_word, group_lane_base + pair * 2 + 1);
      const uint32_t packed = panel_coil < 4
                                  ? ((first & 0xffffu) |
                                     ((second & 0xffffu) << 16))
                                  : ((first >> 16) |
                                     (second & 0xffff0000u));
      const int ky = phase * 256 + ky_local;
      const size_t output_word_base =
          (static_cast<size_t>(ky) * kGrid + kx) * 8;
      output_words[output_word_base + panel_coil] = packed;
    }
    if (phase == 0) __syncthreads();
  }
#endif
}
#endif
#endif

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t status_ = (expr);                                               \
    if (status_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status_));                                \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CUFFT_CHECK(expr)                                                       \
  do {                                                                          \
    cufftResult status_ = (expr);                                               \
    if (status_ != CUFFT_SUCCESS) {                                             \
      std::fprintf(stderr, "cuFFT failure %s:%d: %d\n", __FILE__, __LINE__,  \
                   static_cast<int>(status_));                                  \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

__device__ __forceinline__ float2 complex_add(float2 left, float2 right) {
  return make_float2(left.x + right.x, left.y + right.y);
}

__device__ __forceinline__ float2 complex_sub(float2 left, float2 right) {
  return make_float2(left.x - right.x, left.y - right.y);
}

__device__ __forceinline__ float2 complex_mul(float2 left, float2 right) {
  return make_float2(left.x * right.x - left.y * right.y,
                     left.x * right.y + left.y * right.x);
}

__device__ __forceinline__ uint32_t pack_half_pair(float low, float high) {
  const half low_half = __float2half_rn(low);
  const half high_half = __float2half_rn(high);
  return static_cast<uint32_t>(__half_as_ushort(low_half)) |
         (static_cast<uint32_t>(__half_as_ushort(high_half)) << 16);
}

template <bool kInverse>
__device__ __forceinline__ void dft4(const float2& x0,
                                     const float2& x1,
                                     const float2& x2,
                                     const float2& x3,
                                     float2 (&output)[4]) {
  const float2 sum02 = complex_add(x0, x2);
  const float2 diff02 = complex_sub(x0, x2);
  const float2 sum13 = complex_add(x1, x3);
  const float2 diff13 = complex_sub(x1, x3);
  output[0] = complex_add(sum02, sum13);
  output[2] = complex_sub(sum02, sum13);
  float2 rotated;
  if constexpr (kInverse) {
    rotated = make_float2(-diff13.y, diff13.x);
  } else {
    rotated = make_float2(diff13.y, -diff13.x);
  }
  output[1] = complex_add(diff02, rotated);
  output[3] = complex_sub(diff02, rotated);
}

template <bool kInverse>
__device__ __forceinline__ float2 multiply_w8(float2 value, int index) {
  if (index == 0) return value;
  if (index == 1) {
    if constexpr (kInverse) {
      return make_float2(kSqrtHalf * (value.x - value.y),
                         kSqrtHalf * (value.x + value.y));
    }
    return make_float2(kSqrtHalf * (value.x + value.y),
                       kSqrtHalf * (value.y - value.x));
  }
  if (index == 2) {
    if constexpr (kInverse) return make_float2(-value.y, value.x);
    return make_float2(value.y, -value.x);
  }
  if constexpr (kInverse) {
    return make_float2(-kSqrtHalf * (value.x + value.y),
                       kSqrtHalf * (value.x - value.y));
  }
  return make_float2(kSqrtHalf * (value.y - value.x),
                     -kSqrtHalf * (value.x + value.y));
}

template <bool kInverse>
__device__ __forceinline__ void dft8(const float2 (&input)[8],
                                     float2 (&output)[8]) {
  float2 even[4];
  float2 odd[4];
  dft4<kInverse>(input[0], input[2], input[4], input[6], even);
  dft4<kInverse>(input[1], input[3], input[5], input[7], odd);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const float2 rotated = multiply_w8<kInverse>(odd[index], index);
    output[index] = complex_add(even[index], rotated);
    output[index + 4] = complex_sub(even[index], rotated);
  }
}

template <bool kInverse>
__device__ __forceinline__ void dft4_first2_zero(
    const float2& x0, const float2& x1, float2 (&output)[4]) {
  output[0] = complex_add(x0, x1);
  output[2] = complex_sub(x0, x1);
  const float2 rotated = kInverse ? make_float2(-x1.y, x1.x)
                                  : make_float2(x1.y, -x1.x);
  output[1] = complex_add(x0, rotated);
  output[3] = complex_sub(x0, rotated);
}

template <bool kInverse>
__device__ __forceinline__ void dft8_first4_zero(
    const float2& x0, const float2& x1, const float2& x2, const float2& x3,
    float2 (&output)[8]) {
  float2 even[4];
  float2 odd[4];
  dft4_first2_zero<kInverse>(x0, x2, even);
  dft4_first2_zero<kInverse>(x1, x3, odd);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const float2 rotated = multiply_w8<kInverse>(odd[index], index);
    output[index] = complex_add(even[index], rotated);
    output[index + 4] = complex_sub(even[index], rotated);
  }
}

template <bool kInverse>
__device__ __forceinline__ float2 stage_twiddle(float2 value,
                                                int q,
                                                int j,
                                                int step) {
  const int index = (q * j * step) & (kGrid - 1);
  float2 twiddle = kForwardTwiddle[index];
  if constexpr (kInverse) twiddle.y = -twiddle.y;
  return complex_mul(value, twiddle);
}

__device__ __forceinline__ int reverse_low_bits(int value, int bits) {
  return __brev(static_cast<unsigned int>(value)) >> (32 - bits);
}

template <int kLength>
__device__ __forceinline__ void fft32_stage(float2 (&values)[32]) {
  constexpr int kHalf = kLength / 2;
#pragma unroll
  for (int block = 0; block < 32; block += kLength) {
#pragma unroll
    for (int j = 0; j < kHalf; ++j) {
      const float2 left = values[block + j];
      const float2 right = values[block + j + kHalf];
      const float2 twiddle = kForwardTwiddle[j * (kGrid / kLength)];
      const float2 product = complex_mul(right, twiddle);
      values[block + j] = complex_add(left, product);
      values[block + j + kHalf] = complex_sub(left, product);
    }
  }
}

__device__ __forceinline__ void fft32_register(float2 (&values)[32]) {
  fft32_stage<2>(values);
  fft32_stage<4>(values);
  fft32_stage<8>(values);
  fft32_stage<16>(values);
  fft32_stage<32>(values);
}

__device__ __forceinline__ void fft32_mixed_radix(float2 (&values)[32]) {
#pragma unroll
  for (int n1 = 0; n1 < 4; ++n1) {
    float2 input[8];
    float2 output[8];
#pragma unroll
    for (int n2 = 0; n2 < 8; ++n2) input[n2] = values[n1 * 8 + n2];
    dft8<false>(input, output);
#pragma unroll
    for (int k2 = 0; k2 < 8; ++k2) values[n1 * 8 + k2] = output[k2];
  }
#pragma unroll
  for (int k2 = 0; k2 < 8; ++k2) {
    float2 input[4];
    float2 output[4];
#pragma unroll
    for (int n1 = 0; n1 < 4; ++n1) {
      input[n1] = complex_mul(
          values[n1 * 8 + k2], kForwardTwiddle[n1 * k2 * 16]);
    }
    dft4<false>(input[0], input[1], input[2], input[3], output);
#pragma unroll
    for (int k1 = 0; k1 < 4; ++k1) values[k1 * 8 + k2] = output[k1];
  }
}

__device__ __forceinline__ void fft32_mixed_radix_half_zero(
    float2 (&values)[32]) {
#pragma unroll
  for (int n1 = 0; n1 < 4; ++n1) {
    float2 output[8];
    dft8_first4_zero<false>(values[n1 * 8], values[n1 * 8 + 1],
                            values[n1 * 8 + 2], values[n1 * 8 + 3], output);
#pragma unroll
    for (int k2 = 0; k2 < 8; ++k2) values[n1 * 8 + k2] = output[k2];
  }
#pragma unroll
  for (int k2 = 0; k2 < 8; ++k2) {
    float2 input[4];
    float2 output[4];
#pragma unroll
    for (int n1 = 0; n1 < 4; ++n1) {
      input[n1] = complex_mul(
          values[n1 * 8 + k2], kForwardTwiddle[n1 * k2 * 16]);
    }
    dft4<false>(input[0], input[1], input[2], input[3], output);
#pragma unroll
    for (int k1 = 0; k1 < 4; ++k1) values[k1 * 8 + k2] = output[k1];
  }
}

template <int kOffset>
__device__ __forceinline__ void fft16_mixed_radix(float2 (&values)[32]) {
#pragma unroll
  for (int n1 = 0; n1 < 4; ++n1) {
    float2 output[4];
    dft4<false>(values[kOffset + n1 * 4],
                values[kOffset + n1 * 4 + 1],
                values[kOffset + n1 * 4 + 2],
                values[kOffset + n1 * 4 + 3], output);
#pragma unroll
    for (int k2 = 0; k2 < 4; ++k2) {
      values[kOffset + n1 * 4 + k2] = output[k2];
    }
  }
#pragma unroll
  for (int k2 = 0; k2 < 4; ++k2) {
    float2 input[4];
    float2 output[4];
#pragma unroll
    for (int n1 = 0; n1 < 4; ++n1) {
      input[n1] = complex_mul(
          values[kOffset + n1 * 4 + k2],
          kForwardTwiddle[n1 * k2 * 32]);
    }
    dft4<false>(input[0], input[1], input[2], input[3], output);
#pragma unroll
    for (int k1 = 0; k1 < 4; ++k1) {
      values[kOffset + k1 * 4 + k2] = output[k1];
    }
  }
}

#if JKF_CUSTOM_IFFT_F31
__device__ __forceinline__ float2 inverse_twiddle(int index) {
  const float2 forward = kForwardTwiddle[index & (kGrid - 1)];
  return make_float2(forward.x, -forward.y);
}

__device__ __forceinline__ void fft32_mixed_radix_inverse(
    float2 (&values)[32]) {
#pragma unroll
  for (int n1 = 0; n1 < 4; ++n1) {
    float2 input[8];
    float2 output[8];
#pragma unroll
    for (int n2 = 0; n2 < 8; ++n2) input[n2] = values[n1 * 8 + n2];
    dft8<true>(input, output);
#pragma unroll
    for (int k2 = 0; k2 < 8; ++k2) values[n1 * 8 + k2] = output[k2];
  }
#pragma unroll
  for (int k2 = 0; k2 < 8; ++k2) {
    float2 input[4];
    float2 output[4];
#pragma unroll
    for (int n1 = 0; n1 < 4; ++n1) {
      input[n1] = complex_mul(values[n1 * 8 + k2],
                              inverse_twiddle(n1 * k2 * 16));
    }
    dft4<true>(input[0], input[1], input[2], input[3], output);
#pragma unroll
    for (int k1 = 0; k1 < 4; ++k1) values[k1 * 8 + k2] = output[k1];
  }
}

template <int kOffset>
__device__ __forceinline__ void fft16_mixed_radix_inverse(
    float2 (&values)[32]) {
#pragma unroll
  for (int n1 = 0; n1 < 4; ++n1) {
    float2 output[4];
    dft4<true>(values[kOffset + n1 * 4],
               values[kOffset + n1 * 4 + 1],
               values[kOffset + n1 * 4 + 2],
               values[kOffset + n1 * 4 + 3], output);
#pragma unroll
    for (int k2 = 0; k2 < 4; ++k2) {
      values[kOffset + n1 * 4 + k2] = output[k2];
    }
  }
#pragma unroll
  for (int k2 = 0; k2 < 4; ++k2) {
    float2 input[4];
    float2 output[4];
#pragma unroll
    for (int n1 = 0; n1 < 4; ++n1) {
      input[n1] = complex_mul(values[kOffset + n1 * 4 + k2],
                              inverse_twiddle(n1 * k2 * 32));
    }
    dft4<true>(input[0], input[1], input[2], input[3], output);
#pragma unroll
    for (int k1 = 0; k1 < 4; ++k1) {
      values[kOffset + k1 * 4 + k2] = output[k1];
    }
  }
}

__device__ __forceinline__ void fft512_inverse_phased(
    float2 (&values)[32], float* shared_plane, int row, int lane) {
  fft32_mixed_radix_inverse(values);
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    values[k2] = complex_mul(
        values[k2], inverse_twiddle(lane * k2));
  }

#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    const int shared_index = planar_shared_index(row, k2, lane);
    shared_plane[shared_index] = values[k2].x;
  }
  __syncwarp();
  const int frequency0 = lane * 2;
  const int frequency1 = frequency0 + 1;
#pragma unroll
  for (int input_pair = 0; input_pair < 8; ++input_pair) {
    const int input_lane0 = input_pair * 2;
    const int input_lane1 = input_lane0 + 1;
    const int local_index0 = (input_lane0 & 3) * 4 + (input_lane0 >> 2);
    const int local_index1 = (input_lane1 & 3) * 4 + (input_lane1 >> 2);
    const int shared_index0 = planar_shared_index(row, frequency0, input_lane0);
    const int shared_index1 = planar_shared_index(row, frequency1, input_lane0);
    const float2 real0 =
        *reinterpret_cast<const float2*>(shared_plane + shared_index0);
    const float2 real1 =
        *reinterpret_cast<const float2*>(shared_plane + shared_index1);
    values[local_index0].x = real0.x;
    values[local_index1].x = real0.y;
    values[16 + local_index0].x = real1.x;
    values[16 + local_index1].x = real1.y;
  }
  __syncwarp();
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    const int shared_index = planar_shared_index(row, k2, lane);
    shared_plane[shared_index] = values[k2].y;
  }
  __syncwarp();
#pragma unroll
  for (int input_pair = 0; input_pair < 8; ++input_pair) {
    const int input_lane0 = input_pair * 2;
    const int input_lane1 = input_lane0 + 1;
    const int local_index0 = (input_lane0 & 3) * 4 + (input_lane0 >> 2);
    const int local_index1 = (input_lane1 & 3) * 4 + (input_lane1 >> 2);
    const int shared_index0 = planar_shared_index(row, frequency0, input_lane0);
    const int shared_index1 = planar_shared_index(row, frequency1, input_lane0);
    const float2 imag0 =
        *reinterpret_cast<const float2*>(shared_plane + shared_index0);
    const float2 imag1 =
        *reinterpret_cast<const float2*>(shared_plane + shared_index1);
    values[local_index0].y = imag0.x;
    values[local_index1].y = imag0.y;
    values[16 + local_index0].y = imag1.x;
    values[16 + local_index1].y = imag1.y;
  }
#if JKF_CUSTOM_IFFT_WARP_LOCAL_CROP
  __syncwarp();
#endif
  fft16_mixed_radix_inverse<0>(values);
  fft16_mixed_radix_inverse<16>(values);
}
#endif

template <int kLength>
__device__ __forceinline__ float2 fft16_special_twiddle(float2 value, int j) {
  if constexpr (kLength == 2) {
    return value;
  } else if constexpr (kLength == 4) {
    const float2 rotated = make_float2(value.y, -value.x);
    return j == 0 ? value : rotated;
  } else if constexpr (kLength == 8) {
    return multiply_w8<false>(value, j);
  } else {
    if ((j & 1) == 0) return multiply_w8<false>(value, j >> 1);
    return complex_mul(value, kForwardTwiddle[j * (kGrid / kLength)]);
  }
}

template <int kLength>
__device__ __forceinline__ float2 fft16_shuffle_stage(float2 value,
                                                      int lane) {
  constexpr int kHalf = kLength / 2;
  const float2 partner = make_float2(
      __shfl_xor_sync(0xffffffffu, value.x, kHalf, 16),
      __shfl_xor_sync(0xffffffffu, value.y, kHalf, 16));
  const int j = lane & (kHalf - 1);
#if JKF_CUSTOM_FFT_SPECIAL16
  const bool lower = (lane & kHalf) == 0;
  const float2 operand = lower ? partner : value;
  const float2 product = fft16_special_twiddle<kLength>(operand, j);
  return lower ? complex_add(value, product) : complex_sub(partner, product);
#else
  const float2 twiddle = kForwardTwiddle[j * (kGrid / kLength)];
  if ((lane & kHalf) == 0) {
    return complex_add(value, complex_mul(partner, twiddle));
  }
  return complex_sub(partner, complex_mul(value, twiddle));
#endif
}

__device__ __forceinline__ float2 fft16_halfwarp(float2 value, int lane) {
  const int source_lane = reverse_low_bits(lane, 4);
  value = make_float2(
      __shfl_sync(0xffffffffu, value.x, source_lane, 16),
      __shfl_sync(0xffffffffu, value.y, source_lane, 16));
  value = fft16_shuffle_stage<2>(value, lane);
  value = fft16_shuffle_stage<4>(value, lane);
  value = fft16_shuffle_stage<8>(value, lane);
  value = fft16_shuffle_stage<16>(value, lane);
  return value;
}

template <bool kInverse>
__device__ __forceinline__ void fft512_shared(float2* shared, int butterfly) {
  float2 input[8];
  float2 output[8];

  int base = butterfly * 8;
#pragma unroll
  for (int q = 0; q < 8; ++q) input[q] = shared[base + q];
  dft8<kInverse>(input, output);
#pragma unroll
  for (int q = 0; q < 8; ++q) shared[base + q] = output[q];
  __syncthreads();

  const int block64 = butterfly >> 3;
  const int j64 = butterfly & 7;
  base = block64 * 64 + j64;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    input[q] = stage_twiddle<kInverse>(shared[base + q * 8], q, j64, 8);
  }
  dft8<kInverse>(input, output);
#pragma unroll
  for (int q = 0; q < 8; ++q) shared[base + q * 8] = output[q];
  __syncthreads();

  const int j512 = butterfly;
  base = j512;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    input[q] = stage_twiddle<kInverse>(shared[base + q * 64], q, j512, 1);
  }
  dft8<kInverse>(input, output);
#pragma unroll
  for (int q = 0; q < 8; ++q) shared[base + q * 64] = output[q];
  __syncthreads();
}

#if JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE
__device__ __forceinline__ int regular_shared_index(int row,
                                                     int logical_index) {
#if JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_FAST
#if JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_RAW_XOR
  return (logical_index << 3) ^ (logical_index & 8) ^ row ^
         (logical_index >> 6);
#else
  const int group_bits = (logical_index << 3) ^ (logical_index & 8);
  const int row_bits = (row ^ (logical_index >> 6)) & 7;
#if JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_ADD
  return group_bits + row_bits;
#else
  return group_bits | row_bits;
#endif
#endif
#else
  const int group = logical_index ^ ((logical_index >> 3) & 1);
  const int swizzled_row = row ^ ((logical_index >> 6) & 7);
  return group * 8 + swizzled_row;
#endif
}

template <bool kInverse>
__device__ __forceinline__ void fft512_shared_swizzled(
    float2* shared, int row, int butterfly, float2 (&final_values)[8]) {
  float2 input[8];
  float2 output[8];

#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int logical_index = butterfly * 8 + q;
    input[q] = shared[regular_shared_index(row, logical_index)];
  }
  dft8<kInverse>(input, output);
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int logical_index = butterfly * 8 + q;
    shared[regular_shared_index(row, logical_index)] = output[q];
  }
  __syncthreads();

  const int block64 = butterfly >> 3;
  const int j64 = butterfly & 7;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int logical_index = block64 * 64 + j64 + q * 8;
    input[q] = stage_twiddle<kInverse>(
        shared[regular_shared_index(row, logical_index)], q, j64, 8);
  }
  dft8<kInverse>(input, output);
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int logical_index = block64 * 64 + j64 + q * 8;
    shared[regular_shared_index(row, logical_index)] = output[q];
  }
  __syncthreads();

  const int j512 = butterfly;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int logical_index = j512 + q * 64;
    input[q] = stage_twiddle<kInverse>(
        shared[regular_shared_index(row, logical_index)], q, j512, 1);
  }
  dft8<kInverse>(input, final_values);
}

#if JKF_CUSTOM_IFFT_DIRECT_REGULAR_LOAD
template <bool kInverse>
__device__ __forceinline__ void fft512_shared_swizzled_after_first(
    float2* shared, int row, int butterfly, float2 (&final_values)[8]) {
  float2 input[8];
  float2 output[8];
  const int block64 = butterfly >> 3;
  const int j64 = butterfly & 7;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int logical_index = block64 * 64 + j64 + q * 8;
    input[q] = stage_twiddle<kInverse>(
        shared[regular_shared_index(row, logical_index)], q, j64, 8);
  }
  dft8<kInverse>(input, output);
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int logical_index = block64 * 64 + j64 + q * 8;
    shared[regular_shared_index(row, logical_index)] = output[q];
  }
  __syncthreads();
  const int j512 = butterfly;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int logical_index = j512 + q * 64;
    input[q] = stage_twiddle<kInverse>(
        shared[regular_shared_index(row, logical_index)], q, j512, 1);
  }
  dft8<kInverse>(input, final_values);
}
#endif
#endif

__global__ void fft512_rows_pad_scale(const float2* __restrict__ image,
                                      const float* __restrict__ scaling,
                                      float2* __restrict__ output) {
  extern __shared__ float2 shared[];
  const int row_index = blockIdx.x;
  const int coil = row_index / kGrid;
  const int y = row_index % kGrid;
  const int lane = threadIdx.x;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int input_x = lane + q * kFftThreads;
    float2 value = make_float2(0.0f, 0.0f);
    if (y < kImage && input_x < kImage) {
      value = image[(coil * kImage + y) * kImage + input_x];
      const float scale = scaling[y * kImage + input_x];
      value.x *= scale;
      value.y *= scale;
    }
    const int reversed = 64 * (lane & 7) + 8 * (lane >> 3) + q;
    shared[reversed] = value;
  }
  __syncthreads();
  fft512_shared<false>(shared, lane);
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int x = lane + q * kFftThreads;
    output[(static_cast<size_t>(coil) * kGrid + y) * kGrid + x] = shared[x];
  }
}

__global__ void fft512_rows_grid(const float2* __restrict__ input,
                                 float2* __restrict__ output) {
  extern __shared__ float2 shared[];
  const int row_index = blockIdx.x;
  const int lane = threadIdx.x;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int x = lane + q * kFftThreads;
    const int reversed = 64 * (lane & 7) + 8 * (lane >> 3) + q;
    shared[reversed] = input[static_cast<size_t>(row_index) * kGrid + x];
  }
  __syncthreads();
  fft512_shared<false>(shared, lane);
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int x = lane + q * kFftThreads;
    output[static_cast<size_t>(row_index) * kGrid + x] = shared[x];
  }
}

__global__ void fft512_rows8_pad_scale(const float2* __restrict__ image,
                                       const float* __restrict__ scaling,
                                       float2* __restrict__ output) {
  extern __shared__ float2 shared[];
#if JKF_CUSTOM_FFT_ROWS8_SKIP_ZERO_ROWS
  constexpr int kRowBlocksPerCoil = kImage / 8;
#else
  constexpr int kRowBlocksPerCoil = kGrid / 8;
#endif
#if JKF_CUSTOM_FFT_ROWS8_GRID2D
  const int coil = blockIdx.y;
  const int row_block = blockIdx.x;
#else
  const int coil = blockIdx.x / kRowBlocksPerCoil;
  const int row_block = blockIdx.x % kRowBlocksPerCoil;
#endif
  const int row_base = row_block * 8;
#if JKF_CUSTOM_FFT_ROWS8_INTERLEAVED_TRANSPOSED
  const int tid = threadIdx.x + 8 * threadIdx.y;
#else
  const int tid = threadIdx.x;
#endif
#pragma unroll
  for (int row = 0; row < 8; ++row) {
    const int y = row_base + row;
    const int x = tid;
    float2 value = make_float2(0.0f, 0.0f);
#if JKF_CUSTOM_FFT_ROWS8_SKIP_ZERO_ROWS
    if (x < kImage) {
#else
    if (y < kImage && x < kImage) {
#endif
      value = image[(coil * kImage + y) * kImage + x];
      const float scale = scaling[y * kImage + x];
      value.x *= scale;
      value.y *= scale;
    }
    const int lane = x & 63;
    const int q = x >> 6;
    const int reversed = 64 * (lane & 7) + 8 * (lane >> 3) + q;
#if JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE
    shared[regular_shared_index(row, reversed)] = value;
#else
    shared[row * kGrid + reversed] = value;
#endif
  }
  __syncthreads();
#if JKF_CUSTOM_FFT_ROWS8_INTERLEAVED_TRANSPOSED
  const int row = threadIdx.x;
  const int butterfly = threadIdx.y;
#else
  const int row = tid >> 6;
  const int butterfly = tid & 63;
#endif
#if JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE
  float2 final_values[8];
  fft512_shared_swizzled<false>(shared, row, butterfly, final_values);
#else
  fft512_shared<false>(shared + row * kGrid, butterfly);
#endif
#if JKF_CUSTOM_FFT_ROWS8_INTERLEAVED_TRANSPOSED
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int kx = butterfly + q * 64;
#if JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE
    const float2 value = final_values[q];
#else
    const float2 value = shared[row * kGrid + kx];
#endif
    output[(static_cast<size_t>(coil) * kGrid + kx) * kGrid + row_base + row] =
        value;
  }
#elif JKF_CUSTOM_FFT_ROWS8_DIRECT_TRANSPOSED
#pragma unroll
  for (int output_pair = 0; output_pair < 4; ++output_pair) {
    const int output_row = output_pair * 2;
    const float2 value0 = shared[output_row * kGrid + tid];
    const float2 value1 = shared[(output_row + 1) * kGrid + tid];
    const size_t output_index =
        (static_cast<size_t>(coil) * kGrid + tid) * kGrid + row_base +
        output_row;
    *reinterpret_cast<float4*>(output + output_index) =
        make_float4(value0.x, value0.y, value1.x, value1.y);
  }
#else
#pragma unroll
  for (int output_row = 0; output_row < 8; ++output_row) {
    const int y = row_base + output_row;
    output[(static_cast<size_t>(coil) * kGrid + y) * kGrid + tid] =
        shared[output_row * kGrid + tid];
  }
#endif
}

#if JKF_CUSTOM_IFFT_REGULAR_X
__global__ void ifft512_rows8_regular_transposed(
    const float2* __restrict__ input, float2* __restrict__ output) {
  extern __shared__ float2 shared[];
  const int coil = blockIdx.x / (kGrid / 8);
  const int row_block = blockIdx.x % (kGrid / 8);
  const int row_base = row_block * 8;
#if JKF_CUSTOM_IFFT_DIRECT_REGULAR_LOAD
  const int row = threadIdx.x;
  const int butterfly = threadIdx.y;
  float2 input_values[8];
  float2 stage_values[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int x = (butterfly >> 3) + 8 * (butterfly & 7) + q * 64;
    input_values[q] =
        input[(static_cast<size_t>(coil) * kGrid + row_base + row) * kGrid + x];
  }
  dft8<true>(input_values, stage_values);
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int logical_index = butterfly * 8 + q;
    shared[regular_shared_index(row, logical_index)] = stage_values[q];
  }
  __syncthreads();
  float2 final_values[8];
  fft512_shared_swizzled_after_first<true>(shared, row, butterfly,
                                            final_values);
#else
  const int linear_thread = threadIdx.x + 8 * threadIdx.y;
#pragma unroll
  for (int row = 0; row < 8; ++row) {
    const int lane = linear_thread & 63;
    const int q = linear_thread >> 6;
    const int reversed = 64 * (lane & 7) + 8 * (lane >> 3) + q;
    const float2 value =
        input[(static_cast<size_t>(coil) * kGrid + row_base + row) * kGrid +
              linear_thread];
    shared[regular_shared_index(row, reversed)] = value;
  }
  __syncthreads();
  const int row = threadIdx.x;
  const int butterfly = threadIdx.y;
  float2 final_values[8];
  fft512_shared_swizzled<true>(shared, row, butterfly, final_values);
#endif
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int k = butterfly + q * 64;
    output[(static_cast<size_t>(coil) * kGrid + k) * kGrid + row_base + row] =
        final_values[q];
  }
}

#if JKF_CUSTOM_IFFT_REGULAR_Y_CROP
__global__ void ifft512_rows8_regular_crop_scale(
    const float2* __restrict__ input, const float* __restrict__ scaling,
    float2* __restrict__ output) {
  extern __shared__ float2 shared[];
  const int coil = blockIdx.x / (kImage / 8);
  const int row_block = blockIdx.x % (kImage / 8);
  const int x_base = row_block * 8;
  const int linear_thread = threadIdx.x + 8 * threadIdx.y;
#pragma unroll
  for (int row = 0; row < 8; ++row) {
    const int lane = linear_thread & 63;
    const int q = linear_thread >> 6;
    const int reversed = 64 * (lane & 7) + 8 * (lane >> 3) + q;
    const float2 value =
        input[(static_cast<size_t>(coil) * kGrid + x_base + row) * kGrid +
              linear_thread];
    shared[regular_shared_index(row, reversed)] = value;
  }
  __syncthreads();
  const int row = threadIdx.x;
  const int butterfly = threadIdx.y;
  float2 final_values[8];
  fft512_shared_swizzled<true>(shared, row, butterfly, final_values);
  constexpr float kInverseScale = 1.0f / static_cast<float>(kGrid * kGrid);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int y = butterfly + q * 64;
    const int x = x_base + row;
    float2 value = final_values[q];
    const float scale = scaling[y * kImage + x] * kInverseScale;
    value.x *= scale;
    value.y *= scale;
    output[(static_cast<size_t>(coil) * kImage + y) * kImage + x] = value;
  }
}
#endif
#endif

__global__ void fft512_rows4_vector32_pad_scale(
    const float2* __restrict__ image,
    const float* __restrict__ scaling,
    float2* __restrict__ output) {
  const int lane = threadIdx.x;
  const int row_in_cta = threadIdx.y;
  const int transform = blockIdx.x * 4 + row_in_cta;
  const int coil = transform / kGrid;
  const int y = transform % kGrid;
  float2 values[32];
#pragma unroll
  for (int q = 0; q < 32; ++q) {
    const int x = lane + q * 16;
    float2 value = make_float2(0.0f, 0.0f);
    if (y < kImage && x < kImage) {
      value = image[(coil * kImage + y) * kImage + x];
      const float scale = scaling[y * kImage + x];
      value.x *= scale;
      value.y *= scale;
    }
#if JKF_CUSTOM_FFT_MIXED32
    values[(q & 3) * 8 + (q >> 2)] = value;
#else
    values[reverse_low_bits(q, 5)] = value;
#endif
  }
#if JKF_CUSTOM_FFT_MIXED32
  fft32_mixed_radix(values);
#else
  fft32_register(values);
#endif
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    values[k2] = complex_mul(
        values[k2], kForwardTwiddle[(lane * k2) & (kGrid - 1)]);
    values[k2] = fft16_halfwarp(values[k2], lane);
  }
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    const int k = k2 + lane * 32;
    output[(static_cast<size_t>(coil) * kGrid + y) * kGrid + k] = values[k2];
  }
}

template <bool FuseSense>
__global__ void fft512_rows4_shared16_pad_scale(
    const float2* __restrict__ image,
    const float2* __restrict__ sensitivity_maps,
    const float* __restrict__ scaling,
    float2* __restrict__ output) {
#if JKF_CUSTOM_FFT_PLANAR_SHARED
  extern __shared__ float shared_scalar[];
  float* shared_real = shared_scalar;
#if JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED
  float* shared_imag = shared_scalar;
#else
  float* shared_imag = shared_scalar + kSharedPlaneElements;
#endif
#else
  extern __shared__ float2 shared[];
#endif
  const int lane = threadIdx.x;
  const int row_in_cta = threadIdx.y;
#if JKF_CUSTOM_FFT_SHARED16_SKIP_ZERO_ROWS
#if JKF_CUSTOM_FFT_SHARED16_ZERO_ROWS_FLAT
  const int coil = blockIdx.x >> 6;
  const int y = ((blockIdx.x & 63) << 2) + row_in_cta;
#else
  const int coil = blockIdx.y;
  const int y = blockIdx.x * 4 + row_in_cta;
#endif
#else
  const int transform = blockIdx.x * 4 + row_in_cta;
  const int coil = transform / kGrid;
  const int y = transform % kGrid;
#endif
  const int row_base = y - row_in_cta;
  float2 values[32];
#if JKF_CUSTOM_FFT_PRUNED_X_HALF_ZERO
  static_assert(kImage * 2 == kGrid);
  static_assert(JKF_CUSTOM_FFT_SHARED16_SKIP_ZERO_ROWS);
#pragma unroll
  for (int q = 0; q < 16; ++q) {
    const int x = lane + q * 16;
    float2 value;
    if constexpr (FuseSense) {
      value = complex_mul(
          image[y * kImage + x],
          sensitivity_maps[(coil * kImage + y) * kImage + x]);
    } else {
      value = image[(coil * kImage + y) * kImage + x];
    }
    const float scale = scaling[y * kImage + x];
    value.x *= scale;
    value.y *= scale;
    values[(q & 3) * 8 + (q >> 2)] = value;
  }
  fft32_mixed_radix_half_zero(values);
#else
#pragma unroll
  for (int q = 0; q < 32; ++q) {
    const int x = lane + q * 16;
    float2 value = make_float2(0.0f, 0.0f);
    if (y < kImage && x < kImage) {
      if constexpr (FuseSense) {
        value = complex_mul(
            image[y * kImage + x],
            sensitivity_maps[(coil * kImage + y) * kImage + x]);
      } else {
        value = image[(coil * kImage + y) * kImage + x];
      }
      const float scale = scaling[y * kImage + x];
      value.x *= scale;
      value.y *= scale;
    }
    values[(q & 3) * 8 + (q >> 2)] = value;
  }
  fft32_mixed_radix(values);
#endif
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    values[k2] = complex_mul(
        values[k2], kForwardTwiddle[(lane * k2) & (kGrid - 1)]);
#if !JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED
#if JKF_CUSTOM_FFT_PLANAR_SHARED
    const int shared_index = planar_shared_index(row_in_cta, k2, lane);
    shared_real[shared_index] = values[k2].x;
    shared_imag[shared_index] = values[k2].y;
#else
    shared[(row_in_cta * 32 + k2) * 16 + lane] = values[k2];
#endif
#endif
  }
#if !JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED
  __syncwarp();
#endif
  const int frequency0 = lane * 2;
  const int frequency1 = frequency0 + 1;
#if JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    const int shared_index = planar_shared_index(row_in_cta, k2, lane);
    shared_real[shared_index] = values[k2].x;
  }
  __syncwarp();
#if JKF_CUSTOM_FFT_PHASED_SCALAR_LOAD
  phased_scalar_load<false>(values, shared_real, row_in_cta, lane);
#else
#pragma unroll
  for (int input_pair = 0; input_pair < 8; ++input_pair) {
    const int input_lane0 = input_pair * 2;
    const int input_lane1 = input_lane0 + 1;
    const int local_index0 = (input_lane0 & 3) * 4 + (input_lane0 >> 2);
    const int local_index1 = (input_lane1 & 3) * 4 + (input_lane1 >> 2);
    const int shared_index0 =
        planar_shared_index(row_in_cta, frequency0, input_lane0);
    const int shared_index1 =
        planar_shared_index(row_in_cta, frequency1, input_lane0);
    const float2 real0 =
        *reinterpret_cast<const float2*>(shared_real + shared_index0);
    const float2 real1 =
        *reinterpret_cast<const float2*>(shared_real + shared_index1);
    values[local_index0].x = real0.x;
    values[local_index1].x = real0.y;
    values[16 + local_index0].x = real1.x;
    values[16 + local_index1].x = real1.y;
  }
#endif
  __syncwarp();
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    const int shared_index = planar_shared_index(row_in_cta, k2, lane);
    shared_imag[shared_index] = values[k2].y;
  }
  __syncwarp();
#if JKF_CUSTOM_FFT_PHASED_SCALAR_LOAD
  phased_scalar_load<true>(values, shared_imag, row_in_cta, lane);
#else
#pragma unroll
  for (int input_pair = 0; input_pair < 8; ++input_pair) {
    const int input_lane0 = input_pair * 2;
    const int input_lane1 = input_lane0 + 1;
    const int local_index0 = (input_lane0 & 3) * 4 + (input_lane0 >> 2);
    const int local_index1 = (input_lane1 & 3) * 4 + (input_lane1 >> 2);
    const int shared_index0 =
        planar_shared_index(row_in_cta, frequency0, input_lane0);
    const int shared_index1 =
        planar_shared_index(row_in_cta, frequency1, input_lane0);
    const float2 imag0 =
        *reinterpret_cast<const float2*>(shared_imag + shared_index0);
    const float2 imag1 =
        *reinterpret_cast<const float2*>(shared_imag + shared_index1);
    values[local_index0].y = imag0.x;
    values[local_index1].y = imag0.y;
    values[16 + local_index0].y = imag1.x;
    values[16 + local_index1].y = imag1.y;
  }
#endif
#elif JKF_CUSTOM_FFT_ALIGNED_PAIR_SWIZZLE
#pragma unroll
  for (int input_pair = 0; input_pair < 8; ++input_pair) {
    const int input_lane0 = input_pair * 2;
    const int input_lane1 = input_lane0 + 1;
    const int local_index0 = (input_lane0 & 3) * 4 + (input_lane0 >> 2);
    const int local_index1 = (input_lane1 & 3) * 4 + (input_lane1 >> 2);
    const int shared_index0 =
        planar_shared_index(row_in_cta, frequency0, input_lane0);
    const int shared_index1 =
        planar_shared_index(row_in_cta, frequency1, input_lane0);
    const float2 real0 =
        *reinterpret_cast<const float2*>(shared_real + shared_index0);
    const float2 imag0 =
        *reinterpret_cast<const float2*>(shared_imag + shared_index0);
    const float2 real1 =
        *reinterpret_cast<const float2*>(shared_real + shared_index1);
    const float2 imag1 =
        *reinterpret_cast<const float2*>(shared_imag + shared_index1);
    values[local_index0] = make_float2(real0.x, imag0.x);
    values[local_index1] = make_float2(real0.y, imag0.y);
    values[16 + local_index0] = make_float2(real1.x, imag1.x);
    values[16 + local_index1] = make_float2(real1.y, imag1.y);
  }
#else
#pragma unroll
  for (int input_lane = 0; input_lane < 16; ++input_lane) {
    const int local_index = (input_lane & 3) * 4 + (input_lane >> 2);
#if JKF_CUSTOM_FFT_PLANAR_SHARED
    const int shared_index0 =
        planar_shared_index(row_in_cta, frequency0, input_lane);
    const int shared_index1 =
        planar_shared_index(row_in_cta, frequency1, input_lane);
    values[local_index] =
        make_float2(shared_real[shared_index0], shared_imag[shared_index0]);
    values[16 + local_index] =
        make_float2(shared_real[shared_index1], shared_imag[shared_index1]);
#else
    values[local_index] =
        shared[(row_in_cta * 32 + frequency0) * 16 + input_lane];
    values[16 + local_index] =
        shared[(row_in_cta * 32 + frequency1) * 16 + input_lane];
#endif
  }
#endif
  fft16_mixed_radix<0>(values);
  fft16_mixed_radix<16>(values);
#if JKF_CUSTOM_FFT_X_DIRECT_TRANSPOSED
#pragma unroll
  for (int k1 = 0; k1 < 16; ++k1) {
    const int kx0 = frequency0 + k1 * 32;
    const int kx1 = frequency1 + k1 * 32;
    output[(static_cast<size_t>(coil) * kGrid + kx0) * kGrid + y] = values[k1];
    output[(static_cast<size_t>(coil) * kGrid + kx1) * kGrid + y] =
        values[16 + k1];
  }
#elif JKF_CUSTOM_FFT_X_TRANSPOSED
#pragma unroll
  for (int k1 = 0; k1 < 16; ++k1) {
    shared[row_in_cta * kGrid + frequency0 + k1 * 32] = values[k1];
    shared[row_in_cta * kGrid + frequency1 + k1 * 32] = values[16 + k1];
  }
  __syncthreads();
  const int linear_thread = row_in_cta * 16 + lane;
#pragma unroll
  for (int kx_iteration = 0; kx_iteration < 8; ++kx_iteration) {
    const int kx = linear_thread + kx_iteration * 64;
    const size_t output_index =
        (static_cast<size_t>(coil) * kGrid + kx) * kGrid + row_base;
    const float2 row0 = shared[kx];
    const float2 row1 = shared[kGrid + kx];
    const float2 row2 = shared[2 * kGrid + kx];
    const float2 row3 = shared[3 * kGrid + kx];
    *reinterpret_cast<float4*>(output + output_index) =
        make_float4(row0.x, row0.y, row1.x, row1.y);
    *reinterpret_cast<float4*>(output + output_index + 2) =
        make_float4(row2.x, row2.y, row3.x, row3.y);
  }
#else
#pragma unroll
  for (int k1 = 0; k1 < 16; ++k1) {
    const size_t output_index =
        (static_cast<size_t>(coil) * kGrid + y) * kGrid +
        frequency0 + k1 * 32;
    const float2 value0 = values[k1];
    const float2 value1 = values[16 + k1];
    *reinterpret_cast<float4*>(output + output_index) =
        make_float4(value0.x, value0.y, value1.x, value1.y);
  }
#endif
}

__global__ void fft512_rows4_shared16_grid(const float2* __restrict__ input,
                                           float2* __restrict__ output) {
#if JKF_CUSTOM_FFT_PLANAR_SHARED
  extern __shared__ float shared_scalar[];
  float* shared_real = shared_scalar;
#if JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED
  float* shared_imag = shared_scalar;
#else
  float* shared_imag = shared_scalar + kSharedPlaneElements;
#endif
#else
  extern __shared__ float2 shared[];
#endif
  const int lane = threadIdx.x;
  const int row_in_cta = threadIdx.y;
  const int transform = blockIdx.x * 4 + row_in_cta;
  float2 values[32];
#pragma unroll
  for (int q = 0; q < 32; ++q) {
    const int x = lane + q * 16;
    values[(q & 3) * 8 + (q >> 2)] =
        input[static_cast<size_t>(transform) * kGrid + x];
  }
  fft32_mixed_radix(values);
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    values[k2] = complex_mul(
        values[k2], kForwardTwiddle[(lane * k2) & (kGrid - 1)]);
#if !JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED
#if JKF_CUSTOM_FFT_PLANAR_SHARED
    const int shared_index = planar_shared_index(row_in_cta, k2, lane);
    shared_real[shared_index] = values[k2].x;
    shared_imag[shared_index] = values[k2].y;
#else
    shared[(row_in_cta * 32 + k2) * 16 + lane] = values[k2];
#endif
#endif
  }
#if !JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED
  __syncwarp();
#endif
  const int frequency0 = lane * 2;
  const int frequency1 = frequency0 + 1;
#if JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    const int shared_index = planar_shared_index(row_in_cta, k2, lane);
    shared_real[shared_index] = values[k2].x;
  }
  __syncwarp();
#if JKF_CUSTOM_FFT_PHASED_SCALAR_LOAD
  phased_scalar_load<false>(values, shared_real, row_in_cta, lane);
#else
#pragma unroll
  for (int input_pair = 0; input_pair < 8; ++input_pair) {
    const int input_lane0 = input_pair * 2;
    const int input_lane1 = input_lane0 + 1;
    const int local_index0 = (input_lane0 & 3) * 4 + (input_lane0 >> 2);
    const int local_index1 = (input_lane1 & 3) * 4 + (input_lane1 >> 2);
    const int shared_index0 =
        planar_shared_index(row_in_cta, frequency0, input_lane0);
    const int shared_index1 =
        planar_shared_index(row_in_cta, frequency1, input_lane0);
    const float2 real0 =
        *reinterpret_cast<const float2*>(shared_real + shared_index0);
    const float2 real1 =
        *reinterpret_cast<const float2*>(shared_real + shared_index1);
    values[local_index0].x = real0.x;
    values[local_index1].x = real0.y;
    values[16 + local_index0].x = real1.x;
    values[16 + local_index1].x = real1.y;
  }
#endif
  __syncwarp();
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    const int shared_index = planar_shared_index(row_in_cta, k2, lane);
    shared_imag[shared_index] = values[k2].y;
  }
  __syncwarp();
#if JKF_CUSTOM_FFT_PHASED_SCALAR_LOAD
  phased_scalar_load<true>(values, shared_imag, row_in_cta, lane);
#else
#pragma unroll
  for (int input_pair = 0; input_pair < 8; ++input_pair) {
    const int input_lane0 = input_pair * 2;
    const int input_lane1 = input_lane0 + 1;
    const int local_index0 = (input_lane0 & 3) * 4 + (input_lane0 >> 2);
    const int local_index1 = (input_lane1 & 3) * 4 + (input_lane1 >> 2);
    const int shared_index0 =
        planar_shared_index(row_in_cta, frequency0, input_lane0);
    const int shared_index1 =
        planar_shared_index(row_in_cta, frequency1, input_lane0);
    const float2 imag0 =
        *reinterpret_cast<const float2*>(shared_imag + shared_index0);
    const float2 imag1 =
        *reinterpret_cast<const float2*>(shared_imag + shared_index1);
    values[local_index0].y = imag0.x;
    values[local_index1].y = imag0.y;
    values[16 + local_index0].y = imag1.x;
    values[16 + local_index1].y = imag1.y;
  }
#endif
#elif JKF_CUSTOM_FFT_ALIGNED_PAIR_SWIZZLE
#pragma unroll
  for (int input_pair = 0; input_pair < 8; ++input_pair) {
    const int input_lane0 = input_pair * 2;
    const int input_lane1 = input_lane0 + 1;
    const int local_index0 = (input_lane0 & 3) * 4 + (input_lane0 >> 2);
    const int local_index1 = (input_lane1 & 3) * 4 + (input_lane1 >> 2);
    const int shared_index0 =
        planar_shared_index(row_in_cta, frequency0, input_lane0);
    const int shared_index1 =
        planar_shared_index(row_in_cta, frequency1, input_lane0);
    const float2 real0 =
        *reinterpret_cast<const float2*>(shared_real + shared_index0);
    const float2 imag0 =
        *reinterpret_cast<const float2*>(shared_imag + shared_index0);
    const float2 real1 =
        *reinterpret_cast<const float2*>(shared_real + shared_index1);
    const float2 imag1 =
        *reinterpret_cast<const float2*>(shared_imag + shared_index1);
    values[local_index0] = make_float2(real0.x, imag0.x);
    values[local_index1] = make_float2(real0.y, imag0.y);
    values[16 + local_index0] = make_float2(real1.x, imag1.x);
    values[16 + local_index1] = make_float2(real1.y, imag1.y);
  }
#else
#pragma unroll
  for (int input_lane = 0; input_lane < 16; ++input_lane) {
    const int local_index = (input_lane & 3) * 4 + (input_lane >> 2);
#if JKF_CUSTOM_FFT_PLANAR_SHARED
    const int shared_index0 =
        planar_shared_index(row_in_cta, frequency0, input_lane);
    const int shared_index1 =
        planar_shared_index(row_in_cta, frequency1, input_lane);
    values[local_index] =
        make_float2(shared_real[shared_index0], shared_imag[shared_index0]);
    values[16 + local_index] =
        make_float2(shared_real[shared_index1], shared_imag[shared_index1]);
#else
    values[local_index] =
        shared[(row_in_cta * 32 + frequency0) * 16 + input_lane];
    values[16 + local_index] =
        shared[(row_in_cta * 32 + frequency1) * 16 + input_lane];
#endif
  }
#endif
  fft16_mixed_radix<0>(values);
  fft16_mixed_radix<16>(values);
#pragma unroll
  for (int k1 = 0; k1 < 16; ++k1) {
    const size_t output_index =
        static_cast<size_t>(transform) * kGrid + frequency0 + k1 * 32;
    const float2 value0 = values[k1];
    const float2 value1 = values[16 + k1];
    *reinterpret_cast<float4*>(output + output_index) =
        make_float4(value0.x, value0.y, value1.x, value1.y);
  }
}

#if JKF_CUSTOM_IFFT_F31
__device__ __forceinline__ int ifft_crop_word_index(int y, int x_local) {
#if JKF_CUSTOM_IFFT_WARP_LOCAL_CROP
  const int x_pair = x_local >> 1;
  const int slot =
      ((x_local & 1) ^ ((y >> 1) & 1)) |
      ((x_pair ^ ((y >> 2) & 1)) << 1) |
      (((y & 1) ^ ((y >> 3) & 1)) << 2) |
      ((((y >> 1) & 1) ^ ((y >> 4) & 1)) << 3);
  return x_pair * kSharedTransformStride + ((y >> 3) << 4) + slot;
#else
  const int slot =
      ((x_local & 1) ^ ((y >> 1) & 1)) |
      ((((x_local >> 1) & 1) ^ ((y >> 2) & 1)) << 1) |
      (((y & 1) ^ ((y >> 3) & 1)) << 2) |
      ((((y >> 1) & 1) ^ ((y >> 4) & 1)) << 3);
  return ((y >> 2) << 4) + slot;
#endif
}

__global__ void ifft512_rows4_inverse_transposed(
    const float2* __restrict__ input, float2* __restrict__ output) {
  extern __shared__ float shared_plane[];
  const int lane = threadIdx.x;
  const int row_in_cta = threadIdx.y;
  const int transform = blockIdx.x * 4 + row_in_cta;
  const int coil = transform / kGrid;
  const int y = transform % kGrid;
  float2 values[32];
#pragma unroll
  for (int q = 0; q < 32; ++q) {
    const int x = lane + q * 16;
    values[(q & 3) * 8 + (q >> 2)] =
        input[static_cast<size_t>(transform) * kGrid + x];
  }
  fft512_inverse_phased(values, shared_plane, row_in_cta, lane);
  const int frequency0 = lane * 2;
  const int frequency1 = frequency0 + 1;
#pragma unroll
  for (int k1 = 0; k1 < 16; ++k1) {
    const int x0 = frequency0 + k1 * 32;
    const int x1 = frequency1 + k1 * 32;
    output[(static_cast<size_t>(coil) * kGrid + x0) * kGrid + y] = values[k1];
    output[(static_cast<size_t>(coil) * kGrid + x1) * kGrid + y] =
        values[16 + k1];
  }
}

__global__ void ifft512_rows4_inverse_crop_scale(
    const float2* __restrict__ input,
    const float* __restrict__ scaling_tiled,
    float2* __restrict__ output) {
  extern __shared__ float shared_scalar[];
  const int lane = threadIdx.x;
  const int row_in_cta = threadIdx.y;
  const int coil = blockIdx.x >> 6;
  const int x = ((blockIdx.x & 63) << 2) + row_in_cta;
  const int transform = coil * kGrid + x;
  float2 values[32];
#pragma unroll
  for (int q = 0; q < 32; ++q) {
    const int y = lane + q * 16;
    values[(q & 3) * 8 + (q >> 2)] =
        input[static_cast<size_t>(transform) * kGrid + y];
  }
  fft512_inverse_phased(values, shared_scalar, row_in_cta, lane);
  const int frequency0 = lane * 2;
  const int frequency1 = frequency0 + 1;

#if !JKF_CUSTOM_IFFT_WARP_LOCAL_CROP
  __syncthreads();
#endif
  float2* crop_shared = reinterpret_cast<float2*>(shared_scalar);
#pragma unroll
  for (int k1 = 0; k1 < 8; ++k1) {
    const int y0 = frequency0 + k1 * 32;
    const int y1 = frequency1 + k1 * 32;
    crop_shared[ifft_crop_word_index(y0, row_in_cta)] = values[k1];
    crop_shared[ifft_crop_word_index(y1, row_in_cta)] = values[16 + k1];
  }
  __syncthreads();

  const int linear_thread = lane + 16 * row_in_cta;
#pragma unroll 1
  for (int iteration = 0; iteration < 16; ++iteration) {
    const int point = linear_thread + iteration * 64;
    const int output_y = point >> 2;
    const int output_x_local = point & 3;
    const int output_x = (blockIdx.x & 63) * 4 + output_x_local;
    float2 value =
        crop_shared[ifft_crop_word_index(output_y, output_x_local)];
    const int x_block = blockIdx.x & 63;
    const float scale =
        scaling_tiled[(x_block * kImage + output_y) * 4 + output_x_local];
    value.x *= scale;
    value.y *= scale;
    output[(static_cast<size_t>(coil) * kImage + output_y) * kImage +
           output_x] = value;
  }
}
#endif

#if JKF_CUSTOM_FFT_Y_N16_FUSED
template <int kK1Base>
__device__ __forceinline__ void stage_fused_panel_phase(
    float2 (&values)[32], uint32_t* panel_shared, int row, int coil,
    int frequency0, int frequency1) {
#pragma unroll
  for (int local_k1 = 0; local_k1 < 8; ++local_k1) {
    constexpr int kPhase = kK1Base / 8;
    const int k1 = kK1Base + local_k1;
    const int ky0 = frequency0 + k1 * 32;
    const int ky1 = frequency1 + k1 * 32;
    const int ky_local0 = ky0 - kPhase * 256;
    const int ky_local1 = ky1 - kPhase * 256;
    const float2 value0 = values[k1];
    const float2 value1 = values[16 + k1];
    panel_shared[fused_panel_word_index(ky_local0, row, coil)] =
        pack_half_pair(value0.x, value0.y);
    panel_shared[fused_panel_word_index(ky_local1, row, coil)] =
        pack_half_pair(value1.x, value1.y);
  }
}

__global__ void fft512_allcoils4_shared16_to_n16(
    const float2* __restrict__ input, half* __restrict__ output) {
  extern __shared__ float shared_scalar[];
  const int lane = threadIdx.x;
  const int row_in_cta = threadIdx.y;
  const int coil = threadIdx.z;
  const int kx = blockIdx.x * 4 + row_in_cta;
  const int transform = coil * kGrid + kx;
  float* coil_shared = shared_scalar + coil * kSharedPlaneElements;
  float2 values[32];

#pragma unroll
  for (int q = 0; q < 32; ++q) {
    const int y = lane + q * 16;
    values[(q & 3) * 8 + (q >> 2)] =
        input[static_cast<size_t>(transform) * kGrid + y];
  }
  fft32_mixed_radix(values);
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    values[k2] = complex_mul(
        values[k2], kForwardTwiddle[(lane * k2) & (kGrid - 1)]);
  }

#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    const int shared_index = planar_shared_index(row_in_cta, k2, lane);
    coil_shared[shared_index] = values[k2].x;
  }
  __syncwarp();
  const int frequency0 = lane * 2;
  const int frequency1 = frequency0 + 1;
#pragma unroll
  for (int input_pair = 0; input_pair < 8; ++input_pair) {
    const int input_lane0 = input_pair * 2;
    const int input_lane1 = input_lane0 + 1;
    const int local_index0 = (input_lane0 & 3) * 4 + (input_lane0 >> 2);
    const int local_index1 = (input_lane1 & 3) * 4 + (input_lane1 >> 2);
    const int shared_index0 =
        planar_shared_index(row_in_cta, frequency0, input_lane0);
    const int shared_index1 =
        planar_shared_index(row_in_cta, frequency1, input_lane0);
    const float2 real0 =
        *reinterpret_cast<const float2*>(coil_shared + shared_index0);
    const float2 real1 =
        *reinterpret_cast<const float2*>(coil_shared + shared_index1);
    values[local_index0].x = real0.x;
    values[local_index1].x = real0.y;
    values[16 + local_index0].x = real1.x;
    values[16 + local_index1].x = real1.y;
  }
  __syncwarp();
#pragma unroll
  for (int k2 = 0; k2 < 32; ++k2) {
    const int shared_index = planar_shared_index(row_in_cta, k2, lane);
    coil_shared[shared_index] = values[k2].y;
  }
  __syncwarp();
#pragma unroll
  for (int input_pair = 0; input_pair < 8; ++input_pair) {
    const int input_lane0 = input_pair * 2;
    const int input_lane1 = input_lane0 + 1;
    const int local_index0 = (input_lane0 & 3) * 4 + (input_lane0 >> 2);
    const int local_index1 = (input_lane1 & 3) * 4 + (input_lane1 >> 2);
    const int shared_index0 =
        planar_shared_index(row_in_cta, frequency0, input_lane0);
    const int shared_index1 =
        planar_shared_index(row_in_cta, frequency1, input_lane0);
    const float2 imag0 =
        *reinterpret_cast<const float2*>(coil_shared + shared_index0);
    const float2 imag1 =
        *reinterpret_cast<const float2*>(coil_shared + shared_index1);
    values[local_index0].y = imag0.x;
    values[local_index1].y = imag0.y;
    values[16 + local_index0].y = imag1.x;
    values[16 + local_index1].y = imag1.y;
  }
  fft16_mixed_radix<0>(values);
  fft16_mixed_radix<16>(values);

  __syncthreads();
  uint32_t* panel_shared = reinterpret_cast<uint32_t*>(shared_scalar);
  uint32_t* output_words = reinterpret_cast<uint32_t*>(output);
  const int linear_thread =
      lane + 16 * row_in_cta + 64 * coil;
  const int warp_id = linear_thread >> 5;
  const int warp_lane = linear_thread & 31;
  const int panel_row = warp_lane >> 3;
  const int panel_coil = warp_lane & 7;
  const int pair = panel_coil & 3;
  const int row_lane_base = panel_row * 8;

#if JKF_CUSTOM_FFT_FUSED_PHASE_ROLLED
#pragma unroll 1
#else
#pragma unroll
#endif
  for (int phase = 0; phase < 2; ++phase) {
#if JKF_CUSTOM_FFT_FUSED_PHASE_ROLLED
    if (phase == 0) {
      stage_fused_panel_phase<0>(values, panel_shared, row_in_cta, coil,
                                 frequency0, frequency1);
    } else {
      stage_fused_panel_phase<8>(values, panel_shared, row_in_cta, coil,
                                 frequency0, frequency1);
    }
#else
#pragma unroll
    for (int local_k1 = 0; local_k1 < 8; ++local_k1) {
      const int k1 = phase * 8 + local_k1;
      const int ky0 = frequency0 + k1 * 32;
      const int ky1 = frequency1 + k1 * 32;
      const int ky_local0 = ky0 - phase * 256;
      const int ky_local1 = ky1 - phase * 256;
      const float2 value0 = values[k1];
      const float2 value1 = values[16 + k1];
      panel_shared[fused_panel_word_index(ky_local0, row_in_cta, coil)] =
          pack_half_pair(value0.x, value0.y);
      panel_shared[fused_panel_word_index(ky_local1, row_in_cta, coil)] =
          pack_half_pair(value1.x, value1.y);
    }
#endif
    __syncthreads();

#if JKF_CUSTOM_FFT_FUSED_WRITER_ROLLED
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int ky_iteration = 0; ky_iteration < 16; ++ky_iteration) {
      const int ky_local = warp_id + ky_iteration * 16;
      const uint32_t complex_word =
          panel_shared[fused_panel_word_index(ky_local, panel_row,
                                               panel_coil)];
      const uint32_t first = __shfl_sync(
          0xffffffffu, complex_word, row_lane_base + pair * 2);
      const uint32_t second = __shfl_sync(
          0xffffffffu, complex_word, row_lane_base + pair * 2 + 1);
      const uint32_t packed = panel_coil < 4
                                  ? ((first & 0xffffu) |
                                     ((second & 0xffffu) << 16))
                                  : ((first >> 16) |
                                     (second & 0xffff0000u));
      const int ky = phase * 256 + ky_local;
      const int output_kx = blockIdx.x * 4 + panel_row;
      const size_t output_word_base =
          (static_cast<size_t>(ky) * kGrid + output_kx) * 8;
      output_words[output_word_base + panel_coil] = packed;
    }
    if (phase == 0) __syncthreads();
  }
}
#endif

__global__ void fft512_columns8(const float2* __restrict__ input,
                                float2* __restrict__ output) {
  extern __shared__ float2 shared[];
  const int coil = blockIdx.x / (kGrid / 8);
  const int column_block = blockIdx.x % (kGrid / 8);
  const int column_base = column_block * 8;
  const int tid = threadIdx.x;
#pragma unroll
  for (int iteration = 0; iteration < 8; ++iteration) {
    const int linear = tid + iteration * 512;
    const int y = linear >> 3;
    const int column = linear & 7;
    const int lane = y & 63;
    const int q = y >> 6;
    const int reversed = 64 * (lane & 7) + 8 * (lane >> 3) + q;
    shared[column * kGrid + reversed] =
        input[(static_cast<size_t>(coil) * kGrid + y) * kGrid +
              column_base + column];
  }
  __syncthreads();
  const int column = tid >> 6;
  const int butterfly = tid & 63;
  fft512_shared<false>(shared + column * kGrid, butterfly);
#pragma unroll
  for (int output_column = 0; output_column < 8; ++output_column) {
    output[(static_cast<size_t>(coil) * kGrid + column_base + output_column) *
               kGrid +
           tid] = shared[output_column * kGrid + tid];
  }
}

__global__ void transpose_complex_32(const float2* __restrict__ input,
                                     float2* __restrict__ output) {
  __shared__ float2 tile[32][33];
  const int batch = blockIdx.z;
  const int x = blockIdx.x * 32 + threadIdx.x;
  const int y = blockIdx.y * 32 + threadIdx.y;
  const size_t batch_base = static_cast<size_t>(batch) * kGrid * kGrid;
#pragma unroll
  for (int offset = 0; offset < 32; offset += 8) {
    tile[threadIdx.y + offset][threadIdx.x] =
        input[batch_base + static_cast<size_t>(y + offset) * kGrid + x];
  }
  __syncthreads();
  const int output_x = blockIdx.y * 32 + threadIdx.x;
  const int output_y = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
  for (int offset = 0; offset < 32; offset += 8) {
    output[batch_base + static_cast<size_t>(output_y + offset) * kGrid +
           output_x] = tile[threadIdx.x][threadIdx.y + offset];
  }
}

__global__ void transpose_to_n16_half(const float2* __restrict__ input,
                                      half* __restrict__ output) {
  __shared__ float2 tile[kCoils][16][17];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int input_row = blockIdx.y * 16 + ty;
  const int input_col = blockIdx.x * 16 + tx;
#pragma unroll
  for (int coil = 0; coil < kCoils; ++coil) {
    tile[coil][ty][tx] =
        input[(static_cast<size_t>(coil) * kGrid + input_row) * kGrid +
              input_col];
  }
  __syncthreads();
  const int output_y = blockIdx.x * 16 + ty;
  const int output_x = blockIdx.y * 16 + tx;
  const size_t base =
      (static_cast<size_t>(output_y) * kGrid + output_x) * kChannels;
  uint4 real_values;
  uint4 imag_values;
  const float2 value0 = tile[0][tx][ty];
  const float2 value1 = tile[1][tx][ty];
  const float2 value2 = tile[2][tx][ty];
  const float2 value3 = tile[3][tx][ty];
  const float2 value4 = tile[4][tx][ty];
  const float2 value5 = tile[5][tx][ty];
  const float2 value6 = tile[6][tx][ty];
  const float2 value7 = tile[7][tx][ty];
  real_values.x = pack_half_pair(value0.x, value1.x);
  real_values.y = pack_half_pair(value2.x, value3.x);
  real_values.z = pack_half_pair(value4.x, value5.x);
  real_values.w = pack_half_pair(value6.x, value7.x);
  imag_values.x = pack_half_pair(value0.y, value1.y);
  imag_values.y = pack_half_pair(value2.y, value3.y);
  imag_values.z = pack_half_pair(value4.y, value5.y);
  imag_values.w = pack_half_pair(value6.y, value7.y);
  *reinterpret_cast<uint4*>(output + base) = real_values;
  *reinterpret_cast<uint4*>(output + base + 8) = imag_values;
}

__global__ void baseline_pad_scale(const float2* __restrict__ image,
                                   const float* __restrict__ scaling,
                                   float2* __restrict__ grid) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total = kCoils * kGrid * kGrid;
  if (index >= total) return;
  const int x = index % kGrid;
  const int y = (index / kGrid) % kGrid;
  const int coil = index / (kGrid * kGrid);
  float2 value = make_float2(0.0f, 0.0f);
  if (x < kImage && y < kImage) {
    value = image[(coil * kImage + y) * kImage + x];
    const float scale = scaling[y * kImage + x];
    value.x *= scale;
    value.y *= scale;
  }
  grid[index] = value;
}

__global__ void baseline_complex_to_half(const float2* __restrict__ input,
                                         half* __restrict__ output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t count = kGrid * kGrid * kCoils;
  if (index >= count) return;
  const uint32_t row = index / kCoils;
  const uint32_t coil = index % kCoils;
  const float2 value =
      input[static_cast<size_t>(coil) * kGrid * kGrid + row];
  output[static_cast<size_t>(row) * kChannels + coil] = __float2half_rn(value.x);
  output[static_cast<size_t>(row) * kChannels + coil + kCoils] =
      __float2half_rn(value.y);
}

struct ErrorStats {
  float max_abs = 0.0f;
  double rel_l2 = 0.0;
  size_t nonfinite = 0;
};

ErrorStats compare_complex(const float2* candidate,
                           const float2* reference,
                           size_t elements) {
  std::vector<float2> candidate_host(elements);
  std::vector<float2> reference_host(elements);
  CUDA_CHECK(cudaMemcpy(candidate_host.data(), candidate,
                        elements * sizeof(float2), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(reference_host.data(), reference,
                        elements * sizeof(float2), cudaMemcpyDeviceToHost));
  ErrorStats stats;
  double error_square = 0.0;
  double ref_square = 0.0;
  for (size_t index = 0; index < elements; ++index) {
    const float2 candidate_value = candidate_host[index];
    const float2 reference_value = reference_host[index];
    if (!std::isfinite(candidate_value.x) || !std::isfinite(candidate_value.y)) {
      ++stats.nonfinite;
    }
    const float diff_x = candidate_value.x - reference_value.x;
    const float diff_y = candidate_value.y - reference_value.y;
    stats.max_abs = std::max(stats.max_abs,
                             std::max(std::fabs(diff_x), std::fabs(diff_y)));
    error_square += static_cast<double>(diff_x) * diff_x +
                    static_cast<double>(diff_y) * diff_y;
    ref_square += static_cast<double>(reference_value.x) * reference_value.x +
                  static_cast<double>(reference_value.y) * reference_value.y;
  }
  stats.rel_l2 = std::sqrt(error_square / std::max(ref_square, 1e-30));
  return stats;
}

ErrorStats compare_half(const half* candidate,
                        const half* reference,
                        size_t elements) {
  std::vector<half> candidate_host(elements);
  std::vector<half> reference_host(elements);
  CUDA_CHECK(cudaMemcpy(candidate_host.data(), candidate, elements * sizeof(half),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(reference_host.data(), reference, elements * sizeof(half),
                        cudaMemcpyDeviceToHost));
  ErrorStats stats;
  double error_square = 0.0;
  double ref_square = 0.0;
  for (size_t index = 0; index < elements; ++index) {
    const float candidate_value = __half2float(candidate_host[index]);
    const float reference_value = __half2float(reference_host[index]);
    if (!std::isfinite(candidate_value)) ++stats.nonfinite;
    const float diff = candidate_value - reference_value;
    stats.max_abs = std::max(stats.max_abs, std::fabs(diff));
    error_square += static_cast<double>(diff) * diff;
    ref_square += static_cast<double>(reference_value) * reference_value;
  }
  stats.rel_l2 = std::sqrt(error_square / std::max(ref_square, 1e-30));
  return stats;
}

void launch_candidate(const float2* image,
                      const float* scaling,
                      float2* temp0,
                      float2* temp1,
                      float2* temp2,
                      half* output,
                      cudaStream_t stream) {
#if JKF_CUSTOM_FFT_SHARED16
#if JKF_CUSTOM_FFT_SHARED16_SKIP_ZERO_ROWS
#if JKF_CUSTOM_FFT_SHARED16_ZERO_ROWS_FLAT
  fft512_rows4_shared16_pad_scale<false>
      <<<kCoils * (kImage / 4), dim3(16, 4), kShared16Bytes, stream>>>(
          image, nullptr, scaling, temp0);
#else
  const dim3 shared16_grid(kImage / 4, kCoils);
  fft512_rows4_shared16_pad_scale<false>
      <<<shared16_grid, dim3(16, 4), kShared16Bytes, stream>>>(
          image, nullptr, scaling, temp0);
#endif
#else
  fft512_rows4_shared16_pad_scale<false>
      <<<kCoils * (kGrid / 4), dim3(16, 4), kShared16Bytes, stream>>>(
          image, nullptr, scaling, temp0);
#endif
#elif JKF_CUSTOM_FFT_VECTOR32
  fft512_rows4_vector32_pad_scale<<<kCoils * (kGrid / 4), dim3(16, 4), 0,
                                    stream>>>(image, scaling, temp0);
#elif JKF_CUSTOM_FFT_ROWS8
#if JKF_CUSTOM_FFT_ROWS8_INTERLEAVED_TRANSPOSED
#if JKF_CUSTOM_FFT_ROWS8_SKIP_ZERO_ROWS
#if JKF_CUSTOM_FFT_ROWS8_GRID2D
  const dim3 rows8_grid(kImage / 8, kCoils);
#else
  constexpr int kRows8Blocks = kCoils * (kImage / 8);
#endif
#else
  constexpr int kRows8Blocks = kCoils * (kGrid / 8);
#endif
#if JKF_CUSTOM_FFT_ROWS8_GRID2D
  fft512_rows8_pad_scale<<<rows8_grid, dim3(8, 64),
                           8 * kGrid * sizeof(float2), stream>>>(image, scaling,
                                                                 temp0);
#else
  fft512_rows8_pad_scale<<<kRows8Blocks, dim3(8, 64),
                           8 * kGrid * sizeof(float2), stream>>>(image, scaling,
                                                                 temp0);
#endif
#else
  fft512_rows8_pad_scale<<<kCoils * (kGrid / 8), 512,
                           8 * kGrid * sizeof(float2), stream>>>(image, scaling,
                                                                 temp0);
#endif
#else
  fft512_rows_pad_scale<<<kCoils * kGrid, kFftThreads,
                          kGrid * sizeof(float2), stream>>>(image, scaling,
                                                            temp0);
#endif
#if JKF_CUSTOM_FFT_Y_N16_FUSED_KX1
  fft512_allcoils1_shared16_to_n16<<<kGrid, dim3(16, 1, kCoils),
                                      kFusedKx1SharedBytes, stream>>>(temp0,
                                                                      output,
                                                                      nullptr,
                                                                      nullptr);
#elif JKF_CUSTOM_FFT_Y_N16_FUSED
  fft512_allcoils4_shared16_to_n16<<<kGrid / 4, dim3(16, 4, kCoils),
                                      kFusedYSharedBytes, stream>>>(temp0,
                                                                    output);
#elif JKF_CUSTOM_FFT_Y_SHARED16
#if JKF_CUSTOM_FFT_X_TRANSPOSED || JKF_CUSTOM_FFT_X_DIRECT_TRANSPOSED
  fft512_rows4_shared16_grid<<<kCoils * (kGrid / 4), dim3(16, 4),
                               kShared16Bytes, stream>>>(temp0,
                                                                       temp2);
#else
  transpose_complex_32<<<dim3(kGrid / 32, kGrid / 32, kCoils), dim3(32, 8),
                         0, stream>>>(temp0, temp1);
  fft512_rows4_shared16_grid<<<kCoils * (kGrid / 4), dim3(16, 4),
                               kShared16Bytes, stream>>>(temp1,
                                                                       temp2);
#endif
#elif JKF_CUSTOM_FFT_FUSED_COLUMNS
  fft512_columns8<<<kCoils * (kGrid / 8), 512,
                    8 * kGrid * sizeof(float2), stream>>>(temp0, temp2);
#else
  transpose_complex_32<<<dim3(kGrid / 32, kGrid / 32, kCoils), dim3(32, 8),
                         0, stream>>>(temp0, temp1);
  fft512_rows_grid<<<kCoils * kGrid, kFftThreads,
                     kGrid * sizeof(float2), stream>>>(temp1, temp2);
#endif
#if !JKF_CUSTOM_FFT_Y_N16_FUSED
  transpose_to_n16_half<<<dim3(kGrid / 16, kGrid / 16), dim3(16, 16), 0,
                          stream>>>(temp2, output);
#endif
}

#if JKF_CUSTOM_FFT_Y_N16_FUSED_KX1 && \
    JKF_CUSTOM_FFT_SHARED16_ZERO_ROWS_FLAT
void launch_candidate_sense(const float2* image,
                            const float2* sensitivity_maps,
                            const float* scaling,
                            float2* transposed_workspace,
                            half* output,
                            cudaStream_t stream) {
  fft512_rows4_shared16_pad_scale<true>
      <<<kCoils * (kImage / 4), dim3(16, 4), kShared16Bytes, stream>>>(
          image, sensitivity_maps, scaling, transposed_workspace);
  fft512_allcoils1_shared16_to_n16
      <<<kGrid, dim3(16, 1, kCoils), kFusedKx1SharedBytes, stream>>>(
          transposed_workspace, output, nullptr, nullptr);
}
#endif

#if !JKF_CUSTOM_FFT_NO_CUFFT_RUNTIME
void launch_baseline(const float2* image,
                     const float* scaling,
                     float2* grid_input,
                     float2* grid_output,
                     half* output,
                     cufftHandle plan,
                     cudaStream_t stream) {
  constexpr int kThreads = 256;
  const uint32_t grid_elements = kCoils * kGrid * kGrid;
  baseline_pad_scale<<<(grid_elements + kThreads - 1) / kThreads, kThreads, 0,
                        stream>>>(image, scaling, grid_input);
  CUFFT_CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(grid_input),
                           reinterpret_cast<cufftComplex*>(grid_output),
                           CUFFT_FORWARD));
  const uint32_t panel_elements = kGrid * kGrid * kCoils;
  baseline_complex_to_half
      <<<(panel_elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          grid_output, output);
}
#endif

}  // namespace

#if JKF_CUSTOM_FFT_Y_N16_FUSED_KX1 || JKF_CUSTOM_IFFT_F31
extern "C" cudaError_t jkf_initialize_custom_fft_twiddles() {
  std::vector<float2> twiddle(kGrid);
  for (int index = 0; index < kGrid; ++index) {
    const double angle = -2.0 * 3.14159265358979323846 * index / kGrid;
    twiddle[index] =
        make_float2(static_cast<float>(std::cos(angle)),
                    static_cast<float>(std::sin(angle)));
  }
  return cudaMemcpyToSymbol(kForwardTwiddle, twiddle.data(),
                            twiddle.size() * sizeof(float2));
}
#endif

#if JKF_CUSTOM_FFT_Y_N16_FUSED_KX1
extern "C" void jkf_launch_custom_fft_f30_forward(
    const float2* image, const float* scaling, float2* transposed_workspace,
    half* n16_output, cudaStream_t stream) {
  launch_candidate(image, scaling, transposed_workspace, nullptr, nullptr,
                   n16_output, stream);
}

extern "C" void jkf_launch_custom_fft_f30_forward_split2(
    const float2* image, const float* scaling, float2* transposed_workspace,
    half* n16_high_output, half* n16_residual_output, cudaStream_t stream) {
  fft512_rows4_shared16_pad_scale<false>
      <<<kCoils * (kImage / 4), dim3(16, 4), kShared16Bytes, stream>>>(
          image, nullptr, scaling, transposed_workspace);
  fft512_allcoils1_shared16_to_n16
      <<<kGrid, dim3(16, 1, kCoils), kFusedKx1SharedBytes, stream>>>(
          transposed_workspace, n16_high_output, n16_residual_output, nullptr);
}

extern "C" void jkf_launch_custom_fft_f30_forward_split2_debug(
    const float2* image, const float* scaling, float2* transposed_workspace,
    half* n16_high_output, half* n16_residual_output,
    float2* fp32_endpoint_output, cudaStream_t stream) {
  fft512_rows4_shared16_pad_scale<false>
      <<<kCoils * (kImage / 4), dim3(16, 4), kShared16Bytes, stream>>>(
          image, nullptr, scaling, transposed_workspace);
  fft512_allcoils1_shared16_to_n16
      <<<kGrid, dim3(16, 1, kCoils), kFusedKx1SharedBytes, stream>>>(
          transposed_workspace, n16_high_output, n16_residual_output,
          fp32_endpoint_output);
}

extern "C" void jkf_launch_custom_fft_f30_forward_sense(
    const float2* image, const float2* sensitivity_maps,
    const float* scaling, float2* transposed_workspace,
    half* n16_output, cudaStream_t stream) {
  launch_candidate_sense(image, sensitivity_maps, scaling,
                         transposed_workspace, n16_output, stream);
}

extern "C" void jkf_launch_custom_fft_f30_forward_sense_split2(
    const float2* image, const float2* sensitivity_maps,
    const float* scaling, float2* transposed_workspace,
    half* n16_high_output, half* n16_residual_output, cudaStream_t stream) {
  fft512_rows4_shared16_pad_scale<true>
      <<<kCoils * (kImage / 4), dim3(16, 4), kShared16Bytes, stream>>>(
          image, sensitivity_maps, scaling, transposed_workspace);
  fft512_allcoils1_shared16_to_n16
      <<<kGrid, dim3(16, 1, kCoils), kFusedKx1SharedBytes, stream>>>(
          transposed_workspace, n16_high_output, n16_residual_output, nullptr);
}

#endif

#if JKF_CUSTOM_IFFT_F31
extern "C" void jkf_launch_custom_ifft_f31_adjoint(
    const float2* grid_input, float2* transposed_workspace,
    const float* scaling, const float* scaling_tiled, float2* image_output,
    cudaStream_t stream) {
#if JKF_CUSTOM_IFFT_REGULAR_X
  ifft512_rows8_regular_transposed<<<kCoils * (kGrid / 8), dim3(8, 64),
                                      8 * kGrid * sizeof(float2), stream>>>(
      grid_input, transposed_workspace);
#else
  ifft512_rows4_inverse_transposed<<<kCoils * (kGrid / 4), dim3(16, 4),
                                      kShared16Bytes, stream>>>(
      grid_input, transposed_workspace);
#endif
#if JKF_CUSTOM_IFFT_REGULAR_Y_CROP
  ifft512_rows8_regular_crop_scale<<<kCoils * (kImage / 8), dim3(8, 64),
                                      8 * kGrid * sizeof(float2), stream>>>(
      transposed_workspace, scaling, image_output);
#else
  ifft512_rows4_inverse_crop_scale<<<kCoils * (kImage / 4), dim3(16, 4),
                                      kShared16Bytes, stream>>>(
      transposed_workspace, scaling_tiled, image_output);
#endif
}
#endif

#if !JKF_CUSTOM_FFT_NO_MAIN
int main(int argc, char** argv) {
  const int warmup = argc > 1 ? std::atoi(argv[1]) : 20;
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 100;
  const int stress = argc > 3 ? std::atoi(argv[3]) : 0;
  const bool graph_mode = argc > 4 ? std::atoi(argv[4]) != 0 : false;
  if (graph_mode) {
    std::fprintf(stderr, "custom FFT Graph mode is not admitted\n");
    return 2;
  }

  std::vector<float2> twiddle(kGrid);
  for (int index = 0; index < kGrid; ++index) {
    const double angle = -2.0 * 3.14159265358979323846 * index / kGrid;
    twiddle[index] =
        make_float2(static_cast<float>(std::cos(angle)),
                    static_cast<float>(std::sin(angle)));
  }
  CUDA_CHECK(cudaMemcpyToSymbol(kForwardTwiddle, twiddle.data(),
                                twiddle.size() * sizeof(float2)));

  const size_t image_elements =
      static_cast<size_t>(kCoils) * kImage * kImage;
  const size_t grid_elements =
      static_cast<size_t>(kCoils) * kGrid * kGrid;
  const size_t panel_elements =
      static_cast<size_t>(kGrid) * kGrid * kChannels;
  std::vector<float2> image_host(image_elements);
  std::vector<float> scaling_host(kImage * kImage);
  for (size_t index = 0; index < image_elements; ++index) {
    const int real_code = static_cast<int>((index * 17 + 1) % 257) - 128;
    const int imag_code = static_cast<int>((index * 23 + 2) % 263) - 131;
    image_host[index] =
        make_float2(static_cast<float>(real_code) / (257.0f * 64.0f),
                    static_cast<float>(imag_code) / (263.0f * 64.0f));
  }
  for (int y = 0; y < kImage; ++y) {
    for (int x = 0; x < kImage; ++x) {
      scaling_host[y * kImage + x] =
          1.0f + static_cast<float>((x + 3 * y) & 7) * 0.015625f;
    }
  }

  float2* image = nullptr;
  float* scaling = nullptr;
  float2* grid_input = nullptr;
  float2* baseline_grid = nullptr;
  float2* temp0 = nullptr;
  float2* temp1 = nullptr;
  float2* temp2 = nullptr;
  float2* candidate_grid = nullptr;
  half* baseline_panel = nullptr;
  half* candidate_panel = nullptr;
  CUDA_CHECK(cudaMalloc(&image, image_elements * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&scaling, scaling_host.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&grid_input, grid_elements * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&baseline_grid, grid_elements * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&temp0, grid_elements * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&temp1, grid_elements * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&temp2, grid_elements * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&candidate_grid, grid_elements * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&baseline_panel, panel_elements * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&candidate_panel, panel_elements * sizeof(half)));
  CUDA_CHECK(cudaMemcpy(image, image_host.data(), image_elements * sizeof(float2),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(scaling, scaling_host.data(),
                        scaling_host.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
#if JKF_CUSTOM_FFT_Y_N16_FUSED
  CUDA_CHECK(cudaFuncSetAttribute(
      fft512_allcoils4_shared16_to_n16,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kFusedYSharedBytes)));
#endif
  cufftHandle plan = 0;
  int dimensions[2] = {kGrid, kGrid};
  CUFFT_CHECK(cufftPlanMany(&plan, 2, dimensions, nullptr, 1, kGrid * kGrid,
                            nullptr, 1, kGrid * kGrid, CUFFT_C2C, kCoils));
  CUFFT_CHECK(cufftSetStream(plan, stream));

#if JKF_CUSTOM_FFT_ROWS8_SKIP_ZERO_ROWS || \
    JKF_CUSTOM_FFT_SHARED16_SKIP_ZERO_ROWS
  CUDA_CHECK(cudaMemsetAsync(temp0, 0, grid_elements * sizeof(float2), stream));
#endif

  launch_baseline(image, scaling, grid_input, baseline_grid, baseline_panel,
                  plan, stream);
  launch_candidate(image, scaling, temp0, temp1, temp2, candidate_panel, stream);
#if JKF_CUSTOM_FFT_Y_N16_FUSED
  fft512_rows4_shared16_grid<<<kCoils * (kGrid / 4), dim3(16, 4),
                               kShared16Bytes, stream>>>(temp0, temp2);
#endif
  transpose_complex_32<<<dim3(kGrid / 32, kGrid / 32, kCoils), dim3(32, 8),
                         0, stream>>>(temp2, candidate_grid);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const ErrorStats grid_error =
      compare_complex(candidate_grid, baseline_grid, grid_elements);
  const ErrorStats panel_error =
      compare_half(candidate_panel, baseline_panel, panel_elements);
  const bool correct = grid_error.nonfinite == 0 && panel_error.nonfinite == 0 &&
                       grid_error.rel_l2 <= 2e-5 &&
                       panel_error.rel_l2 <= 5e-4 &&
                       panel_error.max_abs <= 0.5f;
  if (!correct) {
    std::printf(
        "{\"variant\":\"custom_fft_radix8_v0\",\"correct\":false,"
        "\"grid_max_abs\":%.9g,\"grid_rel_l2\":%.9g,"
        "\"panel_max_abs\":%.9g,\"panel_rel_l2\":%.9g,"
        "\"nonfinite\":%zu}\n",
        grid_error.max_abs, grid_error.rel_l2, panel_error.max_abs,
        panel_error.rel_l2, grid_error.nonfinite + panel_error.nonfinite);
    return 3;
  }

  const auto launch_once = [&]() {
#if JKF_CUSTOM_FFT_CANDIDATE
    launch_candidate(image, scaling, temp0, temp1, temp2, candidate_panel,
                     stream);
#else
    launch_baseline(image, scaling, grid_input, baseline_grid, baseline_panel,
                    plan, stream);
#endif
  };
  for (int index = 0; index < warmup; ++index) launch_once();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int index = 0; index < iterations; ++index) launch_once();
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  for (int index = 0; index < stress; ++index) launch_once();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const double latency_ms = static_cast<double>(milliseconds) / iterations;
  const char* variant =
#if JKF_CUSTOM_FFT_CANDIDATE
#if JKF_CUSTOM_FFT_Y_N16_FUSED_KX1
      "custom_fft_f30_kx1_y_n16_fused_endpoint";
#elif JKF_CUSTOM_FFT_FUSED_PHASE_ROLLED
      "custom_fft_f29_rolled_phase_y_n16_fused_endpoint";
#elif JKF_CUSTOM_FFT_FUSED_WRITER_ROLLED
      "custom_fft_f28_rolled_y_n16_fused_endpoint";
#elif JKF_CUSTOM_FFT_Y_N16_FUSED
      "custom_fft_f27_y_n16_fused_endpoint";
#elif JKF_CUSTOM_FFT_PHASED_VOLATILE_SCALAR
      "custom_fft_f26_volatile_scalar_swizzled_endpoint";
#elif JKF_CUSTOM_FFT_PHASED_SCALAR_LOAD
      "custom_fft_f25_phased_scalar_swizzled_endpoint";
#elif JKF_CUSTOM_FFT_PHASED_PLANAR_SHARED
      "custom_fft_f24_phased_planar_swizzled_endpoint";
#elif JKF_CUSTOM_FFT_SHARED16_ZERO_ROWS_FLAT
      "custom_fft_f23_flat_zero_row_swizzled_endpoint";
#elif JKF_CUSTOM_FFT_SHARED16_SKIP_ZERO_ROWS
      "custom_fft_f22_shared16_zero_row_elided_endpoint";
#elif JKF_CUSTOM_FFT_ROWS8_GRID2D
      "custom_fft_f21_grid2d_zero_row_dual_swizzle_endpoint";
#elif JKF_CUSTOM_FFT_ROWS8_SKIP_ZERO_ROWS
      "custom_fft_f20_zero_row_elided_dual_swizzle_endpoint";
#elif JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_RAW_XOR
      "custom_fft_f19_raw_xor_dual_swizzle_endpoint";
#elif JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_ADD
      "custom_fft_f18_add_folded_dual_swizzle_endpoint";
#elif JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_FAST
      "custom_fft_f17_fast_dual_swizzled_endpoint";
#elif JKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE
      "custom_fft_f16_dual_swizzled_endpoint";
#elif JKF_CUSTOM_FFT_ROWS8_INTERLEAVED_TRANSPOSED
      "custom_fft_f15_interleaved_regular_x_swizzled_y_endpoint";
#elif JKF_CUSTOM_FFT_ROWS8_DIRECT_TRANSPOSED
      "custom_fft_f14_regular_x_aligned_pair_y_endpoint";
#elif JKF_CUSTOM_FFT_ALIGNED_PAIR_SWIZZLE
      "custom_fft_f13_aligned_pair_swizzle_endpoint";
#elif JKF_CUSTOM_FFT_XOR_SWIZZLE
      "custom_fft_f12_xor_swizzle_endpoint";
#elif JKF_CUSTOM_FFT_PLANAR_SHARED
      "custom_fft_f11_planar_frequency_pad_endpoint";
#elif JKF_CUSTOM_FFT_X_DIRECT_TRANSPOSED
      "custom_fft_f9_direct_transpose_endpoint";
#elif JKF_CUSTOM_FFT_X_TRANSPOSED
      "custom_fft_f8_producer_transpose_endpoint";
#elif JKF_CUSTOM_FFT_Y_SHARED16
      "custom_fft_f7_shared16_xy_endpoint";
#elif JKF_CUSTOM_FFT_SHARED16
      "custom_fft_f6_shared16_endpoint";
#elif JKF_CUSTOM_FFT_VECTOR32
#if JKF_CUSTOM_FFT_SPECIAL16
      "custom_fft_f5_special16_shuffle_endpoint";
#elif JKF_CUSTOM_FFT_MIXED32
      "custom_fft_f4_mixed32_shuffle_endpoint";
#else
      "custom_fft_f3_vector32_shuffle_endpoint";
#endif
#elif JKF_CUSTOM_FFT_ROWS8
      "custom_fft_f2_rows8_columns8_endpoint";
#elif JKF_CUSTOM_FFT_FUSED_COLUMNS
      "custom_fft_radix8_fused_columns_endpoint";
#else
      "custom_fft_radix8_endpoint";
#endif
#else
      "cufft_forward_endpoint";
#endif
  std::printf(
      "{\"variant\":\"%s\",\"correct\":true,\"us\":%.9g,"
      "\"grid_max_abs\":%.9g,\"grid_rel_l2\":%.9g,"
      "\"panel_max_abs\":%.9g,\"panel_rel_l2\":%.9g,"
      "\"warmup\":%d,\"iters\":%d,\"stress\":%d}\n",
      variant, latency_ms * 1000.0, grid_error.max_abs, grid_error.rel_l2,
      panel_error.max_abs, panel_error.rel_l2, warmup, iterations, stress);
  std::printf(
      "ALPHA_OPS_RESULT {\"status\":\"pass\","
      "\"correctness_status\":\"pass\",\"variant\":\"%s\","
      "\"alpha_latency_ms\":%.12g,\"grid_rel_l2\":%.9g,"
      "\"panel_rel_l2\":%.9g}\n",
      variant, latency_ms, grid_error.rel_l2, panel_error.rel_l2);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cufftDestroy(plan);
  cudaStreamDestroy(stream);
  cudaFree(candidate_panel);
  cudaFree(baseline_panel);
  cudaFree(candidate_grid);
  cudaFree(temp2);
  cudaFree(temp1);
  cudaFree(temp0);
  cudaFree(baseline_grid);
  cudaFree(grid_input);
  cudaFree(scaling);
  cudaFree(image);
  return 0;
}
#endif
