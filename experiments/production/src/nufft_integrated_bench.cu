// One-stream forward plus adjoint NUFFT component pipeline.

#ifndef JKF_VARIANT
#define JKF_VARIANT 5
#endif

#ifndef JKF_FUSED_DIRECT_COMPLEX_OUTPUT
#define JKF_FUSED_DIRECT_COMPLEX_OUTPUT 1
#endif

#ifndef JKF_CUSTOM_FORWARD_F30
#define JKF_CUSTOM_FORWARD_F30 0
#endif

#ifndef JKF_CUSTOM_INVERSE_F31
#define JKF_CUSTOM_INVERSE_F31 0
#endif

#ifndef JKF_CUSTOM_INVERSE_REGULAR_X
#define JKF_CUSTOM_INVERSE_REGULAR_X 0
#endif

#ifndef JKF_CUSTOM_INVERSE_WARP_LOCAL_CROP
#define JKF_CUSTOM_INVERSE_WARP_LOCAL_CROP 0
#endif

#ifndef JKF_CUSTOM_INVERSE_DIRECT_REGULAR_LOAD
#define JKF_CUSTOM_INVERSE_DIRECT_REGULAR_LOAD 0
#endif

#ifndef JKF_CUSTOM_INVERSE_PRENORMALIZED
#define JKF_CUSTOM_INVERSE_PRENORMALIZED 0
#endif

#ifndef JKF_CUSTOM_INVERSE_SAFE_CROP
#define JKF_CUSTOM_INVERSE_SAFE_CROP 0
#endif

#ifndef JKF_CUSTOM_INVERSE_REGULAR_Y_CROP
#define JKF_CUSTOM_INVERSE_REGULAR_Y_CROP 0
#endif

#ifndef JKF_DISABLE_CUFFT_RUNTIME
#define JKF_DISABLE_CUFFT_RUNTIME 0
#endif

#ifndef JKF_FP16X2
#define JKF_FP16X2 0
#endif

#ifndef JKF_FP16X2_DISABLE_DEBUG_ENDPOINT
#define JKF_FP16X2_DISABLE_DEBUG_ENDPOINT 0
#endif

#ifndef JKF_DENSE_TC_CONTROL
#define JKF_DENSE_TC_CONTROL 0
#endif

#if JKF_FP16X2 && (!JKF_CUSTOM_FORWARD_F30 || !JKF_FUSED_DIRECT_COMPLEX_OUTPUT)
#error "FP16x2 requires custom F30 and direct complex output"
#endif

#if JKF_DENSE_TC_CONTROL && !JKF_FUSED_DIRECT_COMPLEX_OUTPUT
#error "Dense Tensor Core control requires direct complex output"
#endif

// 0: independent forward+adjoint keeper, 1: materialized data consistency,
// 2: data consistency fused into the forward Sparse-MMA epilogue.
#ifndef JKF_DATA_CONSISTENCY_MODE
#define JKF_DATA_CONSISTENCY_MODE 0
#endif

// 0: caller provides coil images; 1: materialize x*sensitivity maps;
// 2: fuse x*sensitivity maps into the forward FFT row producer.
#ifndef JKF_SENSE_FORWARD_MODE
#define JKF_SENSE_FORWARD_MODE 0
#endif

#if JKF_DATA_CONSISTENCY_MODE < 0 || JKF_DATA_CONSISTENCY_MODE > 2
#error "JKF_DATA_CONSISTENCY_MODE must be 0, 1, or 2"
#endif

#if JKF_DATA_CONSISTENCY_MODE != 0 && !JKF_FUSED_DIRECT_COMPLEX_OUTPUT
#error "data-consistency modes require direct complex output"
#endif

#if JKF_SENSE_FORWARD_MODE < 0 || JKF_SENSE_FORWARD_MODE > 2
#error "JKF_SENSE_FORWARD_MODE must be 0, 1, or 2"
#endif

#if JKF_SENSE_FORWARD_MODE != 0 && !JKF_CUSTOM_FORWARD_F30
#error "SENSE forward modes require the custom F30 forward endpoint"
#endif

#define JKF_NO_COMPONENT_MAIN
#include "nufft_spmma_bench.cu"

#include <cuda_profiler_api.h>
#include <cufft.h>

#if JKF_CUSTOM_FORWARD_F30 || JKF_CUSTOM_INVERSE_F31
#include "nufft_custom_fft_f30_api.h"
#endif

#ifndef JKF_SYSTEM_PROFILE
#define JKF_SYSTEM_PROFILE 0
#endif

namespace {

constexpr int kImage = 256;
constexpr int kGrid = 512;
constexpr int kCoils = 8;

#define CUFFT_CHECK(expr)                                                       \
  do {                                                                         \
    cufftResult status_ = (expr);                                               \
    if (status_ != CUFFT_SUCCESS) {                                             \
      std::fprintf(stderr, "cuFFT failure %s:%d: status=%d\n", __FILE__,      \
                   __LINE__, static_cast<int>(status_));                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct PackedDirection {
  Header h{};
  DeviceData d{};
  half* dense_residual = nullptr;
#if JKF_DENSE_TC_CONTROL
  half* dense_a_real = nullptr;
  half* dense_a_imag = nullptr;
#endif
};

PackedDirection load_direction(const std::string& real_directory,
                               const std::string& imag_directory,
                               const std::string& dense_control_directory = "") {
  PackedDirection packed;
  packed.h = read_header(real_directory);
  const Header imag_h = read_header(imag_directory);
  if (std::memcmp(&packed.h, &imag_h, sizeof(Header)) != 0 ||
      packed.h.residual_nnz != 0) {
    std::fprintf(stderr, "incompatible real/imag packed direction\n");
    std::exit(2);
  }

  const auto group_offsets = read_array<uint32_t>(
      real_directory + "/group_offsets.bin", packed.h.groups + 1);
  const auto group_row_ids = read_array<int32_t>(
      real_directory + "/group_row_ids.bin",
      static_cast<size_t>(packed.h.groups) * kM);
  const auto tile_col_ids = read_array<int32_t>(
      real_directory + "/tile_col_ids.bin",
      static_cast<size_t>(packed.h.tiles) * kK);
  const auto real_a = read_array<uint16_t>(
      real_directory + "/tile_a_comp.bin",
      static_cast<size_t>(packed.h.tiles) * kM * kKComp);
  const auto real_meta = read_array<uint32_t>(
      real_directory + "/tile_pair_meta.bin",
      static_cast<size_t>(packed.h.tiles) * kM);

  const auto imag_group_offsets = read_array<uint32_t>(
      imag_directory + "/group_offsets.bin", packed.h.groups + 1);
  const auto imag_group_row_ids = read_array<int32_t>(
      imag_directory + "/group_row_ids.bin",
      static_cast<size_t>(packed.h.groups) * kM);
  const auto imag_tile_col_ids = read_array<int32_t>(
      imag_directory + "/tile_col_ids.bin",
      static_cast<size_t>(packed.h.tiles) * kK);
  const auto imag_a = read_array<uint16_t>(
      imag_directory + "/tile_a_comp.bin",
      static_cast<size_t>(packed.h.tiles) * kM * kKComp);
  const auto imag_meta = read_array<uint32_t>(
      imag_directory + "/tile_pair_meta.bin",
      static_cast<size_t>(packed.h.tiles) * kM);
#if JKF_DENSE_TC_CONTROL
  if (dense_control_directory.empty()) {
    std::fprintf(stderr, "dense control directory is required\n");
    std::exit(2);
  }
  const auto dense_a_real = read_array<uint16_t>(
      dense_control_directory + "/real.f16.bin",
      static_cast<size_t>(packed.h.tiles) * kM * kK);
  const auto dense_a_imag = read_array<uint16_t>(
      dense_control_directory + "/imag.f16.bin",
      static_cast<size_t>(packed.h.tiles) * kM * kK);
#endif

  if (group_offsets != imag_group_offsets ||
      group_row_ids != imag_group_row_ids ||
      tile_col_ids != imag_tile_col_ids) {
    std::fprintf(stderr, "real/imag structures differ\n");
    std::exit(2);
  }

  packed.d.group_offsets = device_copy(group_offsets);
  packed.d.group_row_ids = device_copy(group_row_ids);
  packed.d.tile_col_ids = device_copy(tile_col_ids);
  packed.d.tile_a_comp = reinterpret_cast<half*>(device_copy(real_a));
  packed.d.tile_pair_meta = device_copy(real_meta);
  packed.d.tile_a_imag_comp = reinterpret_cast<half*>(device_copy(imag_a));
  packed.d.tile_imag_pair_meta = device_copy(imag_meta);
#if JKF_DENSE_TC_CONTROL
  packed.dense_a_real =
      reinterpret_cast<half*>(device_copy(dense_a_real));
  packed.dense_a_imag =
      reinterpret_cast<half*>(device_copy(dense_a_imag));
#endif
  CUDA_CHECK(cudaMalloc(&packed.d.dense,
                        static_cast<size_t>(packed.h.n_cols) * kN *
                            sizeof(half)));
#if JKF_FP16X2
  CUDA_CHECK(cudaMalloc(&packed.dense_residual,
                        static_cast<size_t>(packed.h.n_cols) * kN *
                            sizeof(half)));
#endif
  const size_t output_bytes =
      static_cast<size_t>(packed.h.output_size) * kN * sizeof(float);
  CUDA_CHECK(cudaMalloc(&packed.d.output, output_bytes));
  CUDA_CHECK(cudaMalloc(&packed.d.real_output, output_bytes));
  CUDA_CHECK(cudaMalloc(&packed.d.imag_output, output_bytes));
  CUDA_CHECK(cudaMemset(packed.d.output, 0, output_bytes));
  CUDA_CHECK(cudaMemset(packed.d.real_output, 0, output_bytes));
  CUDA_CHECK(cudaMemset(packed.d.imag_output, 0, output_bytes));
  return packed;
}

void free_direction(PackedDirection& packed) {
  cudaFree(packed.d.group_offsets);
  cudaFree(packed.d.group_row_ids);
  cudaFree(packed.d.tile_col_ids);
  cudaFree(packed.d.tile_a_comp);
  cudaFree(packed.d.tile_pair_meta);
  cudaFree(packed.d.tile_a_imag_comp);
  cudaFree(packed.d.tile_imag_pair_meta);
#if JKF_DENSE_TC_CONTROL
  cudaFree(packed.dense_a_real);
  cudaFree(packed.dense_a_imag);
#endif
  cudaFree(packed.d.dense);
#if JKF_FP16X2
  cudaFree(packed.dense_residual);
#endif
  cudaFree(packed.d.output);
  cudaFree(packed.d.real_output);
  cudaFree(packed.d.imag_output);
}

#if JKF_DENSE_TC_CONTROL
#include "nufft_dense_control.cuh"
#endif

template <typename T>
void write_device_array(const std::string& path,
                        const T* device_data,
                        size_t elements) {
  std::vector<T> host(elements);
  CUDA_CHECK(cudaMemcpy(host.data(), device_data, elements * sizeof(T),
                        cudaMemcpyDeviceToHost));
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    std::fprintf(stderr, "failed to open output file %s\n", path.c_str());
    std::exit(2);
  }
  const size_t written = std::fwrite(host.data(), sizeof(T), elements, file);
  const int close_status = std::fclose(file);
  if (written != elements || close_status != 0) {
    std::fprintf(stderr, "failed to write output file %s\n", path.c_str());
    std::exit(2);
  }
}

__global__ void integrated_pad_scale(const float2* __restrict__ image,
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

__global__ void integrated_sense_expand(
    const float2* __restrict__ image,
    const float2* __restrict__ sensitivity_maps,
    float2* __restrict__ coil_images) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total = kCoils * kImage * kImage;
  if (index >= total) return;
  const uint32_t pixel = index % (kImage * kImage);
  const float2 value = image[pixel];
  const float2 sensitivity = sensitivity_maps[index];
  coil_images[index] = make_float2(
      value.x * sensitivity.x - value.y * sensitivity.y,
      value.x * sensitivity.y + value.y * sensitivity.x);
}

__global__ void integrated_crop_scale(const float2* __restrict__ grid,
                                      const float* __restrict__ scaling,
                                      float2* __restrict__ image) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total = kCoils * kImage * kImage;
  if (index >= total) return;
  const int x = index % kImage;
  const int y = (index / kImage) % kImage;
  const int coil = index / (kImage * kImage);
  float2 value = grid[(coil * kGrid + y) * kGrid + x];
  const float scale =
      scaling[y * kImage + x] / static_cast<float>(kGrid * kGrid);
  value.x *= scale;
  value.y *= scale;
  image[index] = value;
}

__global__ void integrated_complex_to_half_rows(
    const float2* __restrict__ input,
    uint32_t rows,
    half* __restrict__ output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t count = rows * kCoils;
  if (index >= count) return;
  const uint32_t row = index / kCoils;
  const uint32_t coil = index % kCoils;
  const float2 value = input[static_cast<size_t>(coil) * rows + row];
  output[static_cast<size_t>(row) * kN + coil] = __float2half_rn(value.x);
  output[static_cast<size_t>(row) * kN + coil + kCoils] =
      __float2half_rn(value.y);
}

__global__ void integrated_complex_to_split2_half_rows(
    const float2* __restrict__ input,
    uint32_t rows,
    half* __restrict__ high_output,
    half* __restrict__ residual_output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t count = rows * kCoils;
  if (index >= count) return;
  const uint32_t row = index / kCoils;
  const uint32_t coil = index % kCoils;
  const float2 value = input[static_cast<size_t>(coil) * rows + row];
  const half real_high = __float2half_rn(value.x);
  const half imag_high = __float2half_rn(value.y);
  const size_t base = static_cast<size_t>(row) * kN;
  high_output[base + coil] = real_high;
  high_output[base + coil + kCoils] = imag_high;
  residual_output[base + coil] =
      __float2half_rn(value.x - __half2float(real_high));
  residual_output[base + coil + kCoils] =
      __float2half_rn(value.y - __half2float(imag_high));
}

__global__ void integrated_float_rows_to_complex(
    const float* __restrict__ input,
    uint32_t rows,
    float2* __restrict__ output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t count = rows * kCoils;
  if (index >= count) return;
  const uint32_t row = index / kCoils;
  const uint32_t coil = index % kCoils;
  const size_t base = static_cast<size_t>(row) * kN;
  output[static_cast<size_t>(coil) * rows + row] =
      make_float2(input[base + coil], input[base + coil + kCoils]);
}

__global__ void integrated_residual_to_half_rows(
    const float2* __restrict__ prediction,
    const float2* __restrict__ measurement,
    const float* __restrict__ density,
    uint32_t rows,
    half* __restrict__ output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t pairs_per_row = kCoils / 2;
  const uint32_t count = rows * pairs_per_row;
  if (index >= count) return;
  const uint32_t row = index / pairs_per_row;
  const uint32_t coil0 = (index % pairs_per_row) * 2;
  const uint32_t coil1 = coil0 + 1;
  const float2 prediction0 =
      prediction[static_cast<size_t>(coil0) * rows + row];
  const float2 prediction1 =
      prediction[static_cast<size_t>(coil1) * rows + row];
  const float2 measurement0 =
      measurement[static_cast<size_t>(coil0) * rows + row];
  const float2 measurement1 =
      measurement[static_cast<size_t>(coil1) * rows + row];
  const float weight = density[row];
  const half2 residual_real = __floats2half2_rn(
      (prediction0.x - measurement0.x) * weight,
      (prediction1.x - measurement1.x) * weight);
  const half2 residual_imag = __floats2half2_rn(
      (prediction0.y - measurement0.y) * weight,
      (prediction1.y - measurement1.y) * weight);
  *reinterpret_cast<half2*>(
      output + static_cast<size_t>(row) * kN + coil0) = residual_real;
  *reinterpret_cast<half2*>(
      output + static_cast<size_t>(row) * kN + coil0 + kCoils) = residual_imag;
}

struct IntegratedBuffers {
  float2* forward_object = nullptr;
  float2* sensitivity_maps = nullptr;
  float2* forward_image = nullptr;
  float2* forward_samples = nullptr;
  float2* forward_reference = nullptr;
  float2* adjoint_samples = nullptr;
  float2* adjoint_image = nullptr;
  float2* grid_input = nullptr;
  float2* grid_output = nullptr;
  float2* adjoint_grid_input = nullptr;
  float* forward_scaling = nullptr;
  float* inverse_scaling = nullptr;
  float* scaling_tiled = nullptr;
  float* density = nullptr;
  half* dc_dense_reference = nullptr;
  float2* dc_image_reference = nullptr;
  float2* inverse_reference = nullptr;
  half* forward_dense_reference = nullptr;
  float2* forward_fp32_endpoint = nullptr;
};

void launch_fused_interpolation(PackedDirection& packed,
                                cudaStream_t stream,
                                float* output_override = nullptr,
                                bool conjugate = false,
                                float output_scale = 1.0f) {
#if JKF_DENSE_TC_CONTROL
  float* output = output_override != nullptr ? output_override : packed.d.output;
  launch_dense_control_dispatch(
      packed, packed.d.dense, reinterpret_cast<float2*>(output), conjugate,
      false, output_scale, stream);
#else
  DeviceData launch_data = packed.d;
  if (output_override != nullptr) launch_data.output = output_override;
  launch_variant(packed.h, launch_data, stream, conjugate, false,
                 output_scale);
#endif
}

void launch_fused_interpolation_residual(PackedDirection& packed,
                                         cudaStream_t stream,
                                         float* output_override,
                                         bool conjugate,
                                         float output_scale = 1.0f) {
#if JKF_DENSE_TC_CONTROL
  float* output = output_override != nullptr ? output_override : packed.d.output;
  launch_dense_control_dispatch(
      packed, packed.dense_residual, reinterpret_cast<float2*>(output),
      conjugate, true, output_scale, stream);
#else
  DeviceData launch_data = packed.d;
  launch_data.dense = packed.dense_residual;
  launch_data.output = output_override;
  launch_variant(packed.h, launch_data, stream, conjugate, true,
                 output_scale);
#endif
}

void launch_unfused_interpolation(PackedDirection& packed,
                                  cudaStream_t stream,
                                  bool conjugate = false) {
  constexpr int kWarps = 4;
  constexpr size_t kSharedBytes = kWarps * kM * kN * sizeof(float);
  nufft_gt_spmma_split_f16_f32<kWarps>
      <<<packed.h.groups, kWarps * 32, kSharedBytes, stream>>>(
      packed.d.group_offsets, packed.d.group_row_ids, packed.d.tile_col_ids,
      packed.d.tile_a_comp, packed.d.tile_pair_meta, packed.d.dense,
      packed.d.real_output);
  nufft_gt_spmma_split_f16_f32<kWarps>
      <<<packed.h.groups, kWarps * 32, kSharedBytes, stream>>>(
      packed.d.group_offsets, packed.d.group_row_ids, packed.d.tile_col_ids,
      packed.d.tile_a_imag_comp, packed.d.tile_imag_pair_meta, packed.d.dense,
      packed.d.imag_output);
  constexpr int kThreads = 256;
  const uint32_t blocks =
      (packed.h.output_size * kCoils + kThreads - 1) / kThreads;
  if (conjugate) {
    complex_combine_f32<true><<<blocks, kThreads, 0, stream>>>(
        packed.d.real_output, packed.d.imag_output, packed.h.output_size,
        packed.d.output);
  } else {
    complex_combine_f32<false><<<blocks, kThreads, 0, stream>>>(
        packed.d.real_output, packed.d.imag_output, packed.h.output_size,
        packed.d.output);
  }
}

void launch_data_consistency_fused(PackedDirection& forward,
                                   const IntegratedBuffers& buffers,
                                   half* adjoint_dense,
                                   cudaStream_t stream) {
  constexpr int kWarps = 4;
  nufft_gt_spmma_complex_fused_f16_f32<kWarps, true>
      <<<forward.h.groups, kWarps * 32,
         2 * kWarps * kM * kN * sizeof(float), stream>>>(
          forward.d.group_offsets, forward.d.group_row_ids,
          forward.d.tile_col_ids, forward.d.tile_a_comp,
          forward.d.tile_pair_meta, forward.d.tile_a_imag_comp,
          forward.d.tile_imag_pair_meta, forward.d.dense,
          forward.h.output_size, buffers.adjoint_samples, buffers.density,
          adjoint_dense, nullptr, 1.0f);
}

void launch_data_consistency_materialized(PackedDirection& forward,
                                          const IntegratedBuffers& buffers,
                                          half* adjoint_dense,
                                          cudaStream_t stream) {
  constexpr int kThreads = 256;
  launch_fused_interpolation(
      forward, stream, reinterpret_cast<float*>(buffers.forward_samples));
  const uint32_t pairs = forward.h.output_size * (kCoils / 2);
  integrated_residual_to_half_rows
      <<<(pairs + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          buffers.forward_samples, buffers.adjoint_samples, buffers.density,
          forward.h.output_size, adjoint_dense);
}

void prepare_forward_dense(PackedDirection& forward,
                           const IntegratedBuffers& buffers,
                           cufftHandle plan,
                           cudaStream_t stream) {
#if JKF_CUSTOM_FORWARD_F30
  (void)plan;
#if JKF_SENSE_FORWARD_MODE == 1
  constexpr int kThreads = 256;
  const uint32_t image_elements = kCoils * kImage * kImage;
  integrated_sense_expand
      <<<(image_elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          buffers.forward_object, buffers.sensitivity_maps,
          buffers.forward_image);
#if JKF_FP16X2
  jkf_launch_custom_fft_f30_forward_split2(
      buffers.forward_image, buffers.forward_scaling, buffers.grid_input,
      forward.d.dense, forward.dense_residual, stream);
#else
  jkf_launch_custom_fft_f30_forward(
      buffers.forward_image, buffers.forward_scaling, buffers.grid_input,
      forward.d.dense, stream);
#endif
#elif JKF_SENSE_FORWARD_MODE == 2
#if JKF_FP16X2
  jkf_launch_custom_fft_f30_forward_sense_split2(
      buffers.forward_object, buffers.sensitivity_maps, buffers.forward_scaling,
      buffers.grid_input, forward.d.dense, forward.dense_residual, stream);
#else
  jkf_launch_custom_fft_f30_forward_sense(
      buffers.forward_object, buffers.sensitivity_maps, buffers.forward_scaling,
      buffers.grid_input, forward.d.dense, stream);
#endif
#else
#if JKF_FP16X2
  jkf_launch_custom_fft_f30_forward_split2(
      buffers.forward_image, buffers.forward_scaling, buffers.grid_input,
      forward.d.dense, forward.dense_residual, stream);
#else
  jkf_launch_custom_fft_f30_forward(
      buffers.forward_image, buffers.forward_scaling, buffers.grid_input,
      forward.d.dense, stream);
#endif
#endif
#else
  constexpr int kThreads = 256;
  const uint32_t grid_elements = kCoils * kGrid * kGrid;
  integrated_pad_scale<<<(grid_elements + kThreads - 1) / kThreads, kThreads,
                         0, stream>>>(buffers.forward_image, buffers.forward_scaling,
                                     buffers.grid_input);
  CUFFT_CHECK(cufftExecC2C(plan,
                           reinterpret_cast<cufftComplex*>(buffers.grid_input),
                           reinterpret_cast<cufftComplex*>(buffers.grid_output),
                           CUFFT_FORWARD));
  const uint32_t elements = forward.h.n_cols * kCoils;
  integrated_complex_to_half_rows
      <<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          buffers.grid_output, forward.h.n_cols, forward.d.dense);
#endif
}

void prepare_adjoint_dense(PackedDirection& adjoint,
                           const IntegratedBuffers& buffers,
                           cudaStream_t stream) {
  constexpr int kThreads = 256;
  const uint32_t elements = adjoint.h.n_cols * kCoils;
#if JKF_FP16X2
  integrated_complex_to_split2_half_rows
      <<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          buffers.adjoint_samples, adjoint.h.n_cols, adjoint.d.dense,
          adjoint.dense_residual);
#else
  integrated_complex_to_half_rows
      <<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          buffers.adjoint_samples, adjoint.h.n_cols, adjoint.d.dense);
#endif
}

void launch_integrated_once(PackedDirection& forward,
                            PackedDirection& adjoint,
                            const IntegratedBuffers& buffers,
                            cufftHandle plan,
                            cudaStream_t stream) {
  constexpr int kThreads = 256;
  prepare_forward_dense(forward, buffers, plan, stream);
#if JKF_FUSED_DIRECT_COMPLEX_OUTPUT
#if JKF_FP16X2
  constexpr float kForwardOutputScale = 1.0f / static_cast<float>(kGrid);
  launch_fused_interpolation(
      forward, stream, reinterpret_cast<float*>(buffers.forward_samples), false,
      kForwardOutputScale);
  launch_fused_interpolation_residual(
      forward, stream, reinterpret_cast<float*>(buffers.forward_samples), false,
      kForwardOutputScale);
#else
  launch_fused_interpolation(
      forward, stream, reinterpret_cast<float*>(buffers.forward_samples));
#endif
#else
  launch_fused_interpolation(forward, stream);
  const uint32_t forward_elements = forward.h.output_size * kCoils;
  integrated_float_rows_to_complex
      <<<(forward_elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          forward.d.output, forward.h.output_size, buffers.forward_samples);
#endif

  prepare_adjoint_dense(adjoint, buffers, stream);
#if JKF_FUSED_DIRECT_COMPLEX_OUTPUT
  launch_fused_interpolation(
      adjoint, stream, reinterpret_cast<float*>(buffers.adjoint_grid_input),
      true);
#if JKF_FP16X2
  launch_fused_interpolation_residual(
      adjoint, stream, reinterpret_cast<float*>(buffers.adjoint_grid_input),
      true);
#endif
#else
  launch_fused_interpolation(adjoint, stream, nullptr, true);
  const uint32_t adjoint_elements = adjoint.h.output_size * kCoils;
  integrated_float_rows_to_complex
      <<<(adjoint_elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          adjoint.d.output, adjoint.h.output_size, buffers.grid_input);
#endif
#if JKF_CUSTOM_INVERSE_F31
  (void)plan;
  jkf_launch_custom_ifft_f31_adjoint(
#if JKF_FUSED_DIRECT_COMPLEX_OUTPUT
      buffers.adjoint_grid_input,
#else
      buffers.grid_input,
#endif
      buffers.grid_output, buffers.inverse_scaling, buffers.scaling_tiled,
      buffers.adjoint_image, stream);
#else
  CUFFT_CHECK(cufftExecC2C(plan,
#if JKF_FUSED_DIRECT_COMPLEX_OUTPUT
                           reinterpret_cast<cufftComplex*>(
                               buffers.adjoint_grid_input),
#else
                           reinterpret_cast<cufftComplex*>(buffers.grid_input),
#endif
                           reinterpret_cast<cufftComplex*>(buffers.grid_output),
                           CUFFT_INVERSE));
  const uint32_t image_elements = kCoils * kImage * kImage;
  integrated_crop_scale<<<(image_elements + kThreads - 1) / kThreads, kThreads,
                          0, stream>>>(buffers.grid_output, buffers.inverse_scaling,
                                      buffers.adjoint_image);
#endif
}

void launch_data_consistency_adjoint(PackedDirection& adjoint,
                                     const IntegratedBuffers& buffers,
                                     cufftHandle plan,
                                     cudaStream_t stream) {
  constexpr int kThreads = 256;
  launch_fused_interpolation(
      adjoint, stream, reinterpret_cast<float*>(buffers.adjoint_grid_input),
      true);
#if JKF_CUSTOM_INVERSE_F31
  (void)plan;
  jkf_launch_custom_ifft_f31_adjoint(
      buffers.adjoint_grid_input, buffers.grid_output, buffers.inverse_scaling,
      buffers.scaling_tiled, buffers.adjoint_image, stream);
#else
  CUFFT_CHECK(cufftExecC2C(
      plan, reinterpret_cast<cufftComplex*>(buffers.adjoint_grid_input),
      reinterpret_cast<cufftComplex*>(buffers.grid_output), CUFFT_INVERSE));
  const uint32_t image_elements = kCoils * kImage * kImage;
  integrated_crop_scale<<<(image_elements + kThreads - 1) / kThreads, kThreads,
                          0, stream>>>(buffers.grid_output, buffers.inverse_scaling,
                                      buffers.adjoint_image);
#endif
}

#if JKF_DATA_CONSISTENCY_MODE != 0
void launch_data_consistency_once(PackedDirection& forward,
                                  PackedDirection& adjoint,
                                  const IntegratedBuffers& buffers,
                                  cufftHandle plan,
                                  cudaStream_t stream) {
  prepare_forward_dense(forward, buffers, plan, stream);
#if JKF_DATA_CONSISTENCY_MODE == 1
  launch_data_consistency_materialized(forward, buffers, adjoint.d.dense,
                                       stream);
#elif JKF_DATA_CONSISTENCY_MODE == 2
  launch_data_consistency_fused(forward, buffers, adjoint.d.dense, stream);
#endif
  launch_data_consistency_adjoint(adjoint, buffers, plan, stream);
}
#endif

struct ErrorStats {
  float max_abs = 0.0f;
  double rel_l2 = 0.0;
  size_t nonfinite = 0;
  uint32_t max_ulp = 0;
  float max_scaled_abs = 0.0f;
};

ErrorStats compare_device(const float* candidate,
                          const float* reference,
                          size_t elements) {
  std::vector<float> candidate_host(elements);
  std::vector<float> reference_host(elements);
  CUDA_CHECK(cudaMemcpy(candidate_host.data(), candidate,
                        elements * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(reference_host.data(), reference,
                        elements * sizeof(float), cudaMemcpyDeviceToHost));
  ErrorStats stats;
  double error_square = 0.0;
  double ref_square = 0.0;
  for (size_t i = 0; i < elements; ++i) {
    if (!std::isfinite(candidate_host[i])) ++stats.nonfinite;
    const float diff = std::fabs(candidate_host[i] - reference_host[i]);
    stats.max_abs = std::max(stats.max_abs, diff);
    error_square += static_cast<double>(diff) * diff;
    ref_square += static_cast<double>(reference_host[i]) * reference_host[i];
  }
  stats.rel_l2 = std::sqrt(error_square / std::max(ref_square, 1e-30));
  return stats;
}

ErrorStats compare_half_device(const half* candidate,
                               const half* reference,
                               size_t elements,
                               size_t* bit_mismatches) {
  std::vector<half> candidate_host(elements);
  std::vector<half> reference_host(elements);
  CUDA_CHECK(cudaMemcpy(candidate_host.data(), candidate,
                        elements * sizeof(half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(reference_host.data(), reference,
                        elements * sizeof(half), cudaMemcpyDeviceToHost));
  ErrorStats stats;
  double error_square = 0.0;
  double ref_square = 0.0;
  *bit_mismatches = 0;
  for (size_t i = 0; i < elements; ++i) {
    const uint16_t candidate_bits = __half_as_ushort(candidate_host[i]);
    const uint16_t reference_bits = __half_as_ushort(reference_host[i]);
    if (candidate_bits != reference_bits) ++*bit_mismatches;
    const uint32_t candidate_ordered =
        (candidate_bits & 0x8000u) ? (0x8000u - (candidate_bits & 0x7fffu))
                                   : (0x8000u + candidate_bits);
    const uint32_t reference_ordered =
        (reference_bits & 0x8000u) ? (0x8000u - (reference_bits & 0x7fffu))
                                   : (0x8000u + reference_bits);
    stats.max_ulp =
        std::max(stats.max_ulp,
                 candidate_ordered > reference_ordered
                     ? candidate_ordered - reference_ordered
                     : reference_ordered - candidate_ordered);
    const float candidate_value = __half2float(candidate_host[i]);
    const float reference_value = __half2float(reference_host[i]);
    if (!std::isfinite(candidate_value)) ++stats.nonfinite;
    const float diff = std::fabs(candidate_value - reference_value);
    stats.max_abs = std::max(stats.max_abs, diff);
    stats.max_scaled_abs =
        std::max(stats.max_scaled_abs,
                 diff / std::max(1.0f, std::fabs(reference_value)));
    error_square += static_cast<double>(diff) * diff;
    ref_square += static_cast<double>(reference_value) * reference_value;
  }
  stats.rel_l2 = std::sqrt(error_square / std::max(ref_square, 1e-30));
  return stats;
}

size_t count_device_bit_mismatches(const void* candidate,
                                   const void* reference,
                                   size_t words) {
  std::vector<uint32_t> candidate_host(words);
  std::vector<uint32_t> reference_host(words);
  CUDA_CHECK(cudaMemcpy(candidate_host.data(), candidate,
                        words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(reference_host.data(), reference,
                        words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t i = 0; i < words; ++i) {
    if (candidate_host[i] != reference_host[i]) ++mismatches;
  }
  return mismatches;
}

size_t count_nonfinite_complex(const float2* device, size_t elements) {
  std::vector<float2> host(elements);
  CUDA_CHECK(cudaMemcpy(host.data(), device, elements * sizeof(float2),
                        cudaMemcpyDeviceToHost));
  size_t nonfinite = 0;
  for (const float2 value : host) {
    if (!std::isfinite(value.x) || !std::isfinite(value.y)) ++nonfinite;
  }
  return nonfinite;
}

const char* integrated_variant_name() {
#if JKF_DATA_CONSISTENCY_MODE == 1
#if JKF_CUSTOM_FORWARD_F30 && !JKF_DISABLE_CUFFT_RUNTIME
  return "dc_materialized_f30_forward";
#else
  return "dc_materialized";
#endif
#elif JKF_DATA_CONSISTENCY_MODE == 2
#if JKF_SENSE_FORWARD_MODE == 1
  return "dc_epilogue_fused_sense_materialized";
#elif JKF_SENSE_FORWARD_MODE == 2
  return "dc_epilogue_fused_sense_fft_fused";
#elif JKF_CUSTOM_FORWARD_F30
  return "dc_epilogue_fused_f30_forward";
#else
  return "dc_epilogue_fused";
#endif
#elif JKF_VARIANT == 5
#if JKF_FP16X2
  return "integrated_fused_w4_f30_f38_fp16x2";
#else
#if JKF_CUSTOM_FORWARD_F30 && JKF_CUSTOM_INVERSE_DIRECT_REGULAR_LOAD
#if JKF_DISABLE_CUFFT_RUNTIME
  return "integrated_fused_w4_f30_f36_allcustom_release";
#else
  return "integrated_fused_w4_f30_f36_allcustom";
#endif
#elif JKF_CUSTOM_FORWARD_F30 && JKF_CUSTOM_INVERSE_REGULAR_Y_CROP
#if JKF_DISABLE_CUFFT_RUNTIME
  return "integrated_fused_w4_f30_f38_allcustom_release";
#else
  return "integrated_fused_w4_f30_f38_allcustom";
#endif
#elif JKF_CUSTOM_FORWARD_F30 && JKF_CUSTOM_INVERSE_SAFE_CROP
#if JKF_DISABLE_CUFFT_RUNTIME
  return "integrated_fused_w4_f30_f37_allcustom_release";
#else
  return "integrated_fused_w4_f30_f37_allcustom";
#endif
#elif JKF_CUSTOM_FORWARD_F30 && JKF_CUSTOM_INVERSE_PRENORMALIZED
#if JKF_DISABLE_CUFFT_RUNTIME
  return "integrated_fused_w4_f30_f35_allcustom_release";
#else
  return "integrated_fused_w4_f30_f35_allcustom";
#endif
#elif JKF_CUSTOM_FORWARD_F30 && JKF_CUSTOM_INVERSE_WARP_LOCAL_CROP
#if JKF_DISABLE_CUFFT_RUNTIME
  return "integrated_fused_w4_f30_f34_allcustom_release";
#else
  return "integrated_fused_w4_f30_f34_allcustom";
#endif
#elif JKF_CUSTOM_FORWARD_F30 && JKF_CUSTOM_INVERSE_REGULAR_X
#if JKF_DISABLE_CUFFT_RUNTIME
  return "integrated_fused_w4_f30_f32_allcustom_release";
#else
  return "integrated_fused_w4_f30_f32_allcustom";
#endif
#elif JKF_CUSTOM_FORWARD_F30 && JKF_CUSTOM_INVERSE_F31
  return "integrated_fused_w4_f30_f31_allcustom";
#elif JKF_CUSTOM_FORWARD_F30
  return "integrated_fused_w4_f30_forward";
#else
  return "integrated_fused_w4";
#endif
#endif
#else
  return "integrated_unfused_w4";
#endif
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(
        stderr,
        "usage: %s FORWARD_REAL FORWARD_IMAG ADJOINT_REAL ADJOINT_IMAG "
        "[warmup=20] [iters=100] [stress=0] [graph=0] [seed=0] [churn=0] "
        "[runtime_input_dir] [dump_output_dir]\n",
        argv[0]);
    return 2;
  }
  const int warmup = argc > 5 ? std::atoi(argv[5]) : 20;
  const int iterations = argc > 6 ? std::atoi(argv[6]) : 100;
  const int stress = argc > 7 ? std::atoi(argv[7]) : 0;
  const bool graph_mode = argc > 8 ? std::atoi(argv[8]) != 0 : false;
  const uint32_t seed = argc > 9 ? static_cast<uint32_t>(std::strtoul(
                                      argv[9], nullptr, 10))
                                : 0u;
  const int pointer_churn = argc > 10 ? std::max(0, std::atoi(argv[10])) : 0;
  const bool runtime_input = argc > 11;
  const std::string runtime_input_directory = runtime_input ? argv[11] : "";
  const bool dump_output = argc > 12;
  const std::string dump_output_directory = dump_output ? argv[12] : "";
#if JKF_SENSE_FORWARD_MODE != 0
  if (!runtime_input) {
    std::fprintf(stderr, "SENSE forward mode requires runtime_input_dir\n");
    return 2;
  }
#endif

  PackedDirection forward = load_direction(argv[1], argv[2]);
  PackedDirection adjoint = load_direction(argv[3], argv[4]);
  if (forward.h.n_cols != static_cast<uint32_t>(kGrid * kGrid) ||
      adjoint.h.output_size != static_cast<uint32_t>(kGrid * kGrid) ||
      forward.h.output_size != adjoint.h.n_cols) {
    std::fprintf(stderr, "unexpected standard-256 operator shapes\n");
    return 2;
  }

  const size_t image_elements =
      static_cast<size_t>(kCoils) * kImage * kImage;
  const size_t grid_elements =
      static_cast<size_t>(kCoils) * kGrid * kGrid;
  const size_t sample_elements =
      static_cast<size_t>(kCoils) * forward.h.output_size;
  std::vector<float2> forward_image_host(image_elements);
  std::vector<float2> adjoint_samples_host(sample_elements);
  std::vector<float> scaling_host(kImage * kImage);
  std::vector<float> forward_scaling_host(kImage * kImage);
  std::vector<float> inverse_scaling_host(kImage * kImage);
  std::vector<float> scaling_tiled_host(kImage * kImage);
#if JKF_SENSE_FORWARD_MODE != 0
  std::vector<float2> forward_object_host(kImage * kImage);
  std::vector<float2> sensitivity_maps_host(image_elements);
#endif
#if JKF_DATA_CONSISTENCY_MODE != 0
  std::vector<float2> measurement_image_host(image_elements);
  std::vector<float> density_host(forward.h.output_size);
  constexpr float kDcInputScale = 1.0f / 1024.0f;
#else
  constexpr float kDcInputScale = 1.0f;
#endif
  if (runtime_input) {
    forward_image_host = read_array<float2>(
        runtime_input_directory + "/forward_coils.c64.bin", image_elements);
    adjoint_samples_host = read_array<float2>(
        runtime_input_directory + "/measurement.c64.bin", sample_elements);
    scaling_host = read_array<float>(
        runtime_input_directory + "/scaling.f32.bin", kImage * kImage);
#if JKF_SENSE_FORWARD_MODE != 0
    forward_object_host = read_array<float2>(
        runtime_input_directory + "/truth.c64.bin", kImage * kImage);
    sensitivity_maps_host = read_array<float2>(
        runtime_input_directory + "/sensitivity_maps.c64.bin", image_elements);
#endif
#if JKF_DATA_CONSISTENCY_MODE != 0
    density_host = read_array<float>(
        runtime_input_directory + "/density.f32.bin", forward.h.output_size);
#endif
  } else {
    for (size_t i = 0; i < image_elements; ++i) {
      forward_image_host[i] =
          make_float2(kDcInputScale *
                          static_cast<float>((i * 17 + 1 + seed * 101u) % 257) /
                          257.0f,
                      kDcInputScale *
                          static_cast<float>((i * 23 + 2 + seed * 103u) % 263) /
                          263.0f);
#if JKF_DATA_CONSISTENCY_MODE != 0
      measurement_image_host[i] = make_float2(
          kDcInputScale * static_cast<float>((i * 37 + 5) % 277) / 277.0f,
          kDcInputScale * static_cast<float>((i * 41 + 7) % 281) / 281.0f);
#endif
    }
    for (size_t i = 0; i < sample_elements; ++i) {
      adjoint_samples_host[i] =
          make_float2(
              static_cast<float>((i * 29 + 3 + seed * 107u) % 269) / 269.0f,
              static_cast<float>((i * 31 + 4 + seed * 109u) % 271) / 271.0f);
    }
#if JKF_DATA_CONSISTENCY_MODE != 0
    for (uint32_t row = 0; row < forward.h.output_size; ++row) {
      density_host[row] =
          0.75f + static_cast<float>((row * 43 + 11) % 29) * (1.0f / 64.0f);
    }
#endif
  }
  for (int y = 0; y < kImage; ++y) {
    for (int x = 0; x < kImage; ++x) {
      if (!runtime_input) {
        scaling_host[y * kImage + x] =
            1.0f + static_cast<float>((x + 3 * y + seed) & 7) * 0.015625f;
      }
#if JKF_FP16X2
      forward_scaling_host[y * kImage + x] = scaling_host[y * kImage + x];
#else
      forward_scaling_host[y * kImage + x] =
          scaling_host[y * kImage + x] / static_cast<float>(kGrid);
#endif
      inverse_scaling_host[y * kImage + x] =
          scaling_host[y * kImage + x] * static_cast<float>(kGrid);
      const int x_block = x >> 2;
      const int x_local = x & 3;
      scaling_tiled_host[(x_block * kImage + y) * 4 + x_local] =
          inverse_scaling_host[y * kImage + x] /
          static_cast<float>(kGrid * kGrid);
    }
  }

  std::vector<void*> churn_buffers;
  for (int index = 0; index < pointer_churn; ++index) {
    void* allocation = nullptr;
    const size_t bytes = (1u << 20) +
                         ((seed * 4099u + static_cast<uint32_t>(index) * 65537u) &
                          ((1u << 19) - 1));
    CUDA_CHECK(cudaMalloc(&allocation, bytes));
    CUDA_CHECK(cudaMemset(allocation, 0x5a, bytes));
    churn_buffers.push_back(allocation);
  }

  IntegratedBuffers buffers;
#if JKF_SENSE_FORWARD_MODE != 0
  CUDA_CHECK(cudaMalloc(&buffers.forward_object,
                        kImage * kImage * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&buffers.sensitivity_maps,
                        image_elements * sizeof(float2)));
#endif
  CUDA_CHECK(cudaMalloc(&buffers.forward_image,
                        image_elements * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&buffers.forward_samples,
                        sample_elements * sizeof(float2)));
#if JKF_FUSED_DIRECT_COMPLEX_OUTPUT
  CUDA_CHECK(cudaMalloc(&buffers.forward_reference,
                        sample_elements * sizeof(float2)));
#endif
  CUDA_CHECK(cudaMalloc(&buffers.adjoint_samples,
                        sample_elements * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&buffers.adjoint_image,
                        image_elements * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&buffers.grid_input, grid_elements * sizeof(float2)));
#if JKF_CUSTOM_FORWARD_F30
  CUDA_CHECK(cudaMemset(buffers.grid_input, 0,
                        grid_elements * sizeof(float2)));
#endif
  CUDA_CHECK(cudaMalloc(&buffers.grid_output, grid_elements * sizeof(float2)));
#if JKF_CUSTOM_INVERSE_F31
  CUDA_CHECK(cudaMalloc(&buffers.inverse_reference,
                        image_elements * sizeof(float2)));
#endif
#if JKF_CUSTOM_FORWARD_F30
  CUDA_CHECK(cudaMalloc(&buffers.forward_dense_reference,
                        static_cast<size_t>(forward.h.n_cols) * kN *
                            sizeof(half)));
#endif
#if JKF_FP16X2 && !JKF_FP16X2_DISABLE_DEBUG_ENDPOINT
  CUDA_CHECK(cudaMalloc(&buffers.forward_fp32_endpoint,
                        grid_elements * sizeof(float2)));
#endif
#if JKF_FUSED_DIRECT_COMPLEX_OUTPUT
  CUDA_CHECK(cudaMalloc(&buffers.adjoint_grid_input,
                        grid_elements * sizeof(float2)));
  CUDA_CHECK(cudaMemset(buffers.adjoint_grid_input, 0,
                        grid_elements * sizeof(float2)));
#endif
  CUDA_CHECK(cudaMalloc(&buffers.forward_scaling,
                        forward_scaling_host.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.inverse_scaling,
                        inverse_scaling_host.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.scaling_tiled,
                        scaling_tiled_host.size() * sizeof(float)));
#if JKF_DATA_CONSISTENCY_MODE != 0
  CUDA_CHECK(cudaMalloc(&buffers.density,
                        density_host.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.dc_dense_reference,
                        static_cast<size_t>(adjoint.h.n_cols) * kN *
                            sizeof(half)));
  CUDA_CHECK(cudaMalloc(&buffers.dc_image_reference,
                        image_elements * sizeof(float2)));
#endif
  CUDA_CHECK(cudaMemcpy(buffers.forward_image, forward_image_host.data(),
                        image_elements * sizeof(float2), cudaMemcpyHostToDevice));
#if JKF_SENSE_FORWARD_MODE != 0
  CUDA_CHECK(cudaMemcpy(buffers.forward_object, forward_object_host.data(),
                        kImage * kImage * sizeof(float2),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.sensitivity_maps,
                        sensitivity_maps_host.data(),
                        image_elements * sizeof(float2),
                        cudaMemcpyHostToDevice));
#endif
  CUDA_CHECK(cudaMemcpy(buffers.adjoint_samples, adjoint_samples_host.data(),
                        sample_elements * sizeof(float2), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.forward_scaling, forward_scaling_host.data(),
                        forward_scaling_host.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.inverse_scaling, inverse_scaling_host.data(),
                        inverse_scaling_host.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.scaling_tiled, scaling_tiled_host.data(),
                        scaling_tiled_host.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
#if JKF_DATA_CONSISTENCY_MODE != 0
  CUDA_CHECK(cudaMemcpy(buffers.density, density_host.data(),
                        density_host.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
#endif

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
#if JKF_CUSTOM_FORWARD_F30 || JKF_CUSTOM_INVERSE_F31
  CUDA_CHECK(jkf_initialize_custom_fft_twiddles());
#endif
  cufftHandle plan = 0;
#if !JKF_DISABLE_CUFFT_RUNTIME
  int dimensions[2] = {kGrid, kGrid};
  CUFFT_CHECK(cufftPlanMany(&plan, 2, dimensions, nullptr, 1, kGrid * kGrid,
                            nullptr, 1, kGrid * kGrid, CUFFT_C2C, kCoils));
  CUFFT_CHECK(cufftSetStream(plan, stream));
#endif

#if JKF_FP16X2 && !JKF_FP16X2_DISABLE_DEBUG_ENDPOINT
#if JKF_SENSE_FORWARD_MODE != 0
#error "FP16x2 debug endpoint currently admits coil-image input only"
#endif
  jkf_launch_custom_fft_f30_forward_split2_debug(
      buffers.forward_image, buffers.forward_scaling, buffers.grid_input,
      forward.d.dense, forward.dense_residual,
      buffers.forward_fp32_endpoint, stream);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif

#if JKF_DATA_CONSISTENCY_MODE != 0
  if (!runtime_input) {
    CUDA_CHECK(cudaMemcpyAsync(buffers.forward_image,
                               measurement_image_host.data(),
                               image_elements * sizeof(float2),
                               cudaMemcpyHostToDevice, stream));
    prepare_forward_dense(forward, buffers, plan, stream);
    launch_fused_interpolation(
        forward, stream, reinterpret_cast<float*>(buffers.adjoint_samples));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers.forward_image, forward_image_host.data(),
                               image_elements * sizeof(float2),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
#endif

  ErrorStats forward_error;
  ErrorStats adjoint_error;
  ErrorStats inverse_error;
  ErrorStats forward_fft_error;
  ErrorStats sense_forward_error;
  size_t panel_bit_mismatches = 0;
  size_t final_bit_mismatches = 0;
  size_t sense_panel_bit_mismatches = 0;
#if JKF_SENSE_FORWARD_MODE == 2
  constexpr int kSenseThreads = 256;
  integrated_sense_expand
      <<<(image_elements + kSenseThreads - 1) / kSenseThreads, kSenseThreads,
         0, stream>>>(buffers.forward_object, buffers.sensitivity_maps,
                     buffers.forward_image);
  jkf_launch_custom_fft_f30_forward(
      buffers.forward_image, buffers.forward_scaling, buffers.grid_input,
      buffers.forward_dense_reference, stream);
  prepare_forward_dense(forward, buffers, plan, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  sense_forward_error = compare_half_device(
      forward.d.dense, buffers.forward_dense_reference,
      static_cast<size_t>(forward.h.n_cols) * kN,
      &sense_panel_bit_mismatches);
#endif
#if JKF_DATA_CONSISTENCY_MODE != 0
  prepare_forward_dense(forward, buffers, plan, stream);
  launch_data_consistency_materialized(forward, buffers, adjoint.d.dense,
                                       stream);
  CUDA_CHECK(cudaMemcpyAsync(
      buffers.dc_dense_reference, adjoint.d.dense,
      static_cast<size_t>(adjoint.h.n_cols) * kN * sizeof(half),
      cudaMemcpyDeviceToDevice, stream));
  launch_data_consistency_adjoint(adjoint, buffers, plan, stream);
  CUDA_CHECK(cudaMemcpyAsync(buffers.dc_image_reference,
                             buffers.adjoint_image,
                             image_elements * sizeof(float2),
                             cudaMemcpyDeviceToDevice, stream));

  prepare_forward_dense(forward, buffers, plan, stream);
  launch_data_consistency_fused(forward, buffers, adjoint.d.dense, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  forward_error = compare_half_device(
      adjoint.d.dense, buffers.dc_dense_reference,
      static_cast<size_t>(adjoint.h.n_cols) * kN, &panel_bit_mismatches);
  launch_data_consistency_adjoint(adjoint, buffers, plan, stream);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  adjoint_error = compare_device(
      reinterpret_cast<const float*>(buffers.adjoint_image),
      reinterpret_cast<const float*>(buffers.dc_image_reference),
      image_elements * 2);
  final_bit_mismatches = count_device_bit_mismatches(
      buffers.adjoint_image, buffers.dc_image_reference, image_elements * 2);
#else
#if JKF_FUSED_DIRECT_COMPLEX_OUTPUT
  constexpr int kThreads = 256;
  prepare_forward_dense(forward, buffers, plan, stream);
#if JKF_CUSTOM_FORWARD_F30 && !JKF_DISABLE_CUFFT_RUNTIME
  const uint32_t forward_grid_elements = kCoils * kGrid * kGrid;
  integrated_pad_scale
      <<<(forward_grid_elements + kThreads - 1) / kThreads, kThreads, 0,
         stream>>>(buffers.forward_image, buffers.forward_scaling,
                   buffers.grid_input);
  CUFFT_CHECK(cufftExecC2C(
      plan, reinterpret_cast<cufftComplex*>(buffers.grid_input),
      reinterpret_cast<cufftComplex*>(buffers.grid_output), CUFFT_FORWARD));
  const uint32_t forward_dense_elements = forward.h.n_cols * kCoils;
  integrated_complex_to_half_rows
      <<<(forward_dense_elements + kThreads - 1) / kThreads, kThreads, 0,
         stream>>>(buffers.grid_output, forward.h.n_cols,
                   buffers.forward_dense_reference);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  size_t forward_fft_bit_mismatches = 0;
  forward_fft_error = compare_half_device(
      forward.d.dense, buffers.forward_dense_reference,
      static_cast<size_t>(forward.h.n_cols) * kN,
      &forward_fft_bit_mismatches);
#endif
  launch_fused_interpolation(
      forward, stream, reinterpret_cast<float*>(buffers.forward_samples));
  launch_unfused_interpolation(forward, stream);
  const uint32_t forward_elements = forward.h.output_size * kCoils;
  integrated_float_rows_to_complex
      <<<(forward_elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          forward.d.output, forward.h.output_size, buffers.forward_reference);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  forward_error = compare_device(
      reinterpret_cast<const float*>(buffers.forward_samples),
      reinterpret_cast<const float*>(buffers.forward_reference),
      sample_elements * 2);

  prepare_adjoint_dense(adjoint, buffers, stream);
  launch_fused_interpolation(
      adjoint, stream, reinterpret_cast<float*>(buffers.adjoint_grid_input),
      true);
  launch_unfused_interpolation(adjoint, stream, true);
  const uint32_t adjoint_elements = adjoint.h.output_size * kCoils;
  integrated_float_rows_to_complex
      <<<(adjoint_elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          adjoint.d.output, adjoint.h.output_size, buffers.grid_output);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  adjoint_error = compare_device(
      reinterpret_cast<const float*>(buffers.adjoint_grid_input),
      reinterpret_cast<const float*>(buffers.grid_output), grid_elements * 2);
#if JKF_CUSTOM_INVERSE_F31 && !JKF_DISABLE_CUFFT_RUNTIME
  CUFFT_CHECK(cufftExecC2C(
      plan, reinterpret_cast<cufftComplex*>(buffers.adjoint_grid_input),
      reinterpret_cast<cufftComplex*>(buffers.grid_output), CUFFT_INVERSE));
  integrated_crop_scale<<<(image_elements + kThreads - 1) / kThreads, kThreads,
                          0, stream>>>(buffers.grid_output, buffers.inverse_scaling,
                                      buffers.inverse_reference);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  jkf_launch_custom_ifft_f31_adjoint(
      buffers.adjoint_grid_input, buffers.grid_output, buffers.inverse_scaling,
      buffers.scaling_tiled, buffers.adjoint_image, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  inverse_error = compare_device(
      reinterpret_cast<const float*>(buffers.adjoint_image),
      reinterpret_cast<const float*>(buffers.inverse_reference),
      image_elements * 2);
#endif
#else
  prepare_forward_dense(forward, buffers, plan, stream);
  launch_fused_interpolation(forward, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const size_t forward_panel_elements =
      static_cast<size_t>(forward.h.output_size) * kN;
  std::vector<float> forward_fused(forward_panel_elements);
  CUDA_CHECK(cudaMemcpy(forward_fused.data(), forward.d.output,
                        forward_panel_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  launch_unfused_interpolation(forward, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float* forward_fused_device = nullptr;
  CUDA_CHECK(cudaMalloc(&forward_fused_device,
                        forward_panel_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(forward_fused_device, forward_fused.data(),
                        forward_panel_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  forward_error = compare_device(
      forward_fused_device, forward.d.output, forward_panel_elements);
  cudaFree(forward_fused_device);

  prepare_adjoint_dense(adjoint, buffers, stream);
  launch_fused_interpolation(adjoint, stream, nullptr, true);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const size_t adjoint_panel_elements =
      static_cast<size_t>(adjoint.h.output_size) * kN;
  std::vector<float> adjoint_fused(adjoint_panel_elements);
  CUDA_CHECK(cudaMemcpy(adjoint_fused.data(), adjoint.d.output,
                        adjoint_panel_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  launch_unfused_interpolation(adjoint, stream, true);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float* adjoint_fused_device = nullptr;
  CUDA_CHECK(cudaMalloc(&adjoint_fused_device,
                        adjoint_panel_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(adjoint_fused_device, adjoint_fused.data(),
                        adjoint_panel_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  adjoint_error = compare_device(
      adjoint_fused_device, adjoint.d.output, adjoint_panel_elements);
  cudaFree(adjoint_fused_device);
#endif
#endif

#if JKF_DATA_CONSISTENCY_MODE != 0
  launch_data_consistency_once(forward, adjoint, buffers, plan, stream);
#else
  launch_integrated_once(forward, adjoint, buffers, plan, stream);
#endif
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
#if JKF_DATA_CONSISTENCY_MODE != 0
  const size_t final_nonfinite =
      forward_error.nonfinite +
      count_nonfinite_complex(buffers.adjoint_image, image_elements);
  const bool correct = forward_error.nonfinite == 0 &&
                       adjoint_error.nonfinite == 0 && final_nonfinite == 0 &&
                       panel_bit_mismatches == 0 && final_bit_mismatches == 0
#if JKF_SENSE_FORWARD_MODE == 2
                       && sense_forward_error.nonfinite == 0 &&
                       sense_forward_error.rel_l2 <= 1e-7 &&
                       sense_panel_bit_mismatches == 0
#endif
      ;
#else
  const size_t final_nonfinite =
      count_nonfinite_complex(buffers.forward_samples, sample_elements) +
      count_nonfinite_complex(buffers.adjoint_image, image_elements);
  const bool correct = forward_error.nonfinite == 0 &&
                       adjoint_error.nonfinite == 0 && final_nonfinite == 0 &&
                       forward_error.rel_l2 <= 2e-6 &&
                       adjoint_error.rel_l2 <= 2e-6
#if JKF_CUSTOM_FORWARD_F30 && !JKF_DISABLE_CUFFT_RUNTIME
                       && forward_fft_error.nonfinite == 0 &&
                       forward_fft_error.rel_l2 <= 5e-4 &&
                       forward_fft_error.max_scaled_abs <= 4e-3f
#endif
#if JKF_CUSTOM_INVERSE_F31 && !JKF_DISABLE_CUFFT_RUNTIME
                       && inverse_error.nonfinite == 0 &&
                       inverse_error.rel_l2 <= 2e-5
#endif
      ;
#endif
  if (!correct) {
    std::printf(
        "{\"variant\":\"%s\",\"correct\":false,\"forward_rel_l2\":%.9g,"
        "\"adjoint_rel_l2\":%.9g,\"forward_fft_rel_l2\":%.9g,"
        "\"forward_fft_max_abs\":%.9g,\"forward_fft_max_ulp\":%u,"
        "\"forward_fft_max_scaled_abs\":%.9g,"
        "\"forward_fft_nonfinite\":%zu,"
        "\"inverse_rel_l2\":%.9g,\"inverse_max_abs\":%.9g,"
        "\"inverse_nonfinite\":%zu,"
        "\"panel_bit_mismatches\":%zu,"
        "\"final_bit_mismatches\":%zu,"
        "\"sense_forward_rel_l2\":%.9g,"
        "\"sense_panel_bit_mismatches\":%zu,\"nonfinite\":%zu}\n",
        integrated_variant_name(), forward_error.rel_l2,
        adjoint_error.rel_l2, forward_fft_error.rel_l2,
        forward_fft_error.max_abs, forward_fft_error.max_ulp,
        forward_fft_error.max_scaled_abs, forward_fft_error.nonfinite,
        inverse_error.rel_l2, inverse_error.max_abs, inverse_error.nonfinite,
        panel_bit_mismatches, final_bit_mismatches,
        sense_forward_error.rel_l2, sense_panel_bit_mismatches,
        final_nonfinite);
    return 3;
  }

  const auto launch_timed_once = [&]() {
#if JKF_DATA_CONSISTENCY_MODE != 0
    launch_data_consistency_once(forward, adjoint, buffers, plan, stream);
#else
    launch_integrated_once(forward, adjoint, buffers, plan, stream);
#endif
  };

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  if (graph_mode) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_timed_once();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  }

  const auto launch_benchmark_once = [&]() {
    if (graph_mode) {
      CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    } else {
      launch_timed_once();
    }
  };

  for (int i = 0; i < warmup; ++i) {
    launch_benchmark_once();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
#if JKF_SYSTEM_PROFILE
  CUDA_CHECK(cudaProfilerStart());
#endif
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) {
    launch_benchmark_once();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
#if JKF_SYSTEM_PROFILE
  CUDA_CHECK(cudaProfilerStop());
#endif
  for (int i = 0; i < stress; ++i) {
    launch_benchmark_once();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
#if JKF_DATA_CONSISTENCY_MODE == 0
  if (dump_output) {
    write_device_array(dump_output_directory + "/forward_samples.c64.bin",
                       buffers.forward_samples, sample_elements);
    write_device_array(dump_output_directory + "/adjoint_image.c64.bin",
                       buffers.adjoint_image, image_elements);
    write_device_array(dump_output_directory + "/forward_dense.f16.bin",
                       forward.d.dense,
                       static_cast<size_t>(forward.h.n_cols) * kN);
    write_device_array(dump_output_directory + "/adjoint_dense.f16.bin",
                       adjoint.d.dense,
                       static_cast<size_t>(adjoint.h.n_cols) * kN);
#if JKF_FP16X2 && !JKF_FP16X2_DISABLE_DEBUG_ENDPOINT
    write_device_array(
        dump_output_directory + "/forward_dense_residual.f16.bin",
        forward.dense_residual, static_cast<size_t>(forward.h.n_cols) * kN);
    write_device_array(
        dump_output_directory + "/adjoint_dense_residual.f16.bin",
        adjoint.dense_residual, static_cast<size_t>(adjoint.h.n_cols) * kN);
    write_device_array(
        dump_output_directory + "/forward_fp32_endpoint.c64.bin",
        buffers.forward_fp32_endpoint, grid_elements);
#endif
    write_device_array(dump_output_directory + "/adjoint_grid.c64.bin",
                       buffers.adjoint_grid_input, grid_elements);
  }
#else
  if (dump_output) {
    write_device_array(dump_output_directory + "/adjoint_image.c64.bin",
                       buffers.adjoint_image, image_elements);
  }
#endif
  const double latency_ms = static_cast<double>(elapsed_ms) / iterations;
  std::printf(
      "{\"variant\":\"%s\",\"correct\":true,\"us\":%.9g,"
      "\"forward_rel_l2\":%.9g,\"adjoint_rel_l2\":%.9g,"
      "\"forward_fft_rel_l2\":%.9g,"
      "\"forward_fft_max_ulp\":%u,"
      "\"forward_fft_max_scaled_abs\":%.9g,"
      "\"inverse_rel_l2\":%.9g,"
      "\"forward_max_abs\":%.9g,\"adjoint_max_abs\":%.9g,"
      "\"panel_bit_mismatches\":%zu,\"final_bit_mismatches\":%zu,"
      "\"sense_forward_rel_l2\":%.9g,"
      "\"sense_panel_bit_mismatches\":%zu,"
      "\"nonfinite\":0,\"warmup\":%d,\"iters\":%d,\"stress\":%d,"
      "\"graph\":%s,\"seed\":%u,\"pointer_churn\":%d,"
      "\"runtime_input\":%s,\"dump_output\":%s}\n",
      integrated_variant_name(), latency_ms * 1000.0, forward_error.rel_l2,
      adjoint_error.rel_l2, forward_fft_error.rel_l2,
      forward_fft_error.max_ulp, forward_fft_error.max_scaled_abs,
      inverse_error.rel_l2, forward_error.max_abs, adjoint_error.max_abs,
      panel_bit_mismatches, final_bit_mismatches,
      sense_forward_error.rel_l2, sense_panel_bit_mismatches,
      warmup, iterations, stress,
      graph_mode ? "true" : "false", seed, pointer_churn,
      runtime_input ? "true" : "false", dump_output ? "true" : "false");
  std::printf(
      "ALPHA_OPS_RESULT {\"status\":\"pass\","
      "\"correctness_status\":\"pass\",\"variant\":\"%s\","
      "\"alpha_latency_ms\":%.12g,\"forward_rel_l2\":%.9g,"
      "\"adjoint_rel_l2\":%.9g,\"forward_fft_rel_l2\":%.9g,"
      "\"forward_fft_max_ulp\":%u,"
      "\"forward_fft_max_scaled_abs\":%.9g,"
      "\"inverse_rel_l2\":%.9g,"
      "\"panel_bit_mismatches\":%zu,"
      "\"final_bit_mismatches\":%zu,"
      "\"sense_forward_rel_l2\":%.9g,"
      "\"sense_panel_bit_mismatches\":%zu,"
      "\"graph\":%s,\"seed\":%u,"
      "\"pointer_churn\":%d,\"runtime_input\":%s,"
      "\"dump_output\":%s}\n",
      integrated_variant_name(), latency_ms, forward_error.rel_l2,
      adjoint_error.rel_l2, forward_fft_error.rel_l2,
      forward_fft_error.max_ulp, forward_fft_error.max_scaled_abs,
      inverse_error.rel_l2, panel_bit_mismatches, final_bit_mismatches,
      sense_forward_error.rel_l2, sense_panel_bit_mismatches,
      graph_mode ? "true" : "false", seed, pointer_churn,
      runtime_input ? "true" : "false", dump_output ? "true" : "false");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (graph_exec) cudaGraphExecDestroy(graph_exec);
  if (graph) cudaGraphDestroy(graph);
#if !JKF_DISABLE_CUFFT_RUNTIME
  cufftDestroy(plan);
#endif
  cudaStreamDestroy(stream);
#if JKF_SENSE_FORWARD_MODE != 0
  cudaFree(buffers.forward_object);
  cudaFree(buffers.sensitivity_maps);
#endif
  cudaFree(buffers.forward_image);
  cudaFree(buffers.forward_samples);
#if JKF_FUSED_DIRECT_COMPLEX_OUTPUT
  cudaFree(buffers.forward_reference);
#endif
  cudaFree(buffers.adjoint_samples);
  cudaFree(buffers.adjoint_image);
  cudaFree(buffers.grid_input);
  cudaFree(buffers.grid_output);
#if JKF_CUSTOM_INVERSE_F31 && !JKF_DISABLE_CUFFT_RUNTIME
  cudaFree(buffers.inverse_reference);
#endif
#if JKF_CUSTOM_FORWARD_F30
  cudaFree(buffers.forward_dense_reference);
#endif
#if JKF_FP16X2 && !JKF_FP16X2_DISABLE_DEBUG_ENDPOINT
  cudaFree(buffers.forward_fp32_endpoint);
#endif
#if JKF_FUSED_DIRECT_COMPLEX_OUTPUT
  cudaFree(buffers.adjoint_grid_input);
#endif
  cudaFree(buffers.forward_scaling);
  cudaFree(buffers.inverse_scaling);
  cudaFree(buffers.scaling_tiled);
#if JKF_DATA_CONSISTENCY_MODE != 0
  cudaFree(buffers.density);
  cudaFree(buffers.dc_dense_reference);
  cudaFree(buffers.dc_image_reference);
#endif
  free_direction(forward);
  free_direction(adjoint);
  for (void* allocation : churn_buffers) cudaFree(allocation);
  return 0;
}
