#define JKF_VARIANT 5
#define JKF_CUSTOM_FORWARD_F30 1
#define JKF_CUSTOM_INVERSE_F31 1
#define JKF_CUSTOM_INVERSE_REGULAR_X 1
#define JKF_CUSTOM_INVERSE_PRENORMALIZED 1
#define JKF_CUSTOM_INVERSE_SAFE_CROP 1
#define JKF_CUSTOM_INVERSE_REGULAR_Y_CROP 1
#define JKF_DISABLE_CUFFT_RUNTIME 1
#define JKF_FP16X2 1
#define JKF_FP16X2_DISABLE_DEBUG_ENDPOINT 1
#define JKF_SENSE_FORWARD_MODE 2
#define main jkf_integrated_unused_main
#include "nufft_integrated_bench.cu"
#undef main

#include <array>

#ifndef JKF_CG_TELEMETRY
#define JKF_CG_TELEMETRY 0
#endif

namespace {

constexpr int kCgThreads = 256;
constexpr int kCgBlocks = (kImage * kImage + kCgThreads - 1) / kCgThreads;

__global__ void sense_combine_scaled(
    const float2* __restrict__ coil_images,
    const float2* __restrict__ sensitivity_maps,
    const float2* __restrict__ regularizer_input,
    float inverse_operator_scale_squared,
    float scaled_lambda,
    float2* __restrict__ output) {
  const uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
  if (pixel >= static_cast<uint32_t>(kImage * kImage)) return;
  float real_sum = 0.0f;
  float imag_sum = 0.0f;
#pragma unroll
  for (int coil = 0; coil < kCoils; ++coil) {
    const size_t index = static_cast<size_t>(coil) * kImage * kImage + pixel;
    const float2 value = coil_images[index];
    const float2 sensitivity = sensitivity_maps[index];
    real_sum += sensitivity.x * value.x + sensitivity.y * value.y;
    imag_sum += sensitivity.x * value.y - sensitivity.y * value.x;
  }
  real_sum *= inverse_operator_scale_squared;
  imag_sum *= inverse_operator_scale_squared;
  if (regularizer_input != nullptr) {
    const float2 prior = regularizer_input[pixel];
    real_sum += scaled_lambda * prior.x;
    imag_sum += scaled_lambda * prior.y;
  }
  output[pixel] = make_float2(real_sum, imag_sum);
}

__global__ void cg_initialize(const float2* __restrict__ b,
                              float2* __restrict__ x,
                              float2* __restrict__ r,
                              float2* __restrict__ p) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= static_cast<uint32_t>(kImage * kImage)) return;
  x[index] = make_float2(0.0f, 0.0f);
  r[index] = b[index];
  p[index] = b[index];
}

__global__ void dot_partial(const float2* __restrict__ left,
                            const float2* __restrict__ right,
                            float* __restrict__ partial) {
  __shared__ float shared[kCgThreads];
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  float value = 0.0f;
  if (index < static_cast<uint32_t>(kImage * kImage)) {
    const float2 a = left[index];
    const float2 b = right[index];
    value = a.x * b.x + a.y * b.y;
  }
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = kCgThreads / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) shared[threadIdx.x] += shared[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = shared[0];
}

__global__ void dot_finalize(const float* __restrict__ partial,
                             float* __restrict__ result) {
  __shared__ float shared[kCgThreads];
  const int lane = threadIdx.x;
  shared[lane] = lane < kCgBlocks ? partial[lane] : 0.0f;
  __syncthreads();
  for (int offset = kCgThreads / 2; offset > 0; offset >>= 1) {
    if (lane < offset) shared[lane] += shared[lane + offset];
    __syncthreads();
  }
  if (lane == 0) result[0] = shared[0];
}

void launch_dot(const float2* left, const float2* right, float* partial,
                float* result, cudaStream_t stream) {
  dot_partial<<<kCgBlocks, kCgThreads, 0, stream>>>(left, right, partial);
  dot_finalize<<<1, kCgThreads, 0, stream>>>(partial, result);
}

__global__ void compute_alpha(const float* __restrict__ rho,
                              const float* __restrict__ denominator,
                              float* __restrict__ alpha,
                              int* __restrict__ positive_curvature
#if JKF_CG_TELEMETRY
                              , float* __restrict__ denominator_history,
                              float* __restrict__ alpha_history,
                              int iteration
#endif
                              ) {
  if (threadIdx.x != 0) return;
  const float value = denominator[0];
#if JKF_CG_TELEMETRY
  denominator_history[iteration] = value;
#endif
  if (!isfinite(value) || value <= 0.0f) {
    positive_curvature[0] = 0;
    alpha[0] = 0.0f;
  } else {
    alpha[0] = rho[0] / value;
  }
#if JKF_CG_TELEMETRY
  alpha_history[iteration] = alpha[0];
#endif
}

__global__ void update_x_r(float2* __restrict__ x,
                           float2* __restrict__ r,
                           const float2* __restrict__ p,
                           const float2* __restrict__ ap,
                           const float* __restrict__ alpha) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= static_cast<uint32_t>(kImage * kImage)) return;
  const float step = alpha[0];
  float2 x_value = x[index];
  float2 r_value = r[index];
  const float2 p_value = p[index];
  const float2 ap_value = ap[index];
  x_value.x += step * p_value.x;
  x_value.y += step * p_value.y;
  r_value.x -= step * ap_value.x;
  r_value.y -= step * ap_value.y;
  x[index] = x_value;
  r[index] = r_value;
}

__global__ void compute_beta(float* __restrict__ rho,
                             const float* __restrict__ rho_new,
                             float* __restrict__ beta,
                             float* __restrict__ residual_history,
                             int iteration) {
  if (threadIdx.x != 0) return;
  beta[0] = rho_new[0] / fmaxf(rho[0], 1e-30f);
  rho[0] = rho_new[0];
  residual_history[iteration + 1] = sqrtf(fmaxf(rho_new[0], 0.0f));
}

__global__ void update_p(float2* __restrict__ p,
                         const float2* __restrict__ r,
                         const float* __restrict__ beta) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= static_cast<uint32_t>(kImage * kImage)) return;
  const float scale = beta[0];
  const float2 old_p = p[index];
  const float2 r_value = r[index];
  p[index] = make_float2(r_value.x + scale * old_p.x,
                         r_value.y + scale * old_p.y);
}

__global__ void initialize_scalar_state(float* rho, float* residual_history,
                                        int* positive_curvature) {
  if (threadIdx.x != 0) return;
  residual_history[0] = sqrtf(fmaxf(rho[0], 0.0f));
  positive_curvature[0] = 1;
}

struct CgBuffers {
  float2* measurement = nullptr;
  float2* b = nullptr;
  float2* x = nullptr;
  float2* r = nullptr;
  float2* p = nullptr;
  float2* ap = nullptr;
  float* partial = nullptr;
  float* rho = nullptr;
  float* rho_new = nullptr;
  float* denominator = nullptr;
  float* alpha = nullptr;
  float* beta = nullptr;
  float* residual_history = nullptr;
  int* positive_curvature = nullptr;
#if JKF_CG_TELEMETRY
  float* denominator_history = nullptr;
  float* alpha_history = nullptr;
#endif
};

void launch_adjoint_image(PackedDirection& adjoint, IntegratedBuffers& buffers,
                          const float2* samples, float2* output,
                          const float2* regularizer_input,
                          float inverse_operator_scale_squared,
                          float scaled_lambda, cudaStream_t stream) {
  buffers.adjoint_samples = const_cast<float2*>(samples);
  prepare_adjoint_dense(adjoint, buffers, stream);
  launch_fused_interpolation(
      adjoint, stream, reinterpret_cast<float*>(buffers.adjoint_grid_input),
      true);
  launch_fused_interpolation_residual(
      adjoint, stream, reinterpret_cast<float*>(buffers.adjoint_grid_input),
      true);
  jkf_launch_custom_ifft_f31_adjoint(
      buffers.adjoint_grid_input, buffers.grid_output, buffers.inverse_scaling,
      buffers.scaling_tiled, buffers.adjoint_image, stream);
  sense_combine_scaled<<<kCgBlocks, kCgThreads, 0, stream>>>(
      buffers.adjoint_image, buffers.sensitivity_maps, regularizer_input,
      inverse_operator_scale_squared, scaled_lambda, output);
}

void launch_normal(PackedDirection& forward, PackedDirection& adjoint,
                   IntegratedBuffers& buffers, const float2* input,
                   float2* output, float inverse_operator_scale_squared,
                   float scaled_lambda, cudaStream_t stream) {
  buffers.forward_object = const_cast<float2*>(input);
  prepare_forward_dense(forward, buffers, 0, stream);
  constexpr float kForwardOutputScale = 1.0f / static_cast<float>(kGrid);
  launch_fused_interpolation(
      forward, stream, reinterpret_cast<float*>(buffers.forward_samples), false,
      kForwardOutputScale);
  launch_fused_interpolation_residual(
      forward, stream, reinterpret_cast<float*>(buffers.forward_samples), false,
      kForwardOutputScale);
  launch_adjoint_image(adjoint, buffers, buffers.forward_samples, output, input,
                       inverse_operator_scale_squared, scaled_lambda, stream);
}

void launch_cg(PackedDirection& forward, PackedDirection& adjoint,
               IntegratedBuffers& buffers, CgBuffers& cg,
               float inverse_operator_scale_squared, float scaled_lambda,
               int cg_iterations, cudaStream_t stream) {
  launch_adjoint_image(adjoint, buffers, cg.measurement, cg.b, nullptr,
                       inverse_operator_scale_squared, 0.0f, stream);
  cg_initialize<<<kCgBlocks, kCgThreads, 0, stream>>>(cg.b, cg.x, cg.r, cg.p);
  launch_dot(cg.r, cg.r, cg.partial, cg.rho, stream);
  initialize_scalar_state<<<1, 1, 0, stream>>>(
      cg.rho, cg.residual_history, cg.positive_curvature);
  for (int iteration = 0; iteration < cg_iterations; ++iteration) {
    launch_normal(forward, adjoint, buffers, cg.p, cg.ap,
                  inverse_operator_scale_squared, scaled_lambda, stream);
    launch_dot(cg.p, cg.ap, cg.partial, cg.denominator, stream);
    compute_alpha<<<1, 1, 0, stream>>>(
        cg.rho, cg.denominator, cg.alpha, cg.positive_curvature
#if JKF_CG_TELEMETRY
        , cg.denominator_history, cg.alpha_history, iteration
#endif
        );
    update_x_r<<<kCgBlocks, kCgThreads, 0, stream>>>(
        cg.x, cg.r, cg.p, cg.ap, cg.alpha);
    launch_dot(cg.r, cg.r, cg.partial, cg.rho_new, stream);
    compute_beta<<<1, 1, 0, stream>>>(
        cg.rho, cg.rho_new, cg.beta, cg.residual_history, iteration);
    update_p<<<kCgBlocks, kCgThreads, 0, stream>>>(cg.p, cg.r, cg.beta);
  }
}

template <typename T>
T* allocate_device(size_t count) {
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
  return pointer;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(
        stderr,
        "usage: %s FWD_REAL FWD_IMAG ADJ_REAL ADJ_IMAG RUNTIME_DIR "
        "OPERATOR_SCALE [warmup=5] [iters=20] [stress=0] [graph=0] "
        "[cg_iterations=10] [dump_dir]\n",
        argv[0]);
    return 2;
  }
  const std::string runtime_directory = argv[5];
  const float operator_scale = std::strtof(argv[6], nullptr);
  const int warmup = argc > 7 ? std::atoi(argv[7]) : 5;
  const int iterations = argc > 8 ? std::atoi(argv[8]) : 20;
  const int stress = argc > 9 ? std::atoi(argv[9]) : 0;
  const bool graph_mode = argc > 10 ? std::atoi(argv[10]) != 0 : false;
  const int cg_iterations = argc > 11 ? std::atoi(argv[11]) : 10;
  const bool dump = argc > 12;
  const std::string dump_directory = dump ? argv[12] : "";
  if (!(operator_scale > 0.0f) || cg_iterations <= 0) return 2;

#if JKF_DENSE_TC_CONTROL
  const char* forward_dense_control = std::getenv("JKF_DENSE_FWD_CONTROL");
  const char* adjoint_dense_control = std::getenv("JKF_DENSE_ADJ_CONTROL");
  if (forward_dense_control == nullptr || adjoint_dense_control == nullptr) {
    std::fprintf(stderr,
                 "dense control requires JKF_DENSE_FWD_CONTROL and "
                 "JKF_DENSE_ADJ_CONTROL\n");
    return 2;
  }
  PackedDirection forward =
      load_direction(argv[1], argv[2], forward_dense_control);
  PackedDirection adjoint =
      load_direction(argv[3], argv[4], adjoint_dense_control);
#else
  PackedDirection forward = load_direction(argv[1], argv[2]);
  PackedDirection adjoint = load_direction(argv[3], argv[4]);
#endif
  if (forward.h.n_cols != static_cast<uint32_t>(kGrid * kGrid) ||
      adjoint.h.output_size != static_cast<uint32_t>(kGrid * kGrid) ||
      forward.h.output_size != adjoint.h.n_cols) {
    std::fprintf(stderr, "unexpected operator shape\n");
    return 2;
  }

  const size_t pixels = static_cast<size_t>(kImage) * kImage;
  const size_t image_elements = static_cast<size_t>(kCoils) * pixels;
  const size_t grid_elements = static_cast<size_t>(kCoils) * kGrid * kGrid;
  const size_t sample_elements =
      static_cast<size_t>(kCoils) * forward.h.output_size;
  auto measurement_host = read_array<float2>(
      runtime_directory + "/measurement.c64.bin", sample_elements);
  for (float2& value : measurement_host) {
    value.x /= static_cast<float>(kGrid);
    value.y /= static_cast<float>(kGrid);
  }
  const auto sensitivity_host = read_array<float2>(
      runtime_directory + "/sensitivity_maps.c64.bin", image_elements);
  const auto scaling_host = read_array<float>(
      runtime_directory + "/scaling.f32.bin", pixels);
  std::vector<float> forward_scaling = scaling_host;
  std::vector<float> inverse_scaling(pixels);
  std::vector<float> scaling_tiled(pixels);
  for (int y = 0; y < kImage; ++y) {
    for (int x = 0; x < kImage; ++x) {
      inverse_scaling[y * kImage + x] =
          scaling_host[y * kImage + x] * static_cast<float>(kGrid);
      const int x_block = x >> 2;
      const int x_local = x & 3;
      scaling_tiled[(x_block * kImage + y) * 4 + x_local] =
          inverse_scaling[y * kImage + x] /
          static_cast<float>(kGrid * kGrid);
    }
  }

  IntegratedBuffers buffers;
  buffers.sensitivity_maps = allocate_device<float2>(image_elements);
  buffers.forward_samples = allocate_device<float2>(sample_elements);
  buffers.adjoint_image = allocate_device<float2>(image_elements);
  buffers.grid_input = allocate_device<float2>(grid_elements);
  buffers.grid_output = allocate_device<float2>(grid_elements);
  buffers.adjoint_grid_input = allocate_device<float2>(grid_elements);
  buffers.forward_scaling = allocate_device<float>(pixels);
  buffers.inverse_scaling = allocate_device<float>(pixels);
  buffers.scaling_tiled = allocate_device<float>(pixels);
  CUDA_CHECK(cudaMemset(buffers.grid_input, 0, grid_elements * sizeof(float2)));
  CUDA_CHECK(cudaMemset(buffers.adjoint_grid_input, 0,
                        grid_elements * sizeof(float2)));
  CUDA_CHECK(cudaMemcpy(buffers.sensitivity_maps, sensitivity_host.data(),
                        image_elements * sizeof(float2), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.forward_scaling, forward_scaling.data(),
                        pixels * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.inverse_scaling, inverse_scaling.data(),
                        pixels * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.scaling_tiled, scaling_tiled.data(),
                        pixels * sizeof(float), cudaMemcpyHostToDevice));

  CgBuffers cg;
  cg.measurement = allocate_device<float2>(sample_elements);
  cg.b = allocate_device<float2>(pixels);
  cg.x = allocate_device<float2>(pixels);
  cg.r = allocate_device<float2>(pixels);
  cg.p = allocate_device<float2>(pixels);
  cg.ap = allocate_device<float2>(pixels);
  cg.partial = allocate_device<float>(kCgBlocks);
  cg.rho = allocate_device<float>(1);
  cg.rho_new = allocate_device<float>(1);
  cg.denominator = allocate_device<float>(1);
  cg.alpha = allocate_device<float>(1);
  cg.beta = allocate_device<float>(1);
  cg.residual_history = allocate_device<float>(cg_iterations + 1);
  cg.positive_curvature = allocate_device<int>(1);
#if JKF_CG_TELEMETRY
  cg.denominator_history = allocate_device<float>(cg_iterations);
  cg.alpha_history = allocate_device<float>(cg_iterations);
#endif
  CUDA_CHECK(cudaMemcpy(cg.measurement, measurement_host.data(),
                        sample_elements * sizeof(float2), cudaMemcpyHostToDevice));

  CUDA_CHECK(jkf_initialize_custom_fft_twiddles());
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  const float inverse_scale_squared = 1.0f / (operator_scale * operator_scale);
  const float normalized_lambda =
      1e-4f / static_cast<float>(kGrid * kGrid);
  const float scaled_lambda = normalized_lambda * inverse_scale_squared;

  launch_cg(forward, adjoint, buffers, cg, inverse_scale_squared, scaled_lambda,
            cg_iterations, stream);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int positive_curvature = 0;
  std::vector<float> residuals(cg_iterations + 1);
  CUDA_CHECK(cudaMemcpy(&positive_curvature, cg.positive_curvature, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(residuals.data(), cg.residual_history,
                        residuals.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const size_t nonfinite = count_nonfinite_complex(cg.x, pixels);
  if (!positive_curvature || nonfinite != 0) {
    std::fprintf(stderr, "CG correctness failure curvature=%d nonfinite=%zu\n",
                 positive_curvature, nonfinite);
    return 3;
  }

  const auto launch_once = [&]() {
    launch_cg(forward, adjoint, buffers, cg, inverse_scale_squared,
              scaled_lambda, cg_iterations, stream);
  };
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  if (graph_mode) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_once();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  }
  const auto launch_benchmark = [&]() {
    if (graph_mode) {
      CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    } else {
      launch_once();
    }
  };
  for (int iteration = 0; iteration < warmup; ++iteration) launch_benchmark();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int iteration = 0; iteration < iterations; ++iteration) launch_benchmark();
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  for (int iteration = 0; iteration < stress; ++iteration) launch_benchmark();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (dump) {
    write_device_array(dump_directory + "/reconstruction.c64.bin", cg.x, pixels);
    write_device_array(dump_directory + "/residuals.f32.bin",
                       cg.residual_history, residuals.size());
#if JKF_CG_TELEMETRY
    write_device_array(dump_directory + "/denominators.f32.bin",
                       cg.denominator_history, cg_iterations);
    write_device_array(dump_directory + "/alphas.f32.bin",
                       cg.alpha_history, cg_iterations);
#endif
  }
  std::printf(
#if JKF_DENSE_TC_CONTROL
      "{\"variant\":\"fp16x2_dense_tc_cg10\",\"correct\":true,\"us\":%.9g,"
#else
      "{\"variant\":\"fp16x2_cg10\",\"correct\":true,\"us\":%.9g,"
#endif
      "\"cg_iterations\":%d,\"operator_scale\":%.9g,"
      "\"residual0\":%.9g,\"residual_final\":%.9g,"
      "\"graph\":%s,\"telemetry\":%s,\"nonfinite\":0}\n",
      static_cast<double>(elapsed_ms) * 1000.0 / iterations, cg_iterations,
      operator_scale, residuals.front(), residuals.back(),
      graph_mode ? "true" : "false",
      JKF_CG_TELEMETRY ? "true" : "false");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (graph_exec) cudaGraphExecDestroy(graph_exec);
  if (graph) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(buffers.sensitivity_maps);
  cudaFree(buffers.forward_samples);
  cudaFree(buffers.adjoint_image);
  cudaFree(buffers.grid_input);
  cudaFree(buffers.grid_output);
  cudaFree(buffers.adjoint_grid_input);
  cudaFree(buffers.forward_scaling);
  cudaFree(buffers.inverse_scaling);
  cudaFree(buffers.scaling_tiled);
  cudaFree(cg.measurement);
  cudaFree(cg.b);
  cudaFree(cg.x);
  cudaFree(cg.r);
  cudaFree(cg.p);
  cudaFree(cg.ap);
  cudaFree(cg.partial);
  cudaFree(cg.rho);
  cudaFree(cg.rho_new);
  cudaFree(cg.denominator);
  cudaFree(cg.alpha);
  cudaFree(cg.beta);
  cudaFree(cg.residual_history);
  cudaFree(cg.positive_curvature);
#if JKF_CG_TELEMETRY
  cudaFree(cg.denominator_history);
  cudaFree(cg.alpha_history);
#endif
  free_direction(forward);
  free_direction(adjoint);
  return 0;
}
