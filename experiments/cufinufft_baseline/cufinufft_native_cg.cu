// Native device-resident cuFINUFFT SENSE-CG baseline.

#include <cuda_runtime.h>
#include <cufinufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kImage = 256;
constexpr int kPixels = kImage * kImage;
constexpr int kSamples = 65536;
constexpr int kCoils = 8;
constexpr int kThreads = 256;
constexpr int kBlocks = (kPixels + kThreads - 1) / kThreads;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kLambda = 1.0e-4f;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    const cudaError_t status_ = (expr);                                          \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status_));                                \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CUFINUFFT_CHECK(expr)                                                   \
  do {                                                                          \
    const int status_ = (expr);                                                  \
    if (status_ != 0) {                                                          \
      std::fprintf(stderr, "cuFINUFFT failure %s:%d: status=%d\n", __FILE__,    \
                   __LINE__, status_);                                           \
      std::exit(4);                                                             \
    }                                                                           \
  } while (0)

template <typename T>
std::vector<T> read_array(const std::string& path, size_t count) {
  std::ifstream stream(path, std::ios::binary);
  std::vector<T> result(count);
  stream.read(reinterpret_cast<char*>(result.data()),
              static_cast<std::streamsize>(count * sizeof(T)));
  if (!stream || stream.gcount() !=
                     static_cast<std::streamsize>(count * sizeof(T))) {
    std::fprintf(stderr, "failed to read %s (%zu elements)\n", path.c_str(),
                 count);
    std::exit(2);
  }
  return result;
}

template <typename T>
T* allocate_device(size_t count) {
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
  return pointer;
}

template <typename T>
void write_device_array(const std::string& path, const T* device, size_t count) {
  std::vector<T> host(count);
  CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(T),
                        cudaMemcpyDeviceToHost));
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(host.data()),
               static_cast<std::streamsize>(count * sizeof(T)));
  if (!stream) {
    std::fprintf(stderr, "failed to write %s\n", path.c_str());
    std::exit(2);
  }
}

__global__ void sense_expand(const float2* __restrict__ image,
                             const float2* __restrict__ sensitivity_maps,
                             float2* __restrict__ coil_modes) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= static_cast<uint32_t>(kCoils * kPixels)) return;
  const uint32_t pixel = index % kPixels;
  const float2 value = image[pixel];
  const float2 sensitivity = sensitivity_maps[index];
  coil_modes[index] = make_float2(
      value.x * sensitivity.x - value.y * sensitivity.y,
      value.x * sensitivity.y + value.y * sensitivity.x);
}

__global__ void sense_combine_scaled(
    const float2* __restrict__ coil_modes,
    const float2* __restrict__ sensitivity_maps,
    const float2* __restrict__ regularizer_input,
    float inverse_operator_scale_squared, float scaled_lambda,
    float2* __restrict__ output) {
  const uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
  if (pixel >= static_cast<uint32_t>(kPixels)) return;
  float real_sum = 0.0f;
  float imag_sum = 0.0f;
#pragma unroll
  for (int coil = 0; coil < kCoils; ++coil) {
    const size_t index = static_cast<size_t>(coil) * kPixels + pixel;
    const float2 value = coil_modes[index];
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
  if (index >= static_cast<uint32_t>(kPixels)) return;
  x[index] = make_float2(0.0f, 0.0f);
  r[index] = b[index];
  p[index] = b[index];
}

__global__ void dot_partial(const float2* __restrict__ left,
                            const float2* __restrict__ right,
                            float* __restrict__ partial) {
  __shared__ float shared[kThreads];
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  float value = 0.0f;
  if (index < static_cast<uint32_t>(kPixels)) {
    const float2 a = left[index];
    const float2 b = right[index];
    value = a.x * b.x + a.y * b.y;
  }
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) shared[threadIdx.x] += shared[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = shared[0];
}

__global__ void dot_finalize(const float* __restrict__ partial,
                             float* __restrict__ result) {
  __shared__ float shared[kThreads];
  const int lane = threadIdx.x;
  shared[lane] = lane < kBlocks ? partial[lane] : 0.0f;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (lane < offset) shared[lane] += shared[lane + offset];
    __syncthreads();
  }
  if (lane == 0) result[0] = shared[0];
}

void launch_dot(const float2* left, const float2* right, float* partial,
                float* result, cudaStream_t stream) {
  dot_partial<<<kBlocks, kThreads, 0, stream>>>(left, right, partial);
  dot_finalize<<<1, kThreads, 0, stream>>>(partial, result);
}

__global__ void compute_alpha(const float* __restrict__ rho,
                              const float* __restrict__ denominator,
                              float* __restrict__ alpha,
                              int* __restrict__ positive_curvature) {
  if (threadIdx.x != 0) return;
  const float value = denominator[0];
  if (!isfinite(value) || value <= 0.0f) {
    positive_curvature[0] = 0;
    alpha[0] = 0.0f;
  } else {
    alpha[0] = rho[0] / value;
  }
}

__global__ void update_x_r(float2* __restrict__ x,
                           float2* __restrict__ r,
                           const float2* __restrict__ p,
                           const float2* __restrict__ ap,
                           const float* __restrict__ alpha) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= static_cast<uint32_t>(kPixels)) return;
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
  if (index >= static_cast<uint32_t>(kPixels)) return;
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

struct OperatorBuffers {
  float2* sensitivity_maps = nullptr;
  float2* coil_modes = nullptr;
  float2* samples = nullptr;
};

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
};

void launch_adjoint(cufinufftf_plan adjoint, OperatorBuffers& buffers,
                    const float2* samples, float2* output,
                    const float2* regularizer_input,
                    float inverse_operator_scale_squared, float scaled_lambda,
                    cudaStream_t stream) {
  CUFINUFFT_CHECK(cufinufftf_execute(
      adjoint, reinterpret_cast<cuFloatComplex*>(const_cast<float2*>(samples)),
      reinterpret_cast<cuFloatComplex*>(buffers.coil_modes)));
  sense_combine_scaled<<<kBlocks, kThreads, 0, stream>>>(
      buffers.coil_modes, buffers.sensitivity_maps, regularizer_input,
      inverse_operator_scale_squared, scaled_lambda, output);
}

void launch_normal(cufinufftf_plan forward, cufinufftf_plan adjoint,
                   OperatorBuffers& buffers, const float2* input,
                   float2* output, float inverse_operator_scale_squared,
                   float scaled_lambda, cudaStream_t stream) {
  sense_expand<<<(kCoils * kPixels + kThreads - 1) / kThreads, kThreads, 0,
                 stream>>>(input, buffers.sensitivity_maps, buffers.coil_modes);
  CUFINUFFT_CHECK(cufinufftf_execute(
      forward, reinterpret_cast<cuFloatComplex*>(buffers.samples),
      reinterpret_cast<cuFloatComplex*>(buffers.coil_modes)));
  launch_adjoint(adjoint, buffers, buffers.samples, output, input,
                 inverse_operator_scale_squared, scaled_lambda, stream);
}

void launch_cg(cufinufftf_plan forward, cufinufftf_plan adjoint,
               OperatorBuffers& buffers, CgBuffers& cg,
               float inverse_operator_scale_squared, float scaled_lambda,
               int cg_iterations, cudaStream_t stream) {
  launch_adjoint(adjoint, buffers, cg.measurement, cg.b, nullptr,
                 inverse_operator_scale_squared, 0.0f, stream);
  cg_initialize<<<kBlocks, kThreads, 0, stream>>>(cg.b, cg.x, cg.r, cg.p);
  launch_dot(cg.r, cg.r, cg.partial, cg.rho, stream);
  initialize_scalar_state<<<1, 1, 0, stream>>>(
      cg.rho, cg.residual_history, cg.positive_curvature);
  for (int iteration = 0; iteration < cg_iterations; ++iteration) {
    launch_normal(forward, adjoint, buffers, cg.p, cg.ap,
                  inverse_operator_scale_squared, scaled_lambda, stream);
    launch_dot(cg.p, cg.ap, cg.partial, cg.denominator, stream);
    compute_alpha<<<1, 1, 0, stream>>>(
        cg.rho, cg.denominator, cg.alpha, cg.positive_curvature);
    update_x_r<<<kBlocks, kThreads, 0, stream>>>(
        cg.x, cg.r, cg.p, cg.ap, cg.alpha);
    launch_dot(cg.r, cg.r, cg.partial, cg.rho_new, stream);
    compute_beta<<<1, 1, 0, stream>>>(
        cg.rho, cg.rho_new, cg.beta, cg.residual_history, iteration);
    update_p<<<kBlocks, kThreads, 0, stream>>>(cg.p, cg.r, cg.beta);
  }
}

size_t count_nonfinite(const float2* device, size_t count) {
  std::vector<float2> host(count);
  CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(float2),
                        cudaMemcpyDeviceToHost));
  return static_cast<size_t>(std::count_if(
      host.begin(), host.end(), [](float2 value) {
        return !std::isfinite(value.x) || !std::isfinite(value.y);
      }));
}

float calibrate_operator_scale(cufinufftf_plan forward,
                               cufinufftf_plan adjoint,
                               OperatorBuffers& buffers, float2* truth,
                               float2* scratch, cudaStream_t stream) {
  launch_normal(forward, adjoint, buffers, truth, scratch, 1.0f, 0.0f, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<float2> host(kPixels);
  CUDA_CHECK(cudaMemcpy(host.data(), scratch, kPixels * sizeof(float2),
                        cudaMemcpyDeviceToHost));
  float maximum = 0.0f;
  for (const float2 value : host) {
    maximum = std::max(maximum, std::hypot(value.x, value.y));
  }
  return std::sqrt(std::max(maximum, 1.0e-12f));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(
        stderr,
        "usage: %s TRAJECTORY_BIN RUNTIME_DIR [warmup=5] [iters=20] "
        "[stress=0] [cg_iterations=10] [method=1] [eps=5e-4] [dump_dir]\n",
        argv[0]);
    return 2;
  }
  const std::string trajectory_path = argv[1];
  const std::string runtime_directory = argv[2];
  const int warmup = argc > 3 ? std::atoi(argv[3]) : 5;
  const int iterations = argc > 4 ? std::atoi(argv[4]) : 20;
  const int stress = argc > 5 ? std::atoi(argv[5]) : 0;
  const int cg_iterations = argc > 6 ? std::atoi(argv[6]) : 10;
  const int method = argc > 7 ? std::atoi(argv[7]) : 1;
  const float tolerance = argc > 8 ? std::strtof(argv[8], nullptr) : 5.0e-4f;
  const bool dump = argc > 9;
  const std::string dump_directory = dump ? argv[9] : "";
  if (warmup < 0 || iterations <= 0 || stress < 0 || cg_iterations <= 0 ||
      method <= 0 || !(tolerance > 0.0f)) {
    return 2;
  }
  const char* forward_method_env =
      std::getenv("JKF_CUFINUFFT_FORWARD_METHOD");
  const char* adjoint_method_env =
      std::getenv("JKF_CUFINUFFT_ADJOINT_METHOD");
  const char* gpu_sort_env = std::getenv("JKF_CUFINUFFT_GPU_SORT");
  const char* operator_scale_env =
      std::getenv("JKF_CUFINUFFT_OPERATOR_SCALE");
  const int forward_method =
      forward_method_env ? std::atoi(forward_method_env) : method;
  const int adjoint_method =
      adjoint_method_env ? std::atoi(adjoint_method_env) : method;
  const int gpu_sort = gpu_sort_env ? std::atoi(gpu_sort_env) : 1;
  if (forward_method <= 0 || adjoint_method <= 0 ||
      (gpu_sort != 0 && gpu_sort != 1)) {
    return 2;
  }

  const auto trajectory = read_array<float>(trajectory_path, kSamples * 2);
  std::vector<float> coordinate_x(kSamples);
  std::vector<float> coordinate_y(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    // MRI-NUFFT reverses the last two sample axes for C-order image modes.
    coordinate_x[sample] = trajectory[sample * 2 + 1] * (2.0f * kPi);
    coordinate_y[sample] = trajectory[sample * 2] * (2.0f * kPi);
  }
  const auto measurement = read_array<float2>(
      runtime_directory + "/measurement.c64.bin",
      static_cast<size_t>(kCoils) * kSamples);
  const auto sensitivity_maps = read_array<float2>(
      runtime_directory + "/sensitivity_maps.c64.bin",
      static_cast<size_t>(kCoils) * kPixels);
  std::vector<float2> truth;
  if (operator_scale_env == nullptr) {
    truth = read_array<float2>(runtime_directory + "/truth.c64.bin", kPixels);
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  float* d_x = allocate_device<float>(kSamples);
  float* d_y = allocate_device<float>(kSamples);
  CUDA_CHECK(cudaMemcpy(d_x, coordinate_x.data(), kSamples * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, coordinate_y.data(), kSamples * sizeof(float),
                        cudaMemcpyHostToDevice));

  OperatorBuffers buffers;
  buffers.sensitivity_maps =
      allocate_device<float2>(static_cast<size_t>(kCoils) * kPixels);
  buffers.coil_modes =
      allocate_device<float2>(static_cast<size_t>(kCoils) * kPixels);
  buffers.samples =
      allocate_device<float2>(static_cast<size_t>(kCoils) * kSamples);
  CUDA_CHECK(cudaMemcpy(buffers.sensitivity_maps, sensitivity_maps.data(),
                        sensitivity_maps.size() * sizeof(float2),
                        cudaMemcpyHostToDevice));

  CgBuffers cg;
  cg.measurement =
      allocate_device<float2>(static_cast<size_t>(kCoils) * kSamples);
  cg.b = allocate_device<float2>(kPixels);
  cg.x = allocate_device<float2>(kPixels);
  cg.r = allocate_device<float2>(kPixels);
  cg.p = allocate_device<float2>(kPixels);
  cg.ap = allocate_device<float2>(kPixels);
  cg.partial = allocate_device<float>(kBlocks);
  cg.rho = allocate_device<float>(1);
  cg.rho_new = allocate_device<float>(1);
  cg.denominator = allocate_device<float>(1);
  cg.alpha = allocate_device<float>(1);
  cg.beta = allocate_device<float>(1);
  cg.residual_history = allocate_device<float>(cg_iterations + 1);
  cg.positive_curvature = allocate_device<int>(1);
  CUDA_CHECK(cudaMemcpy(cg.measurement, measurement.data(),
                        measurement.size() * sizeof(float2),
                        cudaMemcpyHostToDevice));
  float2* d_truth = nullptr;
  if (operator_scale_env == nullptr) {
    d_truth = allocate_device<float2>(kPixels);
    CUDA_CHECK(cudaMemcpy(d_truth, truth.data(), kPixels * sizeof(float2),
                          cudaMemcpyHostToDevice));
  }

  cufinufft_opts options;
  cufinufft_default_opts(&options);
  options.gpu_device_id = 0;
  options.gpu_stream = stream;
  options.gpu_maxbatchsize = kCoils;
  options.gpu_method = forward_method;
  options.gpu_sort = gpu_sort;
  options.modeord = 0;
  const auto setup_started = std::chrono::steady_clock::now();
  int64_t modes[3] = {kImage, kImage, 1};
  cufinufftf_plan forward = nullptr;
  cufinufftf_plan adjoint = nullptr;
  CUFINUFFT_CHECK(cufinufftf_makeplan(2, 2, modes, -1, kCoils, tolerance,
                                      &forward, &options));
  options.gpu_method = adjoint_method;
  CUFINUFFT_CHECK(cufinufftf_makeplan(1, 2, modes, +1, kCoils, tolerance,
                                      &adjoint, &options));
  CUFINUFFT_CHECK(cufinufftf_setpts(forward, kSamples, d_x, d_y, nullptr, 0,
                                    nullptr, nullptr, nullptr));
  CUFINUFFT_CHECK(cufinufftf_setpts(adjoint, kSamples, d_x, d_y, nullptr, 0,
                                    nullptr, nullptr, nullptr));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const double setup_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - setup_started)
                              .count();

  const float operator_scale =
      operator_scale_env != nullptr
          ? std::strtof(operator_scale_env, nullptr)
          : calibrate_operator_scale(
                forward, adjoint, buffers, d_truth, cg.ap, stream);
  if (!(operator_scale > 0.0f) || !std::isfinite(operator_scale)) {
    std::fprintf(stderr, "invalid operator scale\n");
    return 2;
  }
  const float inverse_scale_squared = 1.0f / (operator_scale * operator_scale);
  const float scaled_lambda = kLambda * inverse_scale_squared;

  launch_cg(forward, adjoint, buffers, cg, inverse_scale_squared,
            scaled_lambda, cg_iterations, stream);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int positive_curvature = 0;
  std::vector<float> residuals(cg_iterations + 1);
  CUDA_CHECK(cudaMemcpy(&positive_curvature, cg.positive_curvature, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(residuals.data(), cg.residual_history,
                        residuals.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const size_t nonfinite = count_nonfinite(cg.x, kPixels);
  if (!positive_curvature || nonfinite != 0) {
    std::fprintf(stderr, "CG correctness failure curvature=%d nonfinite=%zu\n",
                 positive_curvature, nonfinite);
    return 3;
  }

  const auto launch_once = [&]() {
    launch_cg(forward, adjoint, buffers, cg, inverse_scale_squared,
              scaled_lambda, cg_iterations, stream);
  };
  for (int iteration = 0; iteration < warmup; ++iteration) launch_once();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int iteration = 0; iteration < iterations; ++iteration) launch_once();
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  for (int iteration = 0; iteration < stress; ++iteration) launch_once();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (dump) {
    write_device_array(dump_directory + "/reconstruction.c64.bin", cg.x,
                       kPixels);
    write_device_array(dump_directory + "/residuals.f32.bin",
                       cg.residual_history, residuals.size());
  }
  std::printf(
      "{\"variant\":\"cufinufft_native_cg10\",\"correct\":true,"
      "\"us\":%.9g,\"setup_ms\":%.9g,\"cg_iterations\":%d,"
      "\"operator_scale\":%.9g,\"residual0\":%.9g,"
      "\"residual_final\":%.9g,\"eps\":%.9g,"
      "\"forward_method\":%d,\"adjoint_method\":%d,\"gpu_sort\":%d,"
      "\"operator_scale_source\":\"%s\","
      "\"nonfinite\":0}\n",
      static_cast<double>(elapsed_ms) * 1000.0 / iterations, setup_ms,
      cg_iterations, operator_scale, residuals.front(), residuals.back(),
      tolerance, forward_method, adjoint_method, gpu_sort,
      operator_scale_env != nullptr ? "fixed_env" : "truth_calibration");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cufinufftf_destroy(forward);
  cufinufftf_destroy(adjoint);
  cudaStreamDestroy(stream);
  cudaFree(d_x);
  cudaFree(d_y);
  if (d_truth != nullptr) cudaFree(d_truth);
  cudaFree(buffers.sensitivity_maps);
  cudaFree(buffers.coil_modes);
  cudaFree(buffers.samples);
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
  return 0;
}
