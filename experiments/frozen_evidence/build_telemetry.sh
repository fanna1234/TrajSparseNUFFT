#!/usr/bin/env bash
set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${experiment_dir}/../.." && pwd)
production_dir="${repo_root}/experiments/production"
build_dir=${1:-"${experiment_dir}/build_telemetry"}
source "${repo_root}/tools/cuda_env.sh"
resolve_cuda_toolchain
production_build_dir=${PRODUCTION_BUILD_DIR:-"${production_dir}/build"}
fft_object="${production_build_dir}/nufft_custom_fft_f30_split2_api.o"

if [[ ! -f "${fft_object}" ]]; then
  echo "missing production FFT object; run ./reproduce.sh build first" >&2
  exit 2
fi

mkdir -p "${build_dir}"
"${NVCC_BIN}" \
  -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_120a,code=sm_120a \
  -Xptxas=-v -DJKF_CG_TELEMETRY=1 \
  "${production_dir}/src/nufft_fp16x2_cg_bench.cu" \
  "${fft_object}" \
  -o "${build_dir}/nufft_fp16x2_cg_telemetry" \
  2>&1 | tee "${build_dir}/build.log"

sha256sum \
  "${production_dir}/src/nufft_fp16x2_cg_bench.cu" \
  "${fft_object}" \
  "${build_dir}/nufft_fp16x2_cg_telemetry" \
  > "${build_dir}/sha256sums.txt"

"${CUOBJDUMP_BIN}" --dump-resource-usage \
  "${build_dir}/nufft_fp16x2_cg_telemetry" \
  > "${build_dir}/resource_usage.txt"
"${CUOBJDUMP_BIN}" --dump-sass \
  "${build_dir}/nufft_fp16x2_cg_telemetry" \
  > "${build_dir}/telemetry.sass"
