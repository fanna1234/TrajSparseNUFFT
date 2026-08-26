#!/usr/bin/env bash
set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${experiment_dir}/../.." && pwd)
build_dir=${1:-"${experiment_dir}/build"}
source "${repo_root}/tools/cuda_env.sh"
resolve_cuda_toolchain
finufft_root=${FINUFFT_ROOT:-"${repo_root}/third_party/finufft"}
cufinufft_build=${CUFINUFFT_BUILD:-"${repo_root}/build-cufinufft"}

required_files=(
  "${finufft_root}/include/cufinufft.h"
  "${cufinufft_build}/libcufinufft.a"
  "${cufinufft_build}/src/common/libfinufft_common.a"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "missing cuFINUFFT build input: ${required_file}" >&2
    exit 2
  fi
done

mkdir -p "${build_dir}"
"${NVCC_BIN}" \
  -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_120a,code=sm_120a \
  -Xptxas=-v \
  -I"${finufft_root}/include" \
  "${experiment_dir}/cufinufft_native_cg.cu" \
  "${cufinufft_build}/libcufinufft.a" \
  "${cufinufft_build}/src/common/libfinufft_common.a" \
  -lcufft -lcudart -ldl -lrt -lm \
  -o "${build_dir}/cufinufft_native_cg" \
  2>&1 | tee "${build_dir}/build.log"

sha256sum \
  "${experiment_dir}/cufinufft_native_cg.cu" \
  "${build_dir}/cufinufft_native_cg" \
  > "${build_dir}/sha256sums.txt"

"${CUOBJDUMP_BIN}" --dump-resource-usage "${build_dir}/cufinufft_native_cg" \
  > "${build_dir}/resource_usage.txt"
