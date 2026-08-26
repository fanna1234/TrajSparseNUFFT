#!/usr/bin/env bash
set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${experiment_dir}/../.." && pwd)
production_dir="${repo_root}/experiments/production"
source_dir="${production_dir}/src"
build_dir=${1:-"${experiment_dir}/build"}
source "${repo_root}/tools/cuda_env.sh"
resolve_cuda_toolchain

mkdir -p "${build_dir}"

common_flags=(
  -std=c++17
  -O3
  -lineinfo
  -arch=sm_120a
  -Xptxas=-v
)

f30_defs=(
  -DJKF_CUSTOM_FFT_CANDIDATE=1
  -DJKF_CUSTOM_FFT_SHARED16=1
  -DJKF_CUSTOM_FFT_SHARED16_SKIP_ZERO_ROWS=1
  -DJKF_CUSTOM_FFT_SHARED16_ZERO_ROWS_FLAT=1
  -DJKF_CUSTOM_FFT_Y_SHARED16=1
  -DJKF_CUSTOM_FFT_MIXED32=1
  -DJKF_CUSTOM_FFT_X_DIRECT_TRANSPOSED=1
  -DJKF_CUSTOM_FFT_PLANAR_SHARED=1
  -DJKF_CUSTOM_FFT_ALIGNED_PAIR_SWIZZLE=1
  -DJKF_CUSTOM_FFT_PHASED_PLANAR_SHARED=1
  -DJKF_CUSTOM_FFT_Y_N16_FUSED=1
  -DJKF_CUSTOM_FFT_FUSED_WRITER_ROLLED=1
  -DJKF_CUSTOM_FFT_FUSED_PHASE_ROLLED=1
  -DJKF_CUSTOM_FFT_Y_N16_FUSED_KX1=1
)

"${NVCC_BIN}" "${common_flags[@]}" "${f30_defs[@]}" \
  -DJKF_CUSTOM_IFFT_F31=1 -DJKF_CUSTOM_IFFT_REGULAR_X=1 \
  -DJKF_CUSTOM_IFFT_REGULAR_Y_CROP=1 \
  -DJKF_CUSTOM_FFT_ROWS8=1 -DJKF_CUSTOM_FFT_ROWS8_DIRECT_TRANSPOSED=1 \
  -DJKF_CUSTOM_FFT_ROWS8_INTERLEAVED_TRANSPOSED=1 \
  -DJKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE=1 \
  -DJKF_CUSTOM_FFT_REGULAR_SHARED_SWIZZLE_FAST=1 \
  -DJKF_CUSTOM_FFT_NO_MAIN=1 -DJKF_CUSTOM_FFT_NO_CUFFT_RUNTIME=1 -c \
  "${source_dir}/nufft_custom_fft_forward_bench.cu" \
  -o "${build_dir}/nufft_custom_fft_f30_split2_api.o"

"${NVCC_BIN}" "${common_flags[@]}" \
  "${source_dir}/nufft_fp16x2_cg_bench.cu" \
  "${build_dir}/nufft_custom_fft_f30_split2_api.o" \
  -o "${build_dir}/nufft_fp16x2_cg_sparse_control"

"${NVCC_BIN}" "${common_flags[@]}" -DJKF_DENSE_TC_CONTROL=1 \
  "${source_dir}/nufft_fp16x2_cg_bench.cu" \
  "${build_dir}/nufft_custom_fft_f30_split2_api.o" \
  -o "${build_dir}/nufft_fp16x2_cg_dense_control"

for variant in sparse_control dense_control; do
  binary="${build_dir}/nufft_fp16x2_cg_${variant}"
  "${CUOBJDUMP_BIN}" --dump-resource-usage "${binary}" \
    > "${build_dir}/${variant}.resources.txt"
  "${CUOBJDUMP_BIN}" --dump-sass "${binary}" \
    > "${build_dir}/${variant}.sass"
  if nm -D "${binary}" | grep -qi 'cufft'; then
    echo "unexpected cuFFT dependency in ${variant}" >&2
    exit 3
  fi
done

sha256sum \
  "${source_dir}/nufft_custom_fft_forward_bench.cu" \
  "${source_dir}/nufft_spmma_bench.cu" \
  "${source_dir}/nufft_dense_control.cuh" \
  "${source_dir}/nufft_integrated_bench.cu" \
  "${source_dir}/nufft_fp16x2_cg_bench.cu" \
  "${build_dir}/nufft_custom_fft_f30_split2_api.o" \
  "${build_dir}/nufft_fp16x2_cg_sparse_control" \
  "${build_dir}/nufft_fp16x2_cg_dense_control" \
  > "${build_dir}/sha256sums.txt"

cat "${build_dir}/sha256sums.txt"
