#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${repo_root}/tools/cuda_env.sh"
resolve_cuda_toolchain

for command_name in git cmake python3; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "missing required command: ${command_name}" >&2
    exit 2
  }
done
cmake_version=$(cmake --version | sed -n '1s/[^0-9]*\([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p')
read -r cmake_major cmake_minor <<<"${cmake_version}"
if [[ -z "${cmake_major:-}" ]] \
  || (( cmake_major < 3 || (cmake_major == 3 && cmake_minor < 25) )); then
  echo "CMake 3.25 or newer is required" >&2
  exit 2
fi

source_root=${FINUFFT_ROOT:-"${repo_root}/third_party/finufft"}
build_root=${CUFINUFFT_BUILD:-"${repo_root}/build-cufinufft"}
repository_url=https://github.com/flatironinstitute/finufft.git
commit=$(python3 -c '
import json, pathlib
record = json.loads(pathlib.Path("'"${repo_root}"'/experiments/baselines/baselines.lock.json").read_text())
print(next(row["commit"] for row in record["entries"] if row["name"] == "finufft-cufinufft"))
')

if [[ ! -d "${source_root}/.git" ]]; then
  if [[ -e "${source_root}" ]]; then
    echo "FINUFFT_ROOT exists but is not a Git checkout: ${source_root}" >&2
    exit 2
  fi
  git clone "${repository_url}" "${source_root}"
fi
if [[ "$(git -C "${source_root}" remote get-url origin)" != "${repository_url}" ]]; then
  echo "unexpected FINUFFT origin in ${source_root}" >&2
  exit 2
fi
if [[ -n "$(git -C "${source_root}" status --porcelain)" ]]; then
  echo "refusing to modify dirty FINUFFT checkout: ${source_root}" >&2
  exit 2
fi
if ! git -C "${source_root}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
  git -C "${source_root}" fetch origin "${commit}"
fi
git -C "${source_root}" checkout --detach "${commit}"

cmake -S "${source_root}" -B "${build_root}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${NVCC_BIN}" \
  -DCMAKE_CUDA_ARCHITECTURES=OFF \
  -DCMAKE_CUDA_FLAGS="-gencode=arch=compute_120a,code=sm_120a" \
  -DFINUFFT_USE_CPU=OFF \
  -DFINUFFT_USE_CUDA=ON \
  -DFINUFFT_STATIC_LINKING=ON \
  -DFINUFFT_BUILD_TESTS=OFF \
  -DFINUFFT_BUILD_EXAMPLES=OFF
cmake --build "${build_root}" --target cufinufft finufft_common \
  --parallel "${BUILD_JOBS:-8}"

test -f "${build_root}/libcufinufft.a"
test -f "${build_root}/src/common/libfinufft_common.a"
echo "[OK] cuFINUFFT ${commit} built in ${build_root}"
