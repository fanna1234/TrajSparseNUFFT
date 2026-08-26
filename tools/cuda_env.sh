#!/usr/bin/env bash

resolve_cuda_toolchain() {
  NVCC_BIN=${NVCC:-$(command -v nvcc || true)}
  if [[ -z "${NVCC_BIN}" || ! -x "${NVCC_BIN}" ]]; then
    echo "nvcc not found; set NVCC or add CUDA to PATH" >&2
    return 2
  fi

  local release major minor
  release=$("${NVCC_BIN}" --version | sed -n \
    's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' | tail -n 1)
  if [[ -z "${release}" ]]; then
    echo "unable to parse CUDA version from ${NVCC_BIN}" >&2
    return 2
  fi
  read -r major minor <<<"${release}"
  if (( major < 12 || (major == 12 && minor < 8) || major > 13 || (major == 13 && minor > 3) )); then
    echo "CUDA ${major}.${minor} is outside the supported 12.8--13.3 range" >&2
    return 2
  fi

  CUDA_ARCH=${CUDA_ARCH:-sm_120a}
  if [[ "${CUDA_ARCH}" != "sm_120a" ]]; then
    echo "this implementation is validated only for CUDA_ARCH=sm_120a" >&2
    return 2
  fi

  local cuda_bin
  cuda_bin=$(cd "$(dirname "${NVCC_BIN}")" && pwd)
  CUOBJDUMP_BIN=${CUOBJDUMP:-"${cuda_bin}/cuobjdump"}
  if [[ ! -x "${CUOBJDUMP_BIN}" ]]; then
    echo "cuobjdump not found; set CUOBJDUMP" >&2
    return 2
  fi

  export NVCC_BIN CUOBJDUMP_BIN CUDA_ARCH
  echo "CUDA ${major}.${minor}; arch=${CUDA_ARCH}; nvcc=${NVCC_BIN}"
}
