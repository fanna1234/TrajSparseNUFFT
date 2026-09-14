#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${repo_root}/tools/runtime_env.sh"
output_root=${1:-"${TRAJSPARSE_DATA_ROOT:-${repo_root}/data/ocmr}"}
mkdir -p "${output_root}"

command -v curl >/dev/null 2>&1 || {
  echo "curl is required" >&2
  exit 2
}
if ! command -v sha256sum >/dev/null 2>&1 \
  && ! command -v shasum >/dev/null 2>&1; then
  echo "sha256sum or shasum is required" >&2
  exit 2
fi

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

download_one() {
  local name=$1
  local size=$2
  local expected_sha=$3
  local url=$4
  local target="${output_root}/${name}"
  local partial="${target}.part"

  if [[ -f "${target}" ]]; then
    local existing_size existing_sha
    existing_size=$(wc -c < "${target}" | tr -d ' ')
    existing_sha=$(file_sha256 "${target}")
    if [[ "${existing_size}" == "${size}" && "${existing_sha}" == "${expected_sha}" ]]; then
      echo "[OK] ${name}"
      return
    fi
    echo "existing file fails size or SHA-256 validation: ${target}" >&2
    exit 2
  fi

  curl --fail --location --retry 3 --continue-at - --output "${partial}" "${url}"
  local actual_size actual_sha
  actual_size=$(wc -c < "${partial}" | tr -d ' ')
  actual_sha=$(file_sha256 "${partial}")
  if [[ "${actual_size}" != "${size}" || "${actual_sha}" != "${expected_sha}" ]]; then
    echo "download validation failed for ${name}" >&2
    echo "size=${actual_size}; sha256=${actual_sha}" >&2
    exit 2
  fi
  mv "${partial}" "${target}"
  echo "[OK] ${name}"
}

download_one \
  fs_0152_0_55T.h5 72027848 \
  65ff79868b4b273aeee550c5bbf3e4ce266d12f0d9c23d435e406d5467e026b5 \
  https://ocmr.s3.us-east-2.amazonaws.com/data/fs_0152_0_55T.h5
download_one \
  fs_0005_1_5T.h5 220258096 \
  1f3beee40b9186337b18f5acb6ff803c03e7b3ce4b68b3d8dc9b619cbcd320d4 \
  https://ocmr.s3.us-east-2.amazonaws.com/data/fs_0005_1_5T.h5
download_one \
  fs_0016_3T.h5 417590520 \
  763525771bf7617a2e5cea5c3d106f503df553d2de78f20d5fd2a3f6d4404740 \
  https://ocmr.s3.us-east-2.amazonaws.com/data/fs_0016_3T.h5
