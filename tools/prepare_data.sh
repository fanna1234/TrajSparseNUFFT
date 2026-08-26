#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
gpu_python=${TRAJSPARSE_GPU_PYTHON:-python3}
data_root=${TRAJSPARSE_DATA_ROOT:-"${repo_root}/data/ocmr"}
prepared_root=${TRAJSPARSE_PREPARED_ROOT:-"${repo_root}/reproduced-data"}

if [[ "${gpu_python}" != */* ]]; then
  gpu_python=$(command -v "${gpu_python}" || true)
fi
if [[ -z "${gpu_python}" || ! -x "${gpu_python}" ]]; then
  echo "GPU Python interpreter not found; set TRAJSPARSE_GPU_PYTHON" >&2
  exit 2
fi

if [[ -d "${prepared_root}" && -n "$(find "${prepared_root}" -mindepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite nonempty ${prepared_root}" >&2
  exit 2
fi
mkdir -p "${prepared_root}"

bash "${repo_root}/tools/get_ocmr.sh" "${data_root}"
"${gpu_python}" -c \
  'import h5py, ismrmrd, numpy, skimage, torch, torchkbnufft' \
  || {
    echo "GPU/data Python environment is incomplete; see docs/REPRODUCE.md" >&2
    exit 2
  }
"${gpu_python}" - <<'PY'
import os
import sys
import numpy
import torch
import torchkbnufft

observed = {
    "python": ".".join(map(str, sys.version_info[:3])),
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torchkbnufft": getattr(torchkbnufft, "__version__", "unknown"),
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
matched = (
    sys.version_info[:2] == (3, 13)
    and observed["numpy"] == "2.2.6"
    and observed["torch"].startswith("2.8.0")
    and observed["torchkbnufft"] == "1.5.2"
    and observed["cuda_available"]
)
if not matched and os.environ.get("TRAJSPARSE_ALLOW_VERSION_DRIFT") != "1":
    raise SystemExit(
        "GPU Python environment differs from the measured contract: "
        + repr(observed)
        + "; set TRAJSPARSE_ALLOW_VERSION_DRIFT=1 for a labeled portability run"
    )
print("[OK] GPU Python environment", observed)
PY

trajectory_root="${repo_root}/experiments/data_prep/trajectories"
"${gpu_python}" "${repo_root}/experiments/data_prep/generate_trajectories.py" \
  --output "${trajectory_root}"

cases_root="${prepared_root}/cases"
"${gpu_python}" "${repo_root}/experiments/data_prep/prepare_ocmr_case.py" \
  --input "${data_root}/fs_0152_0_55T.h5" --output "${cases_root}/fs0152"
"${gpu_python}" "${repo_root}/experiments/data_prep/prepare_ocmr_case.py" \
  --input "${data_root}/fs_0005_1_5T.h5" --output "${cases_root}/fs0005_v2"
"${gpu_python}" "${repo_root}/experiments/data_prep/prepare_ocmr_case.py" \
  --input "${data_root}/fs_0016_3T.h5" --output "${cases_root}/fs0016_v2"
"${gpu_python}" "${repo_root}/tools/check_prepared_cases.py" "${cases_root}"

export_runtime() {
  local case_root=$1
  local runtime_root=$2
  local frame=$3
  "${gpu_python}" "${repo_root}/experiments/data_prep/export_real_runtime_inputs.py" \
    --case "${case_root}" --frame "${frame}" --measurement-scale 512 \
    --trajectory "spiral65=${trajectory_root}/spiral_standard_256.npy" \
    --trajectory "radial65=${trajectory_root}/radial_256x256.npy" \
    --trajectory "golden65=${trajectory_root}/golden_256x256.npy" \
    --output "${runtime_root}/frame$(printf '%02d' "${frame}")"
}

development_runtime="${prepared_root}/runtime/development"
heldout_runtime="${prepared_root}/runtime/heldout"
for frame in 0 1 2; do
  export_runtime "${cases_root}/fs0152" "${development_runtime}" "${frame}"
  export_runtime "${cases_root}/fs0005_v2" "${heldout_runtime}/fs0005" "${frame}"
  export_runtime "${cases_root}/fs0016_v2" "${heldout_runtime}/fs0016" "${frame}"
done
"${gpu_python}" "${repo_root}/tools/build_runtime_manifest.py" \
  "${development_runtime}" --expected 3
"${gpu_python}" "${repo_root}/tools/build_runtime_manifest.py" \
  "${heldout_runtime}" --expected 6

packed_root="${prepared_root}/packed"
"${gpu_python}" "${repo_root}/experiments/packing/run_fast_planner_matrix.py" \
  --repo-root "${repo_root}" --python "${gpu_python}" --output "${packed_root}"

dense_root="${prepared_root}/dense"
for case_name in spiral65 radial65 golden65; do
  "${gpu_python}" "${repo_root}/experiments/packing/prepare_dense_control.py" \
    --real "${packed_root}/${case_name}/G/real" \
    --imag "${packed_root}/${case_name}/G/imag" \
    --output "${dense_root}/${case_name}/g"
  "${gpu_python}" "${repo_root}/experiments/packing/prepare_dense_control.py" \
    --real "${packed_root}/${case_name}/G_T/real" \
    --imag "${packed_root}/${case_name}/G_T/imag" \
    --output "${dense_root}/${case_name}/gt"
done

layout_file="${prepared_root}/layout.env"
{
  printf 'export TRAJSPARSE_DEVELOPMENT_CASE=%q\n' "${cases_root}/fs0152"
  printf 'export TRAJSPARSE_DEVELOPMENT_RUNTIME_ROOT=%q\n' "${development_runtime}"
  printf 'export TRAJSPARSE_HELDOUT_CASES_ROOT=%q\n' "${cases_root}"
  printf 'export TRAJSPARSE_HELDOUT_RUNTIME_ROOT=%q\n' "${heldout_runtime}"
  printf 'export TRAJSPARSE_PERFORMANCE_RUNTIME_ROOT=%q\n' "${development_runtime}"
  printf 'export TRAJSPARSE_PACKED_ROOT=%q\n' "${packed_root}"
  printf 'export TRAJSPARSE_DENSE_ROOT=%q\n' "${dense_root}"
} > "${layout_file}"
echo "[OK] prepared data; source ${layout_file} before GPU runs"
