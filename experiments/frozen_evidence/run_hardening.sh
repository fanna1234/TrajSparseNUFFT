#!/usr/bin/env bash
set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${experiment_dir}/../.." && pwd)
source "${repo_root}/tools/runtime_env.sh"
results_root=${TRAJSPARSE_RESULTS_ROOT:-"${repo_root}/reproduced-results"}
output=${1:-"${results_root}/hardening"}
if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite nonempty ${output}" >&2
  exit 2
fi
mkdir -p "${output}"

production="${repo_root}/experiments/production/build/nufft_fp16x2_cg"
dense="${repo_root}/experiments/dense_control/build/nufft_fp16x2_cg_dense_control"
cufinufft="${repo_root}/experiments/cufinufft_baseline/build/cufinufft_native_cg"
pack="${TRAJSPARSE_PACKED_ROOT:?source reproduced-data/layout.env}/golden65"
dense_control="${TRAJSPARSE_DENSE_ROOT:?source reproduced-data/layout.env}/golden65"
runtime="${TRAJSPARSE_HELDOUT_RUNTIME_ROOT:?source reproduced-data/layout.env}/fs0005/frame02/golden65"
trajectory="${repo_root}/experiments/data_prep/trajectories/golden_256x256.f32xy.bin"
sanitizer=${COMPUTE_SANITIZER:-$(command -v compute-sanitizer || true)}
if [[ -z "${sanitizer}" || ! -x "${sanitizer}" ]]; then
  echo "compute-sanitizer not found; set COMPUTE_SANITIZER" >&2
  exit 2
fi

runtime_no_truth="${output}/cufinufft_runtime_no_truth"
mkdir -p "${runtime_no_truth}"
ln -s "${runtime}/measurement.c64.bin" "${runtime_no_truth}/measurement.c64.bin"
ln -s "${runtime}/sensitivity_maps.c64.bin" "${runtime_no_truth}/sensitivity_maps.c64.bin"

nvidia-smi \
  --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,memory.used,memory.total \
  --format=csv,noheader > "${output}/gpu_before.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader > "${output}/processes_before.csv"

with_lock() {
  flock /tmp/trajsparsenufft_gpu.lock "$@"
}

with_lock "${sanitizer}" --tool memcheck --error-exitcode=99 \
  "${production}" \
  "${pack}/G/real" "${pack}/G/imag" \
  "${pack}/G_T/real" "${pack}/G_T/imag" \
  "${runtime}" 1 0 1 0 0 10 \
  > "${output}/production_memcheck.log" 2>&1

with_lock env \
  JKF_DENSE_FWD_CONTROL="${dense_control}/g" \
  JKF_DENSE_ADJ_CONTROL="${dense_control}/gt" \
  "${sanitizer}" --tool memcheck --error-exitcode=99 \
  "${dense}" \
  "${pack}/G/real" "${pack}/G/imag" \
  "${pack}/G_T/real" "${pack}/G_T/imag" \
  "${runtime}" 1 0 1 0 0 10 \
  > "${output}/dense_memcheck.log" 2>&1

with_lock env \
  JKF_CUFINUFFT_FORWARD_METHOD=1 \
  JKF_CUFINUFFT_ADJOINT_METHOD=2 \
  JKF_CUFINUFFT_GPU_SORT=0 \
  JKF_CUFINUFFT_OPERATOR_SCALE=512 \
  "${sanitizer}" --tool memcheck --error-exitcode=99 \
  "${cufinufft}" "${trajectory}" "${runtime_no_truth}" \
  0 1 0 10 2 5e-4 \
  > "${output}/cufinufft_memcheck.log" 2>&1

for replica in 0 1; do
  with_lock "${production}" \
    "${pack}/G/real" "${pack}/G/imag" \
    "${pack}/G_T/real" "${pack}/G_T/imag" \
    "${runtime}" 1 0 1 100 0 10 \
    > "${output}/production_stress100_${replica}.log" 2>&1

  with_lock env \
    JKF_CUFINUFFT_FORWARD_METHOD=1 \
    JKF_CUFINUFFT_ADJOINT_METHOD=2 \
    JKF_CUFINUFFT_GPU_SORT=0 \
    JKF_CUFINUFFT_OPERATOR_SCALE=512 \
    "${cufinufft}" "${trajectory}" "${runtime_no_truth}" \
    0 1 100 10 2 5e-4 \
    > "${output}/cufinufft_stress100_${replica}.log" 2>&1
done

with_lock env \
  JKF_DENSE_FWD_CONTROL="${dense_control}/g" \
  JKF_DENSE_ADJ_CONTROL="${dense_control}/gt" \
  "${dense}" \
  "${pack}/G/real" "${pack}/G/imag" \
  "${pack}/G_T/real" "${pack}/G_T/imag" \
  "${runtime}" 1 0 1 100 0 10 \
  > "${output}/dense_stress100.log" 2>&1

nvidia-smi \
  --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,memory.used,memory.total \
  --format=csv,noheader > "${output}/gpu_after.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader > "${output}/processes_after.csv"

sha256sum \
  "${production}" "${dense}" "${cufinufft}" \
  "${repo_root}/experiments/production/src/nufft_fp16x2_cg_bench.cu" \
  "${repo_root}/experiments/cufinufft_baseline/cufinufft_native_cg.cu" \
  > "${output}/sha256sums.txt"

"${TRAJSPARSE_GPU_PYTHON:-python3}" - "${output}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
memchecks = sorted(root.glob("*_memcheck.log"))
stress = sorted(root.glob("*_stress100*.log"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary = {
    "experiment_id": "frozen_evidence",
    "scale_rule": {"raw_scale": 512.0, "native_scale": 1.0, "truth_free": True},
    "cufinufft_runtime_contains_truth": (root / "cufinufft_runtime_no_truth/truth.c64.bin").exists(),
    "memcheck": [
        {
            "log": str(path),
            "sha256": sha(path),
            "zero_errors": "ERROR SUMMARY: 0 errors" in path.read_text(),
        }
        for path in memchecks
    ],
    "stress": [
        {
            "log": str(path),
            "sha256": sha(path),
            "correct": '"correct":true' in path.read_text(),
        }
        for path in stress
    ],
}
summary["all_memcheck_pass"] = all(row["zero_errors"] for row in summary["memcheck"])
summary["all_stress_pass"] = all(row["correct"] for row in summary["stress"])
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
