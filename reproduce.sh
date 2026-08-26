#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${repo_root}"
group=${1:-help}
if [[ $# -gt 0 ]]; then
  shift
fi
dry_run=0
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 1 || "$1" != "--dry-run" ]]; then
    echo "usage: ./reproduce.sh <group> [--dry-run]" >&2
    exit 2
  fi
  dry_run=1
fi

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ ${dry_run} -eq 0 ]]; then
    "$@"
  fi
}

required_value() {
  local variable_name=$1
  local value=${!variable_name:-}
  if [[ -n "${value}" ]]; then
    printf '%s' "${value}"
  elif [[ ${dry_run} -eq 1 ]]; then
    printf '${%s}' "${variable_name}"
  else
    echo "required environment variable is unset: ${variable_name}" >&2
    exit 2
  fi
}

results_root=${TRAJSPARSE_RESULTS_ROOT:-"${repo_root}/reproduced-results"}
production_dir="${repo_root}/experiments/production"
dense_dir="${repo_root}/experiments/dense_control"
baseline_dir="${repo_root}/experiments/cufinufft_baseline"
heldout_dir="${repo_root}/experiments/quality"

build_baseline() {
  run bash "${repo_root}/tools/build_cufinufft.sh"
  run bash "${baseline_dir}/build.sh"
}

case "${group}" in
  smoke)
    command -v uv >/dev/null 2>&1 || {
      echo "uv is required for the CPU smoke test" >&2
      exit 2
    }
    run uv run --project "${repo_root}" --frozen \
      python -m unittest discover -s "${repo_root}/tests" -v
    ;;

  evidence)
    run python3 "${repo_root}/tools/verify_evidence.py" \
      --repo-root "${repo_root}"
    ;;

  prepare-trajectories)
    run uv run --project "${repo_root}" --frozen python \
      "${repo_root}/experiments/data_prep/generate_trajectories.py" \
      --output \
      "${repo_root}/experiments/data_prep/trajectories"
    ;;

  get-data)
    run bash "${repo_root}/tools/get_ocmr.sh"
    ;;

  prepare-data)
    run bash "${repo_root}/tools/prepare_data.sh"
    ;;

  build)
    run bash "${production_dir}/build.sh"
    run bash "${dense_dir}/build.sh"
    ;;

  build-baseline)
    build_baseline
    ;;

  baseline-quality)
    development_case=$(required_value TRAJSPARSE_DEVELOPMENT_CASE)
    development_runtime=$(required_value TRAJSPARSE_DEVELOPMENT_RUNTIME_ROOT)
    build_baseline
    run python3 "${baseline_dir}/run_quality_matrix.py" \
      --repo-root "${repo_root}" \
      --binary "${baseline_dir}/build/cufinufft_native_cg" \
      --runtime-root "${development_runtime}" \
      --case "${development_case}" \
      --forward-method 1 --adjoint-method 2 --gpu-sort 0 \
      --raw-scale 512 \
      --output "${results_root}/baseline-quality"
    run python3 "${repo_root}/tools/check_baseline_quality.py" \
      "${results_root}/baseline-quality/summary.json" \
      --output "${results_root}/baseline-quality/admission.json"
    ;;

  quality)
    development_case=$(required_value TRAJSPARSE_DEVELOPMENT_CASE)
    development_runtime=$(required_value TRAJSPARSE_DEVELOPMENT_RUNTIME_ROOT)
    heldout_cases=$(required_value TRAJSPARSE_HELDOUT_CASES_ROOT)
    heldout_runtime=$(required_value TRAJSPARSE_HELDOUT_RUNTIME_ROOT)
    packed_root=$(required_value TRAJSPARSE_PACKED_ROOT)
    run bash "${production_dir}/build.sh"
    run python3 "${production_dir}/run_cg_quality_matrix.py" \
      --repo-root "${repo_root}" \
      --binary "${production_dir}/build/nufft_fp16x2_cg" \
      --runtime-root "${development_runtime}" \
      --packed-root "${packed_root}" \
      --case "${development_case}" \
      --raw-scale 512 \
      --output "${results_root}/quality/development"
    run python3 "${heldout_dir}/run_heldout_quality.py" \
      --repo-root "${repo_root}" \
      --binary "${production_dir}/build/nufft_fp16x2_cg" \
      --runtime-root "${heldout_runtime}" \
      --packed-root "${packed_root}" \
      --cases-root "${heldout_cases}" \
      --raw-scale 512 \
      --output "${results_root}/quality/heldout"
    run python3 "${repo_root}/tools/check_reproduced_quality.py" \
      --development "${results_root}/quality/development/summary.json" \
      --heldout "${results_root}/quality/heldout/summary.json"
    ;;

  main-performance)
    runtime_root=$(required_value TRAJSPARSE_PERFORMANCE_RUNTIME_ROOT)
    packed_root=$(required_value TRAJSPARSE_PACKED_ROOT)
    quality_summary=${TRAJSPARSE_CUFINUFFT_QUALITY_SUMMARY:-"${results_root}/baseline-quality/admission.json"}
    if [[ ${dry_run} -eq 0 && ! -f "${quality_summary}" ]]; then
      echo "missing fresh cuFINUFFT quality summary; run baseline-quality first" >&2
      exit 2
    fi
    run bash "${production_dir}/build.sh"
    if [[ ${dry_run} -eq 0 && ! -x "${baseline_dir}/build/cufinufft_native_cg" ]]; then
      echo "missing admitted baseline binary; run baseline-quality first" >&2
      exit 2
    fi
    run python3 "${baseline_dir}/run_paired_fullcg.py" \
      --repo-root "${repo_root}" \
      --production-binary "${production_dir}/build/nufft_fp16x2_cg" \
      --baseline-binary "${baseline_dir}/build/cufinufft_native_cg" \
      --runtime-root "${runtime_root}" \
      --packed-root "${packed_root}" \
      --quality-summary "${quality_summary}" \
      --forward-method 1 --adjoint-method 2 --gpu-sort 0 \
      --raw-scale 512 \
      --output "${results_root}/main-performance"
    run python3 "${repo_root}/tools/check_reproduced_performance.py" \
      cufinufft "${results_root}/main-performance/summary.json"
    ;;

  design-evidence)
    development_case=$(required_value TRAJSPARSE_DEVELOPMENT_CASE)
    runtime_root=$(required_value TRAJSPARSE_PERFORMANCE_RUNTIME_ROOT)
    packed_root=$(required_value TRAJSPARSE_PACKED_ROOT)
    dense_root=$(required_value TRAJSPARSE_DENSE_ROOT)
    run bash "${production_dir}/build.sh"
    run bash "${dense_dir}/build.sh"
    run python3 "${dense_dir}/run_dense_quality_matrix.py" \
      --repo-root "${repo_root}" \
      --binary "${dense_dir}/build/nufft_fp16x2_cg_dense_control" \
      --runtime-root "${runtime_root}" \
      --packed-root "${packed_root}" \
      --dense-root "${dense_root}" \
      --case "${development_case}" \
      --raw-scale 512 \
      --output "${results_root}/dense-quality"
    run python3 "${repo_root}/tools/check_dense_quality.py" \
      "${results_root}/dense-quality/summary.json"
    run python3 "${dense_dir}/run_paired_dense_fullcg.py" \
      --repo-root "${repo_root}" \
      --sparse-binary "${production_dir}/build/nufft_fp16x2_cg" \
      --dense-binary "${dense_dir}/build/nufft_fp16x2_cg_dense_control" \
      --runtime-root "${runtime_root}" \
      --packed-root "${packed_root}" \
      --dense-root "${dense_root}" \
      --quality-summary "${results_root}/dense-quality/summary.json" \
      --raw-scale 512 \
      --output "${results_root}/design-evidence"
    run python3 "${repo_root}/tools/check_reproduced_performance.py" \
      dense "${results_root}/design-evidence/summary.json"
    ;;

  hardening)
    required_value TRAJSPARSE_PACKED_ROOT >/dev/null
    required_value TRAJSPARSE_DENSE_ROOT >/dev/null
    required_value TRAJSPARSE_HELDOUT_RUNTIME_ROOT >/dev/null
    run bash "${production_dir}/build.sh"
    run bash "${dense_dir}/build.sh"
    build_baseline
    run bash "${repo_root}/experiments/frozen_evidence/run_hardening.sh" \
      "${results_root}/hardening"
    ;;

  long-cg)
    heldout_cases=$(required_value TRAJSPARSE_HELDOUT_CASES_ROOT)
    heldout_runtime=$(required_value TRAJSPARSE_HELDOUT_RUNTIME_ROOT)
    packed_root=$(required_value TRAJSPARSE_PACKED_ROOT)
    run bash "${production_dir}/build.sh"
    run bash "${repo_root}/experiments/frozen_evidence/build_telemetry.sh"
    run python3 "${repo_root}/experiments/frozen_evidence/run_long_cg.py" \
      --repo-root "${repo_root}" \
      --binary "${repo_root}/experiments/frozen_evidence/build_telemetry/nufft_fp16x2_cg_telemetry" \
      --runtime-root "${heldout_runtime}" \
      --packed-root "${packed_root}" \
      --cases-root "${heldout_cases}" \
      --output "${results_root}/long-cg"
    run python3 "${repo_root}/tools/check_long_cg.py" \
      "${results_root}/long-cg/summary.json"
    ;;

  help|-h|--help)
    sed -n '/^## Reproduction groups/,$p' "${repo_root}/README.md"
    ;;

  *)
    echo "unknown group: ${group}" >&2
    echo "usage: ./reproduce.sh <group> [--dry-run]" >&2
    exit 2
    ;;
esac
