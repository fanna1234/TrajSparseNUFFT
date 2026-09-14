#!/usr/bin/env bash
# Keep one public configuration while accepting existing reproduction layouts.

resolve_runtime_environment() {
  local suffix current legacy current_value legacy_value
  for suffix in GPU_PYTHON DATA_ROOT PREPARED_ROOT RESULTS_ROOT \
    DEVELOPMENT_CASE DEVELOPMENT_RUNTIME_ROOT HELDOUT_CASES_ROOT \
    HELDOUT_RUNTIME_ROOT PERFORMANCE_RUNTIME_ROOT PACKED_ROOT DENSE_ROOT \
    CUFINUFFT_QUALITY_SUMMARY ALLOW_VERSION_DRIFT; do
    current="TRAJTC_${suffix}"
    legacy="TRAJSPARSE_${suffix}"
    current_value=${!current:-}
    legacy_value=${!legacy:-}
    if [[ -n "${current_value}" && -n "${legacy_value}" \
      && "${current_value}" != "${legacy_value}" ]]; then
      echo "conflicting settings: ${current} and ${legacy}" >&2
      return 2
    fi
    if [[ -n "${current_value}${legacy_value}" ]]; then
      printf -v "${current}" '%s' "${current_value:-${legacy_value}}"
      printf -v "${legacy}" '%s' "${current_value:-${legacy_value}}"
      export "${current}" "${legacy}"
    fi
  done
}

resolve_runtime_environment
