#!/usr/bin/env bash
set -Eeuo pipefail

: "${INPUT_DIR:?Set INPUT_DIR to the Pfam seed directory}"
: "${MAPPING_FILE:?Set MAPPING_FILE to the Pfam mapping file}"
: "${OUT_ROOT:?Set OUT_ROOT to the packed-database output root}"

PY_SCRIPT="${PY_SCRIPT:-$(dirname "$0")/01_pack_fixed_length_hhm_fragments.py}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/packed_db_generation_logs}"
JOBS="${JOBS:-40}"
OVERWRITE="${OVERWRITE:-0}"
FRAGMENTS="${FRAGMENTS:-2 3 4 5 6 7 8 9 10}"

mkdir -p "${LOG_DIR}"

for K in ${FRAGMENTS}; do
    OUT_DIR="${OUT_ROOT}/kl_fragment_${K}_packed"
    LOG_FILE="${LOG_DIR}/kl_fragment_${K}_packed.log"
    echo "[START] fragment=${K}"
    if [[ -d "${OUT_DIR}" && -f "${OUT_DIR}/metadata.json" && "${OVERWRITE}" -eq 0 ]]; then
        echo "[SKIP] existing output: ${OUT_DIR}"
        continue
    fi
    CMD=("${PYTHON:-python3}" "${PY_SCRIPT}" --input_dir "${INPUT_DIR}" --output_dir "${OUT_DIR}" --mapping "${MAPPING_FILE}" --fragment "${K}" --jobs "${JOBS}")
    if [[ "${OVERWRITE}" -eq 1 ]]; then
        CMD+=(--overwrite)
    fi
    "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
done
