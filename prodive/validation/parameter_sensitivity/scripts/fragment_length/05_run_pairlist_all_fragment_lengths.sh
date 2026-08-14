#!/usr/bin/env bash
set -Eeuo pipefail

: "${CPP_EXE:?Set CPP_EXE to the compiled KL C++/CUDA executable}"
: "${PAIR_LIST:?Set PAIR_LIST to sampled_pairs.csv}"
: "${PACKED_DB_ROOT:?Set PACKED_DB_ROOT to the packed database root}"
: "${OUT_ROOT:?Set OUT_ROOT to the KL output root}"

LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs}"
NUM_STREAMS="${NUM_STREAMS:-8}"
MAX_GPUS="${MAX_GPUS:-4}"
FRAGMENTS="${FRAGMENTS:-2 3 4 5 6 7 8 9 10}"
RESUME="${RESUME:-1}"

mkdir -p "${LOG_DIR}"

for K in ${FRAGMENTS}; do
    PACKED_DB="${PACKED_DB_ROOT}/kl_fragment_${K}_packed"
    OUT_DIR="${OUT_ROOT}/fragment_${K}"
    LOG_FILE="${LOG_DIR}/fragment_${K}.log"
    mkdir -p "${OUT_DIR}"
    if [[ "${RESUME}" -eq 1 && -f "${OUT_DIR}/DONE" ]]; then
        echo "[SKIP] fragment=${K}: DONE marker exists."
        continue
    fi
    echo "[START] fragment=${K}"
    "${CPP_EXE}" \
        --packed_db "${PACKED_DB}" \
        --pair_list "${PAIR_LIST}" \
        --outdir "${OUT_DIR}" \
        --num_streams "${NUM_STREAMS}" \
        --max_gpus "${MAX_GPUS}" \
        2>&1 | tee "${LOG_FILE}"
    touch "${OUT_DIR}/DONE"
done
