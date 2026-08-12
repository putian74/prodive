#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="${MODULE_ROOT}/scripts/threshold_scan"
WORK_ROOT="${PRODIVE_WORK_ROOT}/parameter_sensitivity/threshold_scan"
mkdir -p "$WORK_ROOT"

RAW_SPARSE_DIR="${RAW_SPARSE_DIR:-${PRODIVE_DATA_ROOT}/parameter_sensitivity/threshold_scan/sparse_npz}"
PREPROCESSED_ROOT="${WORK_ROOT}/first_second_threshold_grid"
FAMILY_SET_OUT="${WORK_ROOT}/family_sets"
PATH_SCAN_OUT="${WORK_ROOT}/path_scan"
COVERAGE_ROOT="${COVERAGE_ROOT:-${PRODIVE_DATA_ROOT}/parameter_sensitivity/threshold_scan/coverage_files}"

"$PYTHON" "${SCRIPTS}/02_apply_first_second_threshold_grid.py"   --input-dir "$RAW_SPARSE_DIR"   --output-root "$PREPROCESSED_ROOT"   --mapping-file "${PRODIVE_DATA_ROOT}/shared/pfam_mapping_seed_new.txt"   --background-json "${PRODIVE_DATA_ROOT}/shared/result_kl_all_pfam.json"

"$PYTHON" "${SCRIPTS}/03_extract_family_sets_by_threshold.py"   --preprocessed-root "$PREPROCESSED_ROOT"   --mapping-file "${PRODIVE_DATA_ROOT}/shared/pfam_mapping_seed_new.txt"   --output-dir "$FAMILY_SET_OUT"

"$PYTHON" "${SCRIPTS}/04_scan_path_threshold_grid.py"   --preprocessed-root "$PREPROCESSED_ROOT"   --output-root "$PATH_SCAN_OUT"   --mapping-file "${PRODIVE_DATA_ROOT}/shared/pfam_mapping_seed_new.txt"   --coverage-root "$COVERAGE_ROOT"
