#!/usr/bin/env bash
set -Eeuo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the ProDive_release/data directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
: "${RAW_SPARSE_DIR:?Set RAW_SPARSE_DIR to the sparse NPZ directory produced by threshold-scan script 01}"

PYTHON=${PYTHON:-python3}
MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="${MODULE_ROOT}/scripts/threshold_scan"
WORK_ROOT="${PRODIVE_WORK_ROOT}/parameter_sensitivity/threshold_scan"

MAPPING_FILE="${PRODIVE_DATA_ROOT}/shared/pfam_mapping_seed_new.txt"
BACKGROUND_JSON="${PRODIVE_DATA_ROOT}/shared/result_kl_all_pfam.json"
COVERAGE_ROOT="${COVERAGE_ROOT:-${PRODIVE_DATA_ROOT}/shared/PfamA_seed}"

PREPROCESSED_ROOT="${WORK_ROOT}/first_second_threshold_grid"
FAMILY_SET_OUT="${WORK_ROOT}/family_sets"
PATH_SCAN_OUT="${WORK_ROOT}/path_scan"

[[ -d "$RAW_SPARSE_DIR" ]] || { echo "Missing sparse NPZ directory: $RAW_SPARSE_DIR" >&2; exit 1; }
[[ -f "$MAPPING_FILE" ]] || { echo "Missing mapping file: $MAPPING_FILE" >&2; exit 1; }
[[ -f "$BACKGROUND_JSON" ]] || { echo "Missing background JSON: $BACKGROUND_JSON" >&2; exit 1; }
[[ -d "$COVERAGE_ROOT" ]] || { echo "Missing coverage root: $COVERAGE_ROOT" >&2; exit 1; }

mkdir -p "$WORK_ROOT"

"$PYTHON" "${SCRIPTS}/02_apply_first_second_threshold_grid.py" \
  --input-dir "$RAW_SPARSE_DIR" \
  --output-root "$PREPROCESSED_ROOT" \
  --mapping-file "$MAPPING_FILE" \
  --background-json "$BACKGROUND_JSON"

"$PYTHON" "${SCRIPTS}/03_extract_family_sets_by_threshold.py" \
  --preprocessed-root "$PREPROCESSED_ROOT" \
  --mapping-file "$MAPPING_FILE" \
  --output-dir "$FAMILY_SET_OUT"

"$PYTHON" "${SCRIPTS}/04_scan_path_threshold_grid.py" \
  --preprocessed-root "$PREPROCESSED_ROOT" \
  --output-root "$PATH_SCAN_OUT" \
  --mapping-file "$MAPPING_FILE" \
  --coverage-root "$COVERAGE_ROOT"
