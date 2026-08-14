#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external data directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PERCENTILE=${PERCENTILE:-100}
case "$PERCENTILE" in
  5) INPUT_TAG=05 ;;
  10|20|50|100) INPUT_TAG=$PERCENTILE ;;
  *)
    echo "PERCENTILE must be one of: 5, 10, 20, 50, 100" >&2
    exit 2
    ;;
esac

INPUT_CSV=${INPUT_CSV:-"${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_${INPUT_TAG}%.csv"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PRODIVE_WORK_ROOT}/clustering/leiden_gamma_sensitivity_top${PERCENTILE}"}
GAMMAS=${GAMMAS:-"0.01 0.02 0.03 0.04 0.05 0.06 0.08 0.10 0.15 0.20"}
read -r -a GAMMA_VALUES <<< "$GAMMAS"

"$PYTHON" "$MODULE_ROOT/scripts/leiden_gamma_sensitivity.py" \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --gammas "${GAMMA_VALUES[@]}"
