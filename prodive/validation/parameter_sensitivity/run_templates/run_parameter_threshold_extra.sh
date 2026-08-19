#!/usr/bin/env bash
set -Eeuo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the ProDive_methods_data/data directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
: "${THIRD_SCORE_INPUT_ROOT:?Set THIRD_SCORE_INPUT_ROOT to the distributed full-grid path-scan root}"
: "${FAMILY_SET_INPUT_ROOT:?Set FAMILY_SET_INPUT_ROOT to the distributed family-set result root}"
: "${SCORE_CURVE_INPUT_ROOT:?Set SCORE_CURVE_INPUT_ROOT to the per-combination score-cutoff curve root}"

PYTHON_BIN=${PYTHON:-python3}
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARAM_OUT="${PRODIVE_WORK_ROOT}/parameter_sensitivity"
MAPPING_FILE="${PRODIVE_DATA_ROOT}/shared/pfam_mapping_seed_new.txt"

[[ -d "$THIRD_SCORE_INPUT_ROOT" ]] || { echo "Missing third-score input root: $THIRD_SCORE_INPUT_ROOT" >&2; exit 1; }
[[ -d "$FAMILY_SET_INPUT_ROOT" ]] || { echo "Missing family-set input root: $FAMILY_SET_INPUT_ROOT" >&2; exit 1; }
[[ -d "$SCORE_CURVE_INPUT_ROOT" ]] || { echo "Missing score-curve input root: $SCORE_CURVE_INPUT_ROOT" >&2; exit 1; }
[[ -f "$MAPPING_FILE" ]] || { echo "Missing mapping file: $MAPPING_FILE" >&2; exit 1; }

mkdir -p "$PARAM_OUT"

if [[ -z "${PFAM_IDS_FILE:-}" ]]; then
    PFAM_IDS_FILE="${PARAM_OUT}/pfam_ids_from_mapping.txt"
    awk -F ':' '
      NF >= 2 {
        name = $1
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
        if (name != "") print name
      }
    ' "$MAPPING_FILE" > "$PFAM_IDS_FILE"
fi

[[ -s "$PFAM_IDS_FILE" ]] || { echo "Missing or empty Pfam ID file: $PFAM_IDS_FILE" >&2; exit 1; }

"$PYTHON_BIN" "$MODULE_DIR/scripts/threshold_scan/05_collect_third_score_thresholds.py" \
  --root-dir "$THIRD_SCORE_INPUT_ROOT" \
  --pfam-ids-file "$PFAM_IDS_FILE" \
  --output-root "$PARAM_OUT/global_third_score_threshold_analysis"

"$PYTHON_BIN" "$MODULE_DIR/scripts/threshold_scan/06_plot_global_threshold_grid.py" \
  --root-dir "$FAMILY_SET_INPUT_ROOT" \
  --output-dir "$PARAM_OUT/global_family_union_threshold_plots_no_loss"

"$PYTHON_BIN" "$MODULE_DIR/scripts/threshold_scan/07_logistic_second_derivative_extrema.py" \
  --input-curve-root "$SCORE_CURVE_INPUT_ROOT" \
  --output-root "$PARAM_OUT/score_cutoff_xaxis_logistic_second_derivative_extrema_raw_counts"
