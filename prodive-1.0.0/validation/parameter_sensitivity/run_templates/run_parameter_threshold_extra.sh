#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external data archive data directory.}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory.}"
PYTHON_BIN="${PYTHON:-python3}"
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PARAM_DATA="$PRODIVE_DATA_ROOT/parameter_sensitivity"
PARAM_OUT="$PRODIVE_WORK_ROOT/parameter_sensitivity"
mkdir -p "$PARAM_OUT"

# Step A: collect the third-layer score threshold at each F/S combination.
$PYTHON_BIN "$MODULE_DIR/scripts/threshold_scan/05_collect_third_score_thresholds.py" \
  --root-dir "$PARAM_DATA/threshold_scan/path_scan_outputs" \
  --pfam-ids-file "$PRODIVE_DATA_ROOT/shared/pfam_ids.txt" \
  --output-root "$PARAM_OUT/global_third_score_threshold_analysis"

# Step B: merge distributed first/second-threshold family sets and draw threshold-grid plots.
$PYTHON_BIN "$MODULE_DIR/scripts/threshold_scan/06_plot_global_threshold_grid.py" \
  --root-dir "$PARAM_DATA/threshold_scan/threshold_grid_outputs" \
  --output-dir "$PARAM_OUT/global_family_union_threshold_plots_no_loss"

# Step C: fit score-cutoff curves and extract second-derivative extrema.
$PYTHON_BIN "$MODULE_DIR/scripts/threshold_scan/07_logistic_second_derivative_extrema.py" \
  --input-curve-root "$PARAM_DATA/threshold_scan/score_cutoff_curves" \
  --output-root "$PARAM_OUT/score_cutoff_xaxis_logistic_second_derivative_extrema_raw_counts"
