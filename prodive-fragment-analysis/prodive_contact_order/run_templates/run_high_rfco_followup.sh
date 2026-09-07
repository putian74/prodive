#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${PRODIVE_WORK_ROOT}/contact_order"
SUMMARY_CSV="${OUT_ROOT}/plots_vs_random_new_dedup/actual_vs_random_summary.csv"
ACTUAL_CSV="${OUT_ROOT}/contact_order_results.hmmcov_0.8_1.2.csv"
TAIL_DIR="${OUT_ROOT}/high_rfco"
TOPOLOGY_DIR="${OUT_ROOT}/high_rfco_topology_extreme"
SENSITIVITY_DIR="${OUT_ROOT}/remove_high_rfco_check"
SCORE_DIR="${OUT_ROOT}/high_rfco_score_analysis"

"$PYTHON" "$MODULE_ROOT/scripts/03_select_high_rfco_tail.py"   --summary-csv "$SUMMARY_CSV"   --outdir "$TAIL_DIR"

"$PYTHON" "$MODULE_ROOT/scripts/04_draw_high_rfco_contact_topology.py"   --summary-csv "$SUMMARY_CSV"   --actual-csv "$ACTUAL_CSV"   --outdir "$TOPOLOGY_DIR"   --mode extreme

"$PYTHON" "$MODULE_ROOT/scripts/05_remove_high_rfco_sensitivity.py"   --summary-csv "$SUMMARY_CSV"   --outdir "$SENSITIVITY_DIR"

"$PYTHON" "$MODULE_ROOT/scripts/06_merge_high_rfco_with_prodive_score.py"   --selected-subset-csv "${TAIL_DIR}/tail_loose_rfco_ge_0.35.csv"   --original-global-csv "${PRODIVE_DATA_ROOT}/shared/global_high_score_summary_fin.csv"   --outdir "$SCORE_DIR"
