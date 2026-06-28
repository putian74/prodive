#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_ROOT="${PRODIVE_WORK_ROOT}/rmsd/hhsearch_comparison"
mkdir -p "$OUT_ROOT"

"$PYTHON" "$MODULE_ROOT/scripts/01_parse_hhr_to_csv.py"   --input-dir "${PRODIVE_DATA_ROOT}/rmsd/hhsearch_hhr"   --output-csv "${OUT_ROOT}/hhsuite20.csv"   --prob-threshold 20

"$PYTHON" "$MODULE_ROOT/scripts/01_parse_hhr_to_csv.py"   --input-dir "${PRODIVE_DATA_ROOT}/rmsd/hhsearch_hhr"   --output-csv "${OUT_ROOT}/hhsuite70.csv"   --prob-threshold 70

"$PYTHON" "$MODULE_ROOT/scripts/02_make_percentile_subsets.py"   --input-csv "${PRODIVE_DATA_ROOT}/shared/global_high_score_summary_fin.csv"   --output-dir "${OUT_ROOT}/percentile_filtered_reports"

"$PYTHON" "$MODULE_ROOT/scripts/03_check_hhsearch_overlap.py"   --base-dir "${OUT_ROOT}/percentile_filtered_reports"   --hhsuite-csv "${OUT_ROOT}/hhsuite20.csv"   --prob-threshold 20

"$PYTHON" "$MODULE_ROOT/scripts/03_check_hhsearch_overlap.py"   --base-dir "${OUT_ROOT}/percentile_filtered_reports"   --hhsuite-csv "${OUT_ROOT}/hhsuite70.csv"   --prob-threshold 70

"$PYTHON" "$MODULE_ROOT/scripts/04_build_hhsearch_class_tasks.py"   --prodive-overlap-csv "${OUT_ROOT}/percentile_filtered_reports/subset_data_Top_100%_overlap_check_dual_20.csv"   --hhsuite-csv "${OUT_ROOT}/hhsuite20.csv"   --output-dir "${OUT_ROOT}/class_tasks_prob20"

"$PYTHON" "$MODULE_ROOT/scripts/04_build_hhsearch_class_tasks.py"   --prodive-overlap-csv "${OUT_ROOT}/percentile_filtered_reports/subset_data_Top_100%_overlap_check_dual_70.csv"   --hhsuite-csv "${OUT_ROOT}/hhsuite70.csv"   --output-dir "${OUT_ROOT}/class_tasks_prob70"
