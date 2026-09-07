#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
: "${PRODIVE_PFAM_RUNTIME_ROOT:?Set PRODIVE_PFAM_RUNTIME_ROOT to the merged Pfam profile/structure directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${PRODIVE_WORK_ROOT}/contact_order"
mkdir -p "$OUT_ROOT"

ACTUAL_CSV="${OUT_ROOT}/contact_order_results.hmmcov_0.8_1.2.csv"
RANDOM_CSV="${OUT_ROOT}/contact_order_random_results.hmmcov_0.8_1.2.csv"
SUMMARY_TXT="${OUT_ROOT}/contact_order_results.hmmcov_0.8_1.2.summary.txt"
COMPARE_DIR="${OUT_ROOT}/plots_vs_random_new_dedup"

"$PYTHON" "$MODULE_ROOT/scripts/01_compute_contact_order_and_random.py"   --pfam-dir "$PRODIVE_PFAM_RUNTIME_ROOT"   --global-score-csv "${PRODIVE_DATA_ROOT}/shared/global_high_score_summary_fin.csv"   --actual-out-csv "$ACTUAL_CSV"   --random-out-csv "$RANDOM_CSV"   --cache-dir "${CONTACT_ORDER_CACHE:-${OUT_ROOT}/cache}"   --jobs "${JOBS:-24}"   --preprocess-jobs "${PREPROCESS_JOBS:-24}"   --preprocess-mode process   --chunk-size "${CHUNK_SIZE:-500}"   --random-samples "${RANDOM_SAMPLES:-1000}"   --cutoff 8   --exclude-near 2   --long-range-threshold 12   --ok-only   --min-coverage 0.8   --max-coverage 1.2   --summary-txt "$SUMMARY_TXT"

"$PYTHON" "$MODULE_ROOT/scripts/02_compare_actual_vs_random_dedup.py"   --actual-csv "$ACTUAL_CSV"   --random-csv "$RANDOM_CSV"   --outdir "$COMPARE_DIR"   --dedup-identical-interval
