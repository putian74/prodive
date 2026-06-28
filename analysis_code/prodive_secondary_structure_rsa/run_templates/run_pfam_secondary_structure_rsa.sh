#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${PRODIVE_WORK_ROOT}/secondary_structure_rsa/pfam"
mkdir -p "$OUT_ROOT"

REAL_CSV="${OUT_ROOT}/pfam_pairs_ss_rsa_calculated_true_fin_hmmlen.csv"
RANDOM_DB="${OUT_ROOT}/pfam_random_bg_from_csv_direct_pfamseed.sqlite"
PLOT_DIR="${OUT_ROOT}/plots"
TOP_PLOT_DIR="${OUT_ROOT}/top_score_distributions"

"$PYTHON" "$MODULE_ROOT/scripts/pfam/01_compute_pfam_real_ss_rsa.py"   --input-csv "${PRODIVE_DATA_ROOT}/shared/global_high_score_summary_fin.csv"   --pfam-dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed"   --output-csv "$REAL_CSV"   --workers "${WORKERS:-60}"

"$PYTHON" "$MODULE_ROOT/scripts/pfam/02_build_pfam_random_background_sqlite.py"   --struct-csv "$REAL_CSV"   --pfam-dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed"   --output-db "$RANDOM_DB"   --k-per-hit "${K_PER_HIT:-50}"   --workers "${WORKERS:-48}"

"$PYTHON" "$MODULE_ROOT/scripts/pfam/03_plot_pfam_real_vs_random.py"   --db "$RANDOM_DB"   --out-dir "$PLOT_DIR"

"$PYTHON" "$MODULE_ROOT/scripts/pfam/04_plot_pfam_top_score_subsets.py"   --struct-csv "$REAL_CSV"   --score-csv "${PRODIVE_DATA_ROOT}/shared/global_high_score_summary_fin.csv"   --out-dir "$TOP_PLOT_DIR"
