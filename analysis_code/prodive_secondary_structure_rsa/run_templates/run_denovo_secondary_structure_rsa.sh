#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${PRODIVE_WORK_ROOT}/secondary_structure_rsa/denovo"
mkdir -p "$OUT_ROOT"

REAL_CSV="${OUT_ROOT}/denovo_features_calculated_true.csv"
RANDOM_CSV="${OUT_ROOT}/denovo_features_random_controls.csv"
PLOT_PNG="${OUT_ROOT}/denovo_all_null_vs_all_real.png"

"$PYTHON" "$MODULE_ROOT/scripts/denovo/01_compute_denovo_real_ss_rsa.py"   --input-csv "${PRODIVE_DATA_ROOT}/shared/denovo_global_high_score_summary_fin.csv"   --pdb-dir "${PRODIVE_DATA_ROOT}/shared/structures/denovo_structures/downloaded_denovo_pdbs"   --fasta "${PRODIVE_DATA_ROOT}/shared/structures/denovo_structures/all_1927_sequences.fasta"   --output-csv "$REAL_CSV"   --workers "${WORKERS:-20}"

"$PYTHON" "$MODULE_ROOT/scripts/denovo/02_build_denovo_random_controls.py"   --input-positive-csv "${PRODIVE_DATA_ROOT}/shared/denovo_global_high_score_summary_fin.csv"   --pdb-dir "${PRODIVE_DATA_ROOT}/shared/structures/denovo_structures/downloaded_denovo_pdbs"   --fasta "${PRODIVE_DATA_ROOT}/shared/structures/denovo_structures/all_1927_sequences.fasta"   --output-control-csv "$RANDOM_CSV"   --neg-per-pos "${NEG_PER_POS:-30}"   --workers "${WORKERS:-40}"   --random-seed "${RANDOM_SEED:-20260129}"

"$PYTHON" "$MODULE_ROOT/scripts/denovo/03_plot_denovo_real_vs_random.py"   --real-csv "$REAL_CSV"   --null-csv "$RANDOM_CSV"   --out "$PLOT_PNG"
