#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${PRODIVE_WORK_ROOT}/interface_analysis"
mkdir -p "$OUT_ROOT"

"$PYTHON" "$MODULE_ROOT/scripts/01_denovo_interface_random_control.py"   --hit-csv "${PRODIVE_DATA_ROOT}/shared/denovo_global_high_score_summary_fin.csv"   --fasta "${PRODIVE_DATA_ROOT}/shared/structures/denovo_structures/all_1927_sequences.fasta"   --struct-dir "${PRODIVE_DATA_ROOT}/shared/structures/denovo_structures/downloaded_denovo_pdbs"   --out-csv "${OUT_ROOT}/global_high_score_summary_fin.with_interface_randctrl.mp.csv"   --max-workers "${WORKERS:-16}"   --dist-cutoff "${DIST_CUTOFF:-5.0}"   --rand-samples "${RAND_SAMPLES:-3000}"
