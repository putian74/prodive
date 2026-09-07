#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external data directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
: "${PRODIVE_PFAM_RUNTIME_ROOT:?Set PRODIVE_PFAM_RUNTIME_ROOT to the merged Pfam profile/structure directory}"
PYTHON=${PYTHON:-python3}
MODULE_ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULT_ROOT=${RESULT_ROOT:-"${PRODIVE_WORK_ROOT}/rmsd/structural_validation/rmsd_result_tables"}
DENOVO_INPUT_CSV=${DENOVO_INPUT_CSV:-"${PRODIVE_DATA_ROOT}/shared/denovo_global_high_score_summary_fin.csv"}
DENOVO_PDB_DIR=${DENOVO_PDB_DIR:-"${PRODIVE_DATA_ROOT}/shared/structures/denovo_structures/downloaded_denovo_pdbs"}
DENOVO_FASTA=${DENOVO_FASTA-"${PRODIVE_DATA_ROOT}/shared/structures/denovo_structures/all_1927_sequences.fasta"}
DENOVO_RMSD="${RESULT_ROOT}/denovo_vs_pfam_rmsd_results_with_coverage.csv"
RANDOM_CSV=${RANDOM_CSV:-"${RESULT_ROOT}/random_pfam_vs_denovo_10k_coverage80.csv"}
mkdir -p "$RESULT_ROOT"

for required_dir in "$PRODIVE_PFAM_RUNTIME_ROOT" "$DENOVO_PDB_DIR"; do
  [[ -d "$required_dir" ]] || { echo "Input directory not found: $required_dir" >&2; exit 1; }
done
[[ -f "$DENOVO_INPUT_CSV" ]] || { echo "Input CSV not found: $DENOVO_INPUT_CSV" >&2; exit 1; }
FASTA_ARGS=()
if [[ -n "$DENOVO_FASTA" ]]; then
  [[ -f "$DENOVO_FASTA" ]] || { echo "FASTA not found: $DENOVO_FASTA. Set DENOVO_FASTA='' to use automatic chain selection." >&2; exit 1; }
  FASTA_ARGS+=(--fasta-mapping-file "$DENOVO_FASTA")
fi

"$PYTHON" "$MODULE_ROOT/scripts/01_rmsd_denovo_pfam.py" \
  --input-csv "$DENOVO_INPUT_CSV" \
  --denovo-pdb-dir "$DENOVO_PDB_DIR" \
  --pfam-dir "$PRODIVE_PFAM_RUNTIME_ROOT" \
  "${FASTA_ARGS[@]}" \
  --segment-mode "${SEGMENT_MODE:-first}" \
  --output-csv "$DENOVO_RMSD"

if [[ "${RUN_RANDOM_CONTROLS:-0}" == "1" ]]; then
  SEED_ARGS=()
  if [[ -n "${RANDOM_SEED:-}" ]]; then SEED_ARGS+=(--seed "$RANDOM_SEED"); fi
  "$PYTHON" "$MODULE_ROOT/scripts/02_sample_random_denovo_pfam_pairs.py" \
    --pfam-dir "$PRODIVE_PFAM_RUNTIME_ROOT" \
    --denovo-pdb-dir "$DENOVO_PDB_DIR" \
    --output-csv "$RANDOM_CSV" \
    --aligned-lengths "${ALIGNED_LENGTHS:-8,9,10,11,12,13}" \
    --per-length-quota "${PER_LENGTH_QUOTA:-10000}" \
    --coverage-threshold "${COVERAGE_THRESHOLD:-0.8}" \
    --input-window-min "${INPUT_WINDOW_MIN:-8}" \
    --input-window-max "${INPUT_WINDOW_MAX:-20}" \
    --require-substring "${PFAM_PDB_SUBSTRING-model}" \
    --workers "${WORKERS:-20}" \
    "${SEED_ARGS[@]}"
fi

if [[ "${RUN_COMPARISON:-0}" == "1" ]]; then
  [[ -f "$RANDOM_CSV" ]] || { echo "Random-control CSV not found: $RANDOM_CSV. Set RUN_RANDOM_CONTROLS=1 or provide RANDOM_CSV." >&2; exit 1; }
  "$PYTHON" "$MODULE_ROOT/scripts/04_compare_real_random_core.py" \
    --real-csv "$DENOVO_RMSD" --random-csv "$RANDOM_CSV" \
    --core-lengths "${ALIGNED_LENGTHS:-8,9,10,11,12,13}" \
    --coverage-threshold "${COVERAGE_THRESHOLD:-0.8}" \
    --output-stats-csv "${RESULT_ROOT}/denovo_pfam_real_random_core_summary.csv" \
    --output-plot "${RESULT_ROOT}/denovo_pfam_real_vs_random.png" \
    --plot-title "Core region comparison: de novo-Pfam real vs. random"
  "$PYTHON" "$MODULE_ROOT/scripts/05_welch_ttest_from_stats.py" \
    --stats-csv "${RESULT_ROOT}/denovo_pfam_real_random_core_summary.csv" \
    --output-csv "${RESULT_ROOT}/denovo_pfam_welch_ttest.csv"
fi
