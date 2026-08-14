#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the released data directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
: "${PRODIVE_PFAM_RUNTIME_ROOT:?Set PRODIVE_PFAM_RUNTIME_ROOT to the merged Pfam profile/structure runtime directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")" && pwd)"
HH_ROOT=${HH_ROOT:-"${PRODIVE_WORK_ROOT}/rmsd/hhsearch_comparison"}
RESULT_ROOT=${RESULT_ROOT:-"${PRODIVE_WORK_ROOT}/rmsd/structural_validation/rmsd_result_tables"}
OVERLAP20="${HH_ROOT}/percentile_filtered_reports/subset_data_Top_100%_overlap_check_dual_20.csv"
TASK20="${HH_ROOT}/class_tasks_prob20/hhsuite_only_tasks.csv"
mkdir -p "$RESULT_ROOT"

for required_file in "$OVERLAP20" "$TASK20" "${HH_ROOT}/hhsuite20.csv"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required HHsearch preprocessing output not found: $required_file" >&2
    echo "Run ../hhsearch_comparison/run_hhsearch_preprocessing.sh first." >&2
    exit 1
  fi
done

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

SHARED_PRODIVE_RMSD="${RESULT_ROOT}/false_rows_rmsd_HYBRID_PDB_AF.csv"
HHSEARCH_ONLY_RMSD="${RESULT_ROOT}/hhsuite_only_rmsd_HYBRID_PDB_AF.csv"
SHARED_HH_RMSD="${RESULT_ROOT}/false_rows_HH_segments_rmsd_HYBRID_PDB_AF.csv"
PRODIVE_ONLY_STANDARD="${RESULT_ROOT}/Top_100%_final_novel_rmsd_from_selected_results.standard.csv"
DENOVO_RMSD="${RESULT_ROOT}/denovo_vs_pfam_rmsd_results_with_coverage.csv"

"$PYTHON" "$MODULE_ROOT/scripts/01_rmsd_shared_prodive_segments.py" \
  --ref-csv "$OVERLAP20" \
  --pfam-dir "$PRODIVE_PFAM_RUNTIME_ROOT" \
  --output-csv "$SHARED_PRODIVE_RMSD" \
  "${OVERWRITE_ARGS[@]}"

"$PYTHON" "$MODULE_ROOT/scripts/02_rmsd_hhsearch_only.py" \
  --task-csv "$TASK20" \
  --pfam-dir "$PRODIVE_PFAM_RUNTIME_ROOT" \
  --output-csv "$HHSEARCH_ONLY_RMSD" \
  "${OVERWRITE_ARGS[@]}"

"$PYTHON" "$MODULE_ROOT/scripts/03_rmsd_shared_hhsearch_segments.py" \
  --ref-csv "$OVERLAP20" \
  --pfam-dir "$PRODIVE_PFAM_RUNTIME_ROOT" \
  --output-csv "$SHARED_HH_RMSD" \
  "${OVERWRITE_ARGS[@]}"

"$PYTHON" "$MODULE_ROOT/scripts/04_rmsd_prodive_only_pipeline.py" \
  --input-csv "$OVERLAP20" \
  --pfam-dir "$PRODIVE_PFAM_RUNTIME_ROOT" \
  --output-dir "$RESULT_ROOT" \
  --selection-csv "${RESULT_ROOT}/prodive_only_representative_selection.csv" \
  --rmsd-csv "${RESULT_ROOT}/prodive_only_rmsd_from_selected.csv" \
  --standard-csv "$PRODIVE_ONLY_STANDARD" \
  --num-workers "${WORKERS:-40}"

"$PYTHON" "$MODULE_ROOT/scripts/05_rmsd_denovo_pfam.py" \
  --input-csv "${PRODIVE_DATA_ROOT}/shared/denovo_global_high_score_summary_fin.csv" \
  --denovo-pdb-dir "${PRODIVE_DATA_ROOT}/shared/structures/denovo_structures/downloaded_denovo_pdbs" \
  --fasta-mapping-file "${PRODIVE_DATA_ROOT}/shared/structures/denovo_structures/all_1927_sequences.fasta" \
  --pfam-dir "$PRODIVE_PFAM_RUNTIME_ROOT" \
  --output-csv "$DENOVO_RMSD"

if [[ "${FILTER_PROB70:-1}" == "1" ]]; then
  "$PYTHON" "$MODULE_ROOT/scripts/06_filter_rmsd_by_hhsearch_probability.py" \
    --hhsuite-csv "${HH_ROOT}/hhsuite20.csv" \
    --overlap-csv "$OVERLAP20" \
    --hhsuite-only-rmsd "$HHSEARCH_ONLY_RMSD" \
    --shared-prodive-rmsd "$SHARED_PRODIVE_RMSD" \
    --shared-hh-rmsd "$SHARED_HH_RMSD" \
    --prob-threshold 70 \
    --output-dir "$RESULT_ROOT"
fi
