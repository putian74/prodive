#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external data directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

if [[ -z "${SS_RSA_CSV:-}" ]]; then
  : "${PRODIVE_PRECOMPUTED_ROOT:?Set PRODIVE_PRECOMPUTED_ROOT or provide SS_RSA_CSV}"
  SS_RSA_CSV="${PRODIVE_PRECOMPUTED_ROOT}/secondary_structure_rsa/pfam_pairs_ss_rsa_calculated_true_fin_hmmlen.csv"
fi

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTERING_ROOT=${CLUSTERING_ROOT:-"${PRODIVE_WORK_ROOT}/clustering/leiden_cluster_fin_0.05_CPM_fin_hmm_len"}

"$PYTHON" "$MODULE_ROOT/scripts/summarize_leiden_struct_stats.py" \
  --input "Top_05=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_05%.csv" \
  --input "Top_10=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_10%.csv" \
  --input "Top_20=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_20%.csv" \
  --input "Top_50=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_50%.csv" \
  --input "Top_100=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_100%.csv" \
  --clustering-root "$CLUSTERING_ROOT" \
  --struct-csv "$SS_RSA_CSV"
