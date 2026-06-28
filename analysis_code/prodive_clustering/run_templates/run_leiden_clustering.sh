#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SS_RSA_CSV="${SS_RSA_CSV:-${PRODIVE_WORK_ROOT}/secondary_structure_rsa/pfam/pfam_pairs_ss_rsa_calculated_true_fin_hmmlen.csv}"
OUT_ROOT="${PRODIVE_WORK_ROOT}/clustering/leiden_cluster_fin_0.05_CPM_fin_hmm_len"
mkdir -p "$OUT_ROOT"

"$PYTHON" "$MODULE_ROOT/scripts/build_leiden_clusters_with_struct_stats.py"   --input "Top_05=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_05%.csv"   --input "Top_10=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_10%.csv"   --input "Top_20=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_20%.csv"   --input "Top_50=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_50%.csv"   --input "Top_100=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_100%.csv"   --struct-csv "$SS_RSA_CSV"   --output-root "$OUT_ROOT"
