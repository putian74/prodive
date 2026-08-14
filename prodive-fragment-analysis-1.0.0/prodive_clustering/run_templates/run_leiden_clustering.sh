#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external data directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${PRODIVE_WORK_ROOT}/clustering/leiden_cluster_fin_0.05_CPM_fin_hmm_len"
mkdir -p "$OUT_ROOT"

"$PYTHON" "$MODULE_ROOT/scripts/build_leiden_clusters.py" \
  --input "Top_05=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_05%.csv" \
  --input "Top_10=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_10%.csv" \
  --input "Top_20=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_20%.csv" \
  --input "Top_50=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_50%.csv" \
  --input "Top_100=${PRODIVE_DATA_ROOT}/shared/score_percentile_subsets/subset_data_Top_100%.csv" \
  --output-root "$OUT_ROOT"
