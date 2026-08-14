#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external data directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${PRODIVE_WORK_ROOT}/disorder_overlap"
mkdir -p "$OUT_ROOT"

SEED_MATCH_OUT="${OUT_ROOT}/disprot_seed_match_output"
SEGMENT_OVERLAP_OUT="${OUT_ROOT}/global_hmm_to_disprot_region_overlap"
OBS_EXP_OUT="${OUT_ROOT}/observed_expected_results"
FIGURE_OUT="${OUT_ROOT}/real_vs_random_figures"

"$PYTHON" "$MODULE_ROOT/scripts/01_compare_disprot_regions_to_pfam_seed.py" \
  --disprot-tsv "${PRODIVE_DATA_ROOT}/disorder_overlap/DisProt_current_IDPO.tsv" \
  --pfam-seed-dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed" \
  --out-dir "$SEED_MATCH_OUT"

"$PYTHON" "$MODULE_ROOT/scripts/02_map_hmm_segments_to_disprot_regions.py" \
  --global-score-file "${PRODIVE_DATA_ROOT}/shared/global_high_score_summary_fin.csv" \
  --covered-region-file "${SEED_MATCH_OUT}/disprot_regions_covered_in_seed.csv" \
  --detail-file "${SEED_MATCH_OUT}/disprot_seed_match_detail.csv" \
  --pfam-dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed" \
  --output-dir "$SEGMENT_OVERLAP_OUT"

"$PYTHON" "$MODULE_ROOT/scripts/03_pfam_disorder_observed_expected.py" \
  --disprot-tsv "${PRODIVE_DATA_ROOT}/disorder_overlap/DisProt_current_IDPO.tsv" \
  --detail-csv "${SEED_MATCH_OUT}/disprot_seed_match_detail.csv" \
  --mapped-csv "${SEGMENT_OVERLAP_OUT}/all_mapped_segments_with_overlap_status.csv" \
  --out-dir "$OBS_EXP_OUT" \
  --n-permutations "${N_PERMUTATIONS:-1000}" \
  --seed "${RANDOM_SEED:-20260331}"

"$PYTHON" "$MODULE_ROOT/scripts/04_plot_real_vs_random.py" \
  --in-dir "$OBS_EXP_OUT" \
  --out-dir "$FIGURE_OUT"
