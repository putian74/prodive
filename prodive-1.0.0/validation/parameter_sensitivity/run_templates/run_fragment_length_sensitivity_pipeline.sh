#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

: "${KL_CPP_EXE:?Set KL_CPP_EXE to the compiled KL C++/CUDA executable}"
MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="${MODULE_ROOT}/scripts/fragment_length"
WORK_ROOT="${PRODIVE_WORK_ROOT}/parameter_sensitivity/fragment_length"
mkdir -p "$WORK_ROOT"
FRAGMENTS="${FRAGMENTS:-2 3 4 5 6 7 8 9 10 12 14 16 18 20}"

FRAGMENTS="$FRAGMENTS" INPUT_DIR="${PRODIVE_DATA_ROOT}/shared/PfamA_seed" MAPPING_FILE="${PRODIVE_DATA_ROOT}/shared/pfam_mapping_seed_new.txt" OUT_ROOT="$WORK_ROOT" bash "${SCRIPTS}/02_generate_multi_fragment_packed_db.sh"

"$PYTHON" "${SCRIPTS}/03_generate_fixed_pairlist.py"   --mapping-file "${PRODIVE_DATA_ROOT}/shared/pfam_mapping_seed_new.txt"   --out-dir "${WORK_ROOT}/fixed_pairlists"   --sample-n "${SAMPLE_N:-50000}"   --seed "${RANDOM_SEED:-20260511}"

"$PYTHON" "${SCRIPTS}/04_compute_background_kl_for_fragment_lengths.py"   --pfam-list-file "${WORK_ROOT}/total_mapping_cleaned_names.txt"   --pfam-seed-dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed"   --out-dir "${WORK_ROOT}/background_kl"   --fragment-lengths "${FRAGMENTS// /,}"

FRAGMENTS="$FRAGMENTS" CPP_EXE="$KL_CPP_EXE" PAIR_LIST="${WORK_ROOT}/fixed_pairlists/sampled_pairs.csv" PACKED_DB_ROOT="$WORK_ROOT" OUT_ROOT="${WORK_ROOT}/raw_cpp_outputs" bash "${SCRIPTS}/05_run_pairlist_all_fragment_lengths.sh"

"$PYTHON" "${SCRIPTS}/06_analyze_fragment_length_sensitivity.py"   --input-root "${WORK_ROOT}/raw_cpp_outputs"   --output-dir "${WORK_ROOT}/analysis_results"   --mapping-file "${PRODIVE_DATA_ROOT}/shared/pfam_mapping_seed_new.txt"   --background-json-pattern "${WORK_ROOT}/background_kl/result_kl_all_pfam_fragment_{k}.json"

"$PYTHON" "${SCRIPTS}/07_plot_fragment_length_sensitivity.py"   --result-dir "${WORK_ROOT}/analysis_results"
