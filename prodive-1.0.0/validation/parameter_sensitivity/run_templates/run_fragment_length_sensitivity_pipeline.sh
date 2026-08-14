#!/usr/bin/env bash
set -Eeuo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the ProDive_release/data directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
: "${KL_CPP_EXE:?Set KL_CPP_EXE to the compiled KL C++/CUDA executable}"

PYTHON=${PYTHON:-python3}
MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="${MODULE_ROOT}/scripts/fragment_length"
WORK_ROOT="${PRODIVE_WORK_ROOT}/parameter_sensitivity/fragment_length"

PFAM_ROOT="${PRODIVE_DATA_ROOT}/shared/PfamA_seed"
MAPPING_FILE="${PRODIVE_DATA_ROOT}/shared/pfam_mapping_seed_new.txt"
RELEASE_PAIR_LIST="${PRODIVE_DATA_ROOT}/parameter_sensitivity/fragment_length/sampled_pairlist.csv"

FRAGMENTS="${FRAGMENTS:-2 3 4 5 6 7 8 9 10 12 14 16 18 20}"
PAIR_LIST="${PAIR_LIST:-$RELEASE_PAIR_LIST}"
REGENERATE_PAIRLIST="${REGENERATE_PAIRLIST:-0}"

[[ -d "$PFAM_ROOT" ]] || { echo "Missing Pfam root: $PFAM_ROOT" >&2; exit 1; }
[[ -f "$MAPPING_FILE" ]] || { echo "Missing mapping file: $MAPPING_FILE" >&2; exit 1; }
[[ -x "$KL_CPP_EXE" ]] || { echo "KL executable is not executable: $KL_CPP_EXE" >&2; exit 1; }

mkdir -p "$WORK_ROOT"

if [[ "$REGENERATE_PAIRLIST" == "1" ]]; then
    GENERATED_PAIR_DIR="${WORK_ROOT}/fixed_pairlists"
    "$PYTHON" "${SCRIPTS}/03_generate_fixed_pairlist.py" \
      --mapping-file "$MAPPING_FILE" \
      --out-dir "$GENERATED_PAIR_DIR" \
      --sample-n "${SAMPLE_N:-50000}" \
      --seed "${RANDOM_SEED:-20260511}" \
      --overwrite
    PAIR_LIST="${GENERATED_PAIR_DIR}/sampled_pairs.csv"
elif [[ "$REGENERATE_PAIRLIST" != "0" ]]; then
    echo "REGENERATE_PAIRLIST must be 0 or 1." >&2
    exit 1
fi

[[ -f "$PAIR_LIST" ]] || { echo "Missing pair list: $PAIR_LIST" >&2; exit 1; }

FRAGMENTS="$FRAGMENTS" \
INPUT_DIR="$PFAM_ROOT" \
MAPPING_FILE="$MAPPING_FILE" \
OUT_ROOT="$WORK_ROOT" \
bash "${SCRIPTS}/02_generate_multi_fragment_packed_db.sh"

"$PYTHON" "${SCRIPTS}/04_compute_background_kl_for_fragment_lengths.py" \
  --mapping-file "$MAPPING_FILE" \
  --pfam-seed-dir "$PFAM_ROOT" \
  --out-dir "${WORK_ROOT}/background_kl" \
  --fragment-lengths "${FRAGMENTS// /,}"

FRAGMENTS="$FRAGMENTS" \
CPP_EXE="$KL_CPP_EXE" \
PAIR_LIST="$PAIR_LIST" \
PACKED_DB_ROOT="$WORK_ROOT" \
OUT_ROOT="${WORK_ROOT}/raw_cpp_outputs" \
bash "${SCRIPTS}/05_run_pairlist_all_fragment_lengths.sh"

"$PYTHON" "${SCRIPTS}/06_analyze_fragment_length_sensitivity.py" \
  --input-root "${WORK_ROOT}/raw_cpp_outputs" \
  --output-dir "${WORK_ROOT}/analysis_results" \
  --mapping-file "$MAPPING_FILE" \
  --background-json-pattern "${WORK_ROOT}/background_kl/result_kl_all_pfam_fragment_{k}.json"

"$PYTHON" "${SCRIPTS}/07_plot_fragment_length_sensitivity.py" \
  --result-dir "${WORK_ROOT}/analysis_results"
