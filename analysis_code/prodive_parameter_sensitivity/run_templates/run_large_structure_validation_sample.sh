#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external data archive data directory.}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory.}"
PYTHON_BIN="${PYTHON:-python3}"
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PARAM_DATA="$PRODIVE_DATA_ROOT/parameter_sensitivity"
OUT_DIR="$PRODIVE_WORK_ROOT/parameter_sensitivity/bigger_data_sec_rsa"
SAMPLE_DIR="$OUT_DIR/F5p5_S8p0_random_samples"
mkdir -p "$SAMPLE_DIR"

INPUT_CSV="$PARAM_DATA/threshold_scan/path_scan_outputs/F5p5_S8p0_filtered_paths_after_final_score_threshold.csv"
PREFIX="F5p5_S8p0_filtered_paths_after_final_score_threshold"

$PYTHON_BIN "$MODULE_DIR/scripts/large_structure_validation/01_sample_filtered_paths.py" \
  --input-csv "$INPUT_CSV" \
  --output-dir "$SAMPLE_DIR" \
  --output-prefix "$PREFIX" \
  --num-samples 3 \
  --sample-size 100000 \
  --random-seed 42

for sample_csv in "$SAMPLE_DIR"/${PREFIX}_sample_*_100000.csv; do
  out_csv="${sample_csv%.csv}_rsa_sec.csv"
  $PYTHON_BIN "$MODULE_DIR/scripts/large_structure_validation/02_compute_sample_ss_rsa.py" \
    --input-csv "$sample_csv" \
    --pfam-dir "$PRODIVE_DATA_ROOT/shared/PfamA_seed" \
    --output-csv "$out_csv" \
    --workers "${PRODIVE_WORKERS:-20}" \
    --max-len-diff 5 \
    --min-mean-plddt 70 \
    --max-head-tail-pae 10 \
    --min-found-ratio 0.8
done
