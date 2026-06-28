#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

: "${ESM2_MODEL_DIR:?Set ESM2_MODEL_DIR to the local ESM2 model checkpoint directory}"
MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_ROOT="${PRODIVE_WORK_ROOT}/esm2/denovo"
REP_DIR="${WORK_ROOT}/01_representatives"
INFER_DIR="${WORK_ROOT}/02_inference"
ENTROPY_DIR="${WORK_ROOT}/03_entropy"
mkdir -p "$REP_DIR" "$INFER_DIR" "$ENTROPY_DIR"

"$PYTHON" "$MODULE_ROOT/scripts/01b_prepare_denovo_representatives.py"   --input_csv "${PRODIVE_DATA_ROOT}/shared/denovo_global_high_score_summary_fin.csv"   --denovo_fasta "${PRODIVE_DATA_ROOT}/shared/structures/denovo_structures/all_1927_sequences.fasta"   --pfam_dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed"   --out_dir "$REP_DIR"

"$PYTHON" "$MODULE_ROOT/scripts/02_split_fragdom_record_ranges.py"   --input "${REP_DIR}/full_sequences.mixed.with_frag_ranges.txt"   --output "${WORK_ROOT}/record_lengths.jsonl"

"$PYTHON" "$MODULE_ROOT/scripts/03_run_masked_esm2_inference.py"   --model-dir "$ESM2_MODEL_DIR"   --input "${REP_DIR}/full_sequences.mixed.with_frag_ranges.txt"   --record-range "${RECORD_RANGE:-1:100}"   --output "${INFER_DIR}/esm_probabilities_${RECORD_RANGE:-1_100}.dat"   --device "${DEVICE:-cuda}"

"$PYTHON" "$MODULE_ROOT/scripts/04_calculate_entropy_from_probabilities.py"   --fragdom "${REP_DIR}/full_sequences.mixed.with_frag_ranges.txt"   --esm-prob-files "${INFER_DIR}"/*.dat   --outdir "$ENTROPY_DIR"   --workers "${WORKERS:-8}"
