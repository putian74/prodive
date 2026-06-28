#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

: "${ESM2_MODEL_DIR:?Set ESM2_MODEL_DIR to the local ESM2 model checkpoint directory}"
MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_ROOT="${PRODIVE_WORK_ROOT}/esm2/pfam"
REP_DIR="${WORK_ROOT}/01_representatives"
INFER_DIR="${WORK_ROOT}/02_inference"
ENTROPY_DIR="${WORK_ROOT}/03_entropy"
MSA_DIR="${WORK_ROOT}/04_esm_vs_msa"
FILTER_DIR="${WORK_ROOT}/05_fragment_vs_background_after_msa_filtering"
STRUCT_DIR="${WORK_ROOT}/06_structure_availability"
SS_RSA_DIR="${WORK_ROOT}/07_ss_rsa_entropy"
mkdir -p "$REP_DIR" "$INFER_DIR" "$ENTROPY_DIR" "$MSA_DIR" "$FILTER_DIR" "$STRUCT_DIR" "$SS_RSA_DIR"

"$PYTHON" "$MODULE_ROOT/scripts/01_prepare_pfam_representatives.py"   --input_csv "${PRODIVE_DATA_ROOT}/shared/global_high_score_summary_fin.csv"   --pfam_dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed"   --out_dir "$REP_DIR"

"$PYTHON" "$MODULE_ROOT/scripts/02_split_fragdom_record_ranges.py"   --input "${REP_DIR}/representatives.full.with_frag_ranges.txt"   --output "${WORK_ROOT}/record_lengths.jsonl"

"$PYTHON" "$MODULE_ROOT/scripts/03_run_masked_esm2_inference.py"   --model-dir "$ESM2_MODEL_DIR"   --input "${REP_DIR}/representatives.full.with_frag_ranges.txt"   --record-range "${RECORD_RANGE:-1:100}"   --output "${INFER_DIR}/esm_probabilities_${RECORD_RANGE:-1_100}.dat"   --device "${DEVICE:-cuda}"

"$PYTHON" "$MODULE_ROOT/scripts/04_calculate_entropy_from_probabilities.py"   --fragdom "${REP_DIR}/representatives.full.with_frag_ranges.txt"   --esm-prob-files "${INFER_DIR}"/*.dat   --outdir "$ENTROPY_DIR"   --workers "${WORKERS:-8}"

"$PYTHON" "$MODULE_ROOT/scripts/05_compare_esm2_vs_msa_conservation.py"   --representative-file "${REP_DIR}/representatives.full.with_frag_ranges.txt"   --esm-entropy-tsv "${ENTROPY_DIR}/per_position_entropy.tsv"   --pfam-root "${PRODIVE_DATA_ROOT}/shared/PfamA_seed"   --out-dir "$MSA_DIR"   --workers "${WORKERS:-32}"

"$PYTHON" "$MODULE_ROOT/scripts/06_fragment_vs_background_after_msa_filtering.py"   --representative-file "${REP_DIR}/representatives.full.with_frag_ranges.txt"   --esm-entropy-tsv "${ENTROPY_DIR}/per_position_entropy.tsv"   --complete-fragment-fasta "${REP_DIR}/conserved_fragments.pairs.fasta"   --pfam-root "${PRODIVE_DATA_ROOT}/shared/PfamA_seed"   --precomputed-msa-tsv "${MSA_DIR}/representative_position_msa_conservation.tsv"   --out-dir "$FILTER_DIR"   --workers "${WORKERS:-32}"

"$PYTHON" "$MODULE_ROOT/scripts/07_check_structure_availability.py"   --fragdom-file "${REP_DIR}/representatives.full.with_frag_ranges.txt"   --pfam-root "${PRODIVE_DATA_ROOT}/shared/PfamA_seed"   --out-dir "$STRUCT_DIR"

"$PYTHON" "$MODULE_ROOT/scripts/08_entropy_by_secondary_structure_rsa.py"   --fragdom-file "${REP_DIR}/representatives.full.with_frag_ranges.txt"   --structure-availability-tsv "${STRUCT_DIR}/full_esm_representative_structure_availability.tsv"   --entropy-tsv "${ENTROPY_DIR}/per_position_entropy.tsv"   --out-dir "$SS_RSA_DIR"   --workers "${WORKERS:-32}"
