#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external data directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_ROOT="${PRODIVE_WORK_ROOT}/esm2/pfam"
REP_DIR="${WORK_ROOT}/01_representatives"
INFER_DIR="${WORK_ROOT}/02_inference"
ENTROPY_DIR="${WORK_ROOT}/03_entropy"
MSA_DIR="${WORK_ROOT}/04_esm_vs_msa"
FILTER_DIR="${WORK_ROOT}/05_fragment_vs_background_after_msa_filtering"
mkdir -p "$REP_DIR" "$INFER_DIR" "$ENTROPY_DIR" "$MSA_DIR" "$FILTER_DIR"

"$PYTHON" "$MODULE_ROOT/scripts/01_prepare_pfam_representatives.py" \
  --input_csv "${PRODIVE_DATA_ROOT}/shared/global_high_score_summary_fin.csv" \
  --pfam_dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed" \
  --out_dir "$REP_DIR"

"$PYTHON" "$MODULE_ROOT/scripts/02_split_fragdom_record_ranges.py" \
  --input "${REP_DIR}/representatives.full.with_frag_ranges.txt" \
  --output "${WORK_ROOT}/record_lengths.jsonl"

if [[ -n "${ESM2_PROBABILITY_DIR:-}" ]]; then
  PROBABILITY_DIR="$ESM2_PROBABILITY_DIR"
else
  : "${ESM2_MODEL_DIR:?Set ESM2_MODEL_DIR to a local facebook/esm2_t36_3B_UR50D checkpoint, or set ESM2_PROBABILITY_DIR}"
  RECORD_RANGE_VALUE=${RECORD_RANGE:-1:50}
  if [[ ! "$RECORD_RANGE_VALUE" =~ ^[0-9]+:[0-9]+$ ]]; then
    echo "RECORD_RANGE must use the inclusive START:END form, for example 1:50" >&2
    exit 2
  fi
  RECORD_TAG=${RECORD_RANGE_VALUE/:/-}

  "$PYTHON" "$MODULE_ROOT/scripts/03_run_masked_esm2_inference.py" \
    --model-dir "$ESM2_MODEL_DIR" \
    --input "${REP_DIR}/representatives.full.with_frag_ranges.txt" \
    --record-range "$RECORD_RANGE_VALUE" \
    --output "${INFER_DIR}/representatives-${RECORD_TAG}_3B.dat" \
    --device "${DEVICE:-cuda}"

  if [[ "${INFERENCE_ONLY:-0}" == "1" ]]; then
    exit 0
  fi
  PROBABILITY_DIR="$INFER_DIR"
fi

shopt -s nullglob
PROBABILITY_FILES=("${PROBABILITY_DIR}"/*.dat)
shopt -u nullglob
if (( ${#PROBABILITY_FILES[@]} == 0 )); then
  echo "No ESM2 probability .dat files found in ${PROBABILITY_DIR}" >&2
  exit 1
fi

"$PYTHON" "$MODULE_ROOT/scripts/04_calculate_entropy_from_probabilities.py" \
  --fragdom "${REP_DIR}/representatives.full.with_frag_ranges.txt" \
  --esm-prob-files "${PROBABILITY_FILES[@]}" \
  --outdir "$ENTROPY_DIR" \
  --workers "${WORKERS:-8}"

"$PYTHON" "$MODULE_ROOT/scripts/05_compare_esm2_vs_msa_conservation.py" \
  --representative-file "${REP_DIR}/representatives.full.with_frag_ranges.txt" \
  --esm-entropy-tsv "${ENTROPY_DIR}/per_position_entropy.tsv" \
  --pfam-root "${PRODIVE_DATA_ROOT}/shared/PfamA_seed" \
  --out-dir "$MSA_DIR" \
  --workers "${WORKERS:-32}"

"$PYTHON" "$MODULE_ROOT/scripts/06_fragment_vs_background_after_msa_filtering.py" \
  --representative-file "${REP_DIR}/representatives.full.with_frag_ranges.txt" \
  --esm-entropy-tsv "${ENTROPY_DIR}/per_position_entropy.tsv" \
  --complete-fragment-fasta "${REP_DIR}/conserved_fragments.pairs.fasta" \
  --pfam-root "${PRODIVE_DATA_ROOT}/shared/PfamA_seed" \
  --precomputed-msa-tsv "${MSA_DIR}/representative_position_msa_conservation.tsv" \
  --out-dir "$FILTER_DIR" \
  --workers "${WORKERS:-32}"
