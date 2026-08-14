#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external data directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
: "${PFAM_STRUCTURE_ROOT:?Set PFAM_STRUCTURE_ROOT to the PfamA_seed_structure directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${PRODIVE_WORK_ROOT}/3did_interface"
mkdir -p "$OUT_DIR"

"$PYTHON" "$MODULE_ROOT/scripts/01_map_fragments_to_3did_interfaces.py" \
  --input-csv "${PRODIVE_DATA_ROOT}/shared/global_high_score_summary_fin.csv" \
  --pfam-seed-dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed" \
  --pfam-structure-dir "$PFAM_STRUCTURE_ROOT" \
  --three-did-flat "${PRODIVE_DATA_ROOT}/3did/3did_flat.gz" \
  --output-csv "${OUT_DIR}/pfam_pfam_3did_interface.csv" \
  --min-seq-ratio 0.8 \
  --max-seq-ratio 1.2 \
  --min-struct-coverage 0.8 \
  --max-search-depth 200 \
  --random-n "${RANDOM_SAMPLES:-200}" \
  --random-seed "${RANDOM_SEED:-20260601}" \
  --workers "${WORKERS:-20}"
