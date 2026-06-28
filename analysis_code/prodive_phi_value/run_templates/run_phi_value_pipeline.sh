#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${PRODIVE_WORK_ROOT}/phi_value"
mkdir -p "$OUT_ROOT"

SIFTS_OUT="${OUT_ROOT}/pfam_seed_sifts_output_2sm_dual"
MISSING_OUT="${OUT_ROOT}/pfam_missing_check"
FRAGMENT_OUT="${OUT_ROOT}/pfam_fragment_extract"
MAPPING_OUT="${OUT_ROOT}/batch_seed_interval_mapping"

"$PYTHON" "$MODULE_ROOT/scripts/01_match_pfdb_seed_sifts_residues.py"   --pfdb-2sm-csv "${PRODIVE_DATA_ROOT}/phi_value/Final_2Sm.csv"   --pfam-seed-dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed"   --out-root "$SIFTS_OUT"

"$PYTHON" "$MODULE_ROOT/scripts/02_check_phi_pfam_ids_in_missing_list.py"   --pfam-csv "${SIFTS_OUT}/author_numbering/sifts_residue_match_best_hit.csv"   --missing-txt "${PRODIVE_DATA_ROOT}/phi_value/missing_pfam_ids.txt"   --out-dir "$MISSING_OUT"

"$PYTHON" "$MODULE_ROOT/scripts/03_extract_global_score_rows_for_phi_pfams.py"   --best-hit-file "${SIFTS_OUT}/author_numbering/sifts_residue_match_best_hit.csv"   --global-score-file "${PRODIVE_DATA_ROOT}/shared/global_high_score_summary_fin.csv"   --out-dir "$FRAGMENT_OUT"

"$PYTHON" "$MODULE_ROOT/scripts/04_map_phi_related_fragments_to_seed_intervals.py"   --sifts-csv "${SIFTS_OUT}/author_numbering/sifts_residue_match_best_hit.csv"   --pfam-dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed"   --matches-dir "${FRAGMENT_OUT}/per_pfam"   --output-root "$MAPPING_OUT"
