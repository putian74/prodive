#!/usr/bin/env bash
set -euo pipefail

: "${PRODIVE_DATA_ROOT:?Set PRODIVE_DATA_ROOT to the external ProDive data archive data/ directory}"
: "${PRODIVE_WORK_ROOT:?Set PRODIVE_WORK_ROOT to a writable output directory}"
PYTHON=${PYTHON:-python3}

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${PRODIVE_WORK_ROOT}/start2fold_hdx"
mkdir -p "$OUT_ROOT"

XML_DIR="${PRODIVE_DATA_ROOT}/start2fold_hdx/start2fold_xml"
PARSED_CSV="${OUT_ROOT}/parsed_start2fold_all_classes.csv"
SEED_MATCH_DIR="${OUT_ROOT}/start2fold_seed_match_output"
OVERLAP_OUT_DIR="${OUT_ROOT}/global_hmm_to_start2fold_overlap_with_random"
MULTITYPE_OUT_DIR="${OVERLAP_OUT_DIR}/multitype_analysis_with_random"

"$PYTHON" "$MODULE_ROOT/scripts/02_parse_start2fold_xml.py"   --dataset-dir "$XML_DIR"   --output-csv "$PARSED_CSV"

"$PYTHON" "$MODULE_ROOT/scripts/03_match_start2fold_to_pfam_seed.py"   --start2fold-csv "$PARSED_CSV"   --pfam-seed-dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed"   --out-dir "$SEED_MATCH_DIR"   --residue-index-mode uniprot

"$PYTHON" "$MODULE_ROOT/scripts/04_map_prodive_segments_to_start2fold_and_random.py"   --global-score-file "${PRODIVE_DATA_ROOT}/shared/global_high_score_summary_fin.csv"   --covered-protein-file "${SEED_MATCH_DIR}/start2fold_proteins_covered_in_seed.csv"   --detail-file "${SEED_MATCH_DIR}/start2fold_seed_match_detail.csv"   --pfam-dir "${PRODIVE_DATA_ROOT}/shared/PfamA_seed"   --output-dir "$OVERLAP_OUT_DIR"   --random-samples-per-segment "${RANDOM_SAMPLES_PER_SEGMENT:-1000}"   --random-seed "${RANDOM_SEED:-20260409}"

"$PYTHON" "$MODULE_ROOT/scripts/05_multitype_real_vs_random_analysis.py"   --real-input-file "${OVERLAP_OUT_DIR}/all_mapped_segments_with_start2fold_metrics.csv"   --random-input-file "${OVERLAP_OUT_DIR}/random_sampled_windows_all.csv"   --out-dir "$MULTITYPE_OUT_DIR"   --thresh-mode count   --count-threshold 2

"$PYTHON" "$MODULE_ROOT/scripts/06_plot_start2fold_hdx_results.py"   --result-dir "$OVERLAP_OUT_DIR"   --plot-dir "${OVERLAP_OUT_DIR}/plots_only"

"$PYTHON" "$MODULE_ROOT/scripts/07_plot_selected_stf_fragment_intervals.py"   --multitype-file "${MULTITYPE_OUT_DIR}/actual_segment_multitype_annotations.csv"   --raw-file "${OVERLAP_OUT_DIR}/all_mapped_segments_with_start2fold_metrics.csv"   --out-dir "${OVERLAP_OUT_DIR}/fragment_interval_plots"   --target-stf-ids STF0006 STF0025

"$PYTHON" "$MODULE_ROOT/scripts/08_plot_exact_combo_bubbles.py"   --input-csv "${OVERLAP_OUT_DIR}/all_mapped_segments_with_start2fold_metrics.csv"   --output-plot "${MULTITYPE_OUT_DIR}/exact_combo_comparison_bubbles.png"
