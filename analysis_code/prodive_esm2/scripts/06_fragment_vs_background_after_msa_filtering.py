#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fragment vs background ESM2 entropy after removing highly conserved MSA positions.

Purpose
-------
This script tests whether fragment positions still show lower ESM2 masked-token
entropy after excluding positions that are already extremely conserved in the
Pfam seed MSA.

It does four main things:
1. Read the full representative sequences.
2. Map each representative sequence back to the corresponding Pfam seed FASTA MSA
   and compute per-position MSA conservation, unless a precomputed MSA table exists.
3. Rebuild fragment coverage from conserved_fragments.pairs.fasta, not from the
   incomplete frag_dom field in the representative file.
4. Compare ESM2 entropy between complete fragment positions and background positions
   under several MSA-conservation filters:
   - no MSA filter
   - remove MSA entropy == 0 positions
   - remove majority_freq >= 0.95 positions
   - remove per-record lowest 0%, 5%, 10%, 20%, and 50% MSA-entropy positions
   - remove global lowest 0%, 5%, 10%, 20%, and 50% MSA-entropy positions

Important interpretation
------------------------
Lower ESM2 entropy means stronger model certainty / stronger sequence constraint.
Lower MSA entropy means stronger MSA conservation.

The fair comparison is:
    define highly conserved MSA positions using an explicit rule,
    remove those positions from both fragment and background,
    then compare remaining fragment vs remaining background ESM2 entropy.

This script outputs two complementary definitions for percentile-based removal:
    1) per-record removal: remove the lowest x% MSA-entropy positions within each representative;
    2) global removal: remove the lowest x% MSA-entropy positions after pooling all analysis positions.

This is a control/sensitivity analysis, not data manipulation, as long as the raw
comparison and all filtering rules are reported. Percentile filters use exact-count
removal with reproducible random tie-breaking at equal MSA entropy boundary values.
"""
from __future__ import annotations

import argparse
import gzip
import math
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from scipy import stats as scipy_stats
except Exception:  # scipy may not be installed on every server
    scipy_stats = None

# ==============================================================================
# Config: edit only this section if paths/thresholds need to change
# ==============================================================================
REPRESENTATIVE_FILE = Path(
    "CHANGE_ME"
)
ESM_ENTROPY_TSV = Path(
    "CHANGE_ME"
)
COMPLETE_FRAGMENT_FASTA = Path(
    "CHANGE_ME"
)
PFAM_ROOT = Path(
    "<PRODIVE_DATA_ROOT>/shared/PfamA_seed"
)

OUT_DIR = Path(
    "CHANGE_ME"
)
# Reuse the MSA table produced by the previous ESM2-vs-MSA-only script if it exists.
# If this file does not exist, this script recomputes MSA conservation and writes a new one under OUT_DIR.
PRECOMPUTED_MSA_TSV = Path(
    "CHANGE_ME"
)
USE_PRECOMPUTED_MSA_IF_AVAILABLE = True

# If True, existing summary outputs and plots under OUT_DIR are removed before running.
OVERWRITE_OUTPUT = True

# If True, the full merged per-position table is written. This can be large.
SAVE_MERGED_ANALYSIS_TABLE = False

# If True, write detailed fragment FASTA mapping status and assigned interval tables.
# These can be useful for debugging but are not needed for the main analysis.
SAVE_FRAGMENT_MAPPING_DETAIL = False

# MSA quality filters used before any fragment/background comparison.
MIN_COLUMN_AA_COUNT = 5
MAX_GAP_FRACTION = 0.80

# Definition of highly conserved MSA positions.
# Per-record lowest x% MSA-entropy positions are removed from both fragment and background.
# 0.00 is an explicit no-removal baseline.
# Per-record percentile filters: each representative sequence gets its own MSA-entropy threshold.
MSA_LOW_ENTROPY_FRACTIONS = [0.00, 0.05, 0.10, 0.20, 0.50]

# 0.00 is an explicit no-removal baseline.
# Global percentile filters: all analysis-ready positions are pooled to get one shared threshold.
GLOBAL_MSA_LOW_ENTROPY_FRACTIONS = [0.00, 0.05, 0.10, 0.20, 0.50]

# Percentile filters use exact-count removal. Positions with lower MSA entropy are
# always removed first. If many positions have the same boundary MSA entropy, only
# the required number of tied positions are removed using a reproducible random
# tie-breaker, instead of removing all tied boundary positions.
PERCENTILE_TIE_BREAK_SEED = 20260526
PERCENTILE_REMOVAL_ROUNDING = "round"  # options: "round", "floor", "ceil"

MAJORITY_FREQ_CUTOFF = 0.95
ENTROPY_ZERO_EPS = 1e-12

# Per-record fragment-vs-background deltas are computed only for records with enough positions in both groups.
MIN_FRAGMENT_POSITIONS_PER_RECORD = 3
MIN_BACKGROUND_POSITIONS_PER_RECORD = 10

# Runtime settings.
NUM_WORKERS = 32
MAX_INFLIGHT = NUM_WORKERS * 4
POSITION_WRITE_CHUNK_SIZE = 200_000
STATUS_WRITE_CHUNK_SIZE = 2_000
MAKE_PLOTS = True

# Histogram settings for position-level ESM2 entropy distributions.
# These reproduce the old count/density style, but use complete fragments rebuilt from
# conserved_fragments.pairs.fasta rather than the incomplete frag_dom field.
HIST_BINS = 80


def str_to_bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_float_list(value: str) -> List[float]:
    if value is None or str(value).strip() == "":
        return []
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def refresh_output_paths() -> None:
    global DIAGNOSTICS_DIR, MSA_DIR, FRAGMENT_MAPPING_DIR, MERGED_DIR
    global ALL_FILTERS_TABLE_DIR, ALL_FILTERS_FIG_DIR, ALL_FILTERS_HIST_DIR
    global MAIN_TABLE_DIR, MAIN_FIG_DIR, MAIN_HIST_DIR
    global GLOBAL_TABLE_DIR, GLOBAL_FIG_DIR, GLOBAL_HIST_DIR
    global ABS_TABLE_DIR, ABS_FIG_DIR, ABS_HIST_DIR, OUTPUT_DIRS
    global MERGE_DIAGNOSTICS_TXT, OUTPUT_STRUCTURE_TXT
    global PER_POSITION_MSA_TSV, RECORD_STATUS_TSV
    global FRAGMENT_MAPPING_SUMMARY_TSV, FRAGMENT_MAPPING_STATUS_TSV, FRAGMENT_INTERVALS_TSV
    global MERGED_ANALYSIS_TSV, FILTER_RETENTION_TSV, GROUP_SUMMARY_TSV, CONTRAST_SUMMARY_TSV
    global PER_RECORD_DELTA_TSV, PER_RECORD_DELTA_SUMMARY_TSV, HISTOGRAM_COUNTS_TSV
    global DISTRIBUTION_SHAPE_SUMMARY_TSV, ENTROPY_REGION_FRACTION_TSV
    global FILTER_RETENTION_PER_RECORD_TSV, CONTRAST_SUMMARY_PER_RECORD_TSV
    global PER_RECORD_DELTA_SUMMARY_PER_RECORD_TSV, HISTOGRAM_COUNTS_PER_RECORD_TSV
    global DISTRIBUTION_SHAPE_SUMMARY_PER_RECORD_TSV, ENTROPY_REGION_FRACTION_PER_RECORD_TSV
    global FILTER_RETENTION_GLOBAL_TSV, CONTRAST_SUMMARY_GLOBAL_TSV
    global PER_RECORD_DELTA_SUMMARY_GLOBAL_TSV, HISTOGRAM_COUNTS_GLOBAL_TSV
    global DISTRIBUTION_SHAPE_SUMMARY_GLOBAL_TSV, ENTROPY_REGION_FRACTION_GLOBAL_TSV
    global FILTER_RETENTION_ABSOLUTE_TSV, CONTRAST_SUMMARY_ABSOLUTE_TSV
    global PER_RECORD_DELTA_SUMMARY_ABSOLUTE_TSV, HISTOGRAM_COUNTS_ABSOLUTE_TSV
    global DISTRIBUTION_SHAPE_SUMMARY_ABSOLUTE_TSV, ENTROPY_REGION_FRACTION_ABSOLUTE_TSV

    DIAGNOSTICS_DIR = OUT_DIR / "00_diagnostics"
    MSA_DIR = OUT_DIR / "01_msa_conservation"
    FRAGMENT_MAPPING_DIR = OUT_DIR / "02_complete_fragment_mapping"
    MERGED_DIR = OUT_DIR / "03_merged_table"

    ALL_FILTERS_TABLE_DIR = OUT_DIR / "04_all_filters" / "tables"
    ALL_FILTERS_FIG_DIR = OUT_DIR / "04_all_filters" / "figures"
    ALL_FILTERS_HIST_DIR = ALL_FILTERS_FIG_DIR / "histograms"

    MAIN_TABLE_DIR = OUT_DIR / "05_main_per_record_percentile" / "tables"
    MAIN_FIG_DIR = OUT_DIR / "05_main_per_record_percentile" / "figures"
    MAIN_HIST_DIR = MAIN_FIG_DIR / "histograms"

    GLOBAL_TABLE_DIR = OUT_DIR / "06_sensitivity_global_percentile" / "tables"
    GLOBAL_FIG_DIR = OUT_DIR / "06_sensitivity_global_percentile" / "figures"
    GLOBAL_HIST_DIR = GLOBAL_FIG_DIR / "histograms"

    ABS_TABLE_DIR = OUT_DIR / "07_absolute_conservation_filters" / "tables"
    ABS_FIG_DIR = OUT_DIR / "07_absolute_conservation_filters" / "figures"
    ABS_HIST_DIR = ABS_FIG_DIR / "histograms"

    OUTPUT_DIRS = [
        OUT_DIR, DIAGNOSTICS_DIR, MSA_DIR, FRAGMENT_MAPPING_DIR, MERGED_DIR,
        ALL_FILTERS_TABLE_DIR, ALL_FILTERS_FIG_DIR, ALL_FILTERS_HIST_DIR,
        MAIN_TABLE_DIR, MAIN_FIG_DIR, MAIN_HIST_DIR,
        GLOBAL_TABLE_DIR, GLOBAL_FIG_DIR, GLOBAL_HIST_DIR,
        ABS_TABLE_DIR, ABS_FIG_DIR, ABS_HIST_DIR,
    ]

    MERGE_DIAGNOSTICS_TXT = DIAGNOSTICS_DIR / "merge_header_overlap_diagnostics.txt"
    OUTPUT_STRUCTURE_TXT = DIAGNOSTICS_DIR / "output_structure.txt"
    PER_POSITION_MSA_TSV = MSA_DIR / "representative_position_msa_conservation.tsv"
    RECORD_STATUS_TSV = MSA_DIR / "representative_fasta_msa_mapping_status.tsv"
    FRAGMENT_MAPPING_SUMMARY_TSV = FRAGMENT_MAPPING_DIR / "complete_fragment_mapping_summary.tsv"
    FRAGMENT_MAPPING_STATUS_TSV = FRAGMENT_MAPPING_DIR / "complete_fragment_fasta_mapping_status.tsv"
    FRAGMENT_INTERVALS_TSV = FRAGMENT_MAPPING_DIR / "complete_fragment_intervals_assigned.tsv"
    MERGED_ANALYSIS_TSV = MERGED_DIR / "esm2_msa_complete_fragment_merged_analysis_table.tsv"

    FILTER_RETENTION_TSV = ALL_FILTERS_TABLE_DIR / "msa_conserved_site_filter_retention_summary.tsv"
    GROUP_SUMMARY_TSV = ALL_FILTERS_TABLE_DIR / "fragment_vs_background_esm2_group_summary.by_msa_filter.tsv"
    CONTRAST_SUMMARY_TSV = ALL_FILTERS_TABLE_DIR / "fragment_vs_background_esm2_contrast_summary.by_msa_filter.tsv"
    PER_RECORD_DELTA_TSV = ALL_FILTERS_TABLE_DIR / "per_record_fragment_vs_background_esm2_delta.by_msa_filter.tsv"
    PER_RECORD_DELTA_SUMMARY_TSV = ALL_FILTERS_TABLE_DIR / "per_record_fragment_vs_background_esm2_delta_summary.by_msa_filter.tsv"
    HISTOGRAM_COUNTS_TSV = ALL_FILTERS_TABLE_DIR / "histogram_counts_fragment_vs_background_esm2_entropy.by_msa_filter.tsv"
    DISTRIBUTION_SHAPE_SUMMARY_TSV = ALL_FILTERS_TABLE_DIR / "fragment_vs_background_esm2_distribution_shape_summary.by_msa_filter.tsv"
    ENTROPY_REGION_FRACTION_TSV = ALL_FILTERS_TABLE_DIR / "fragment_vs_background_esm2_entropy_region_fraction_summary.by_msa_filter.tsv"

    FILTER_RETENTION_PER_RECORD_TSV = MAIN_TABLE_DIR / "msa_conserved_site_filter_retention_summary.per_record_percentile_filters.tsv"
    CONTRAST_SUMMARY_PER_RECORD_TSV = MAIN_TABLE_DIR / "fragment_vs_background_esm2_contrast_summary.per_record_percentile_filters.tsv"
    PER_RECORD_DELTA_SUMMARY_PER_RECORD_TSV = MAIN_TABLE_DIR / "per_record_fragment_vs_background_esm2_delta_summary.per_record_percentile_filters.tsv"
    HISTOGRAM_COUNTS_PER_RECORD_TSV = MAIN_TABLE_DIR / "histogram_counts_fragment_vs_background_esm2_entropy.per_record_percentile_filters.tsv"
    DISTRIBUTION_SHAPE_SUMMARY_PER_RECORD_TSV = MAIN_TABLE_DIR / "fragment_vs_background_esm2_distribution_shape_summary.per_record_percentile_filters.tsv"
    ENTROPY_REGION_FRACTION_PER_RECORD_TSV = MAIN_TABLE_DIR / "fragment_vs_background_esm2_entropy_region_fraction_summary.per_record_percentile_filters.tsv"

    FILTER_RETENTION_GLOBAL_TSV = GLOBAL_TABLE_DIR / "msa_conserved_site_filter_retention_summary.global_percentile_filters.tsv"
    CONTRAST_SUMMARY_GLOBAL_TSV = GLOBAL_TABLE_DIR / "fragment_vs_background_esm2_contrast_summary.global_percentile_filters.tsv"
    PER_RECORD_DELTA_SUMMARY_GLOBAL_TSV = GLOBAL_TABLE_DIR / "per_record_fragment_vs_background_esm2_delta_summary.global_percentile_filters.tsv"
    HISTOGRAM_COUNTS_GLOBAL_TSV = GLOBAL_TABLE_DIR / "histogram_counts_fragment_vs_background_esm2_entropy.global_percentile_filters.tsv"
    DISTRIBUTION_SHAPE_SUMMARY_GLOBAL_TSV = GLOBAL_TABLE_DIR / "fragment_vs_background_esm2_distribution_shape_summary.global_percentile_filters.tsv"
    ENTROPY_REGION_FRACTION_GLOBAL_TSV = GLOBAL_TABLE_DIR / "fragment_vs_background_esm2_entropy_region_fraction_summary.global_percentile_filters.tsv"

    FILTER_RETENTION_ABSOLUTE_TSV = ABS_TABLE_DIR / "msa_conserved_site_filter_retention_summary.absolute_conservation_filters.tsv"
    CONTRAST_SUMMARY_ABSOLUTE_TSV = ABS_TABLE_DIR / "fragment_vs_background_esm2_contrast_summary.absolute_conservation_filters.tsv"
    PER_RECORD_DELTA_SUMMARY_ABSOLUTE_TSV = ABS_TABLE_DIR / "per_record_fragment_vs_background_esm2_delta_summary.absolute_conservation_filters.tsv"
    HISTOGRAM_COUNTS_ABSOLUTE_TSV = ABS_TABLE_DIR / "histogram_counts_fragment_vs_background_esm2_entropy.absolute_conservation_filters.tsv"
    DISTRIBUTION_SHAPE_SUMMARY_ABSOLUTE_TSV = ABS_TABLE_DIR / "fragment_vs_background_esm2_distribution_shape_summary.absolute_conservation_filters.tsv"
    ENTROPY_REGION_FRACTION_ABSOLUTE_TSV = ABS_TABLE_DIR / "fragment_vs_background_esm2_entropy_region_fraction_summary.absolute_conservation_filters.tsv"


def configure_runtime(args: argparse.Namespace) -> None:
    global REPRESENTATIVE_FILE, ESM_ENTROPY_TSV, COMPLETE_FRAGMENT_FASTA, PFAM_ROOT, OUT_DIR
    global PRECOMPUTED_MSA_TSV, USE_PRECOMPUTED_MSA_IF_AVAILABLE, OVERWRITE_OUTPUT
    global SAVE_MERGED_ANALYSIS_TABLE, SAVE_FRAGMENT_MAPPING_DETAIL
    global MIN_COLUMN_AA_COUNT, MAX_GAP_FRACTION, MSA_LOW_ENTROPY_FRACTIONS
    global GLOBAL_MSA_LOW_ENTROPY_FRACTIONS, PERCENTILE_TIE_BREAK_SEED
    global PERCENTILE_REMOVAL_ROUNDING, MAJORITY_FREQ_CUTOFF, ENTROPY_ZERO_EPS
    global MIN_FRAGMENT_POSITIONS_PER_RECORD, MIN_BACKGROUND_POSITIONS_PER_RECORD
    global NUM_WORKERS, MAX_INFLIGHT, POSITION_WRITE_CHUNK_SIZE, STATUS_WRITE_CHUNK_SIZE
    global MAKE_PLOTS, HIST_BINS

    REPRESENTATIVE_FILE = Path(args.representative_file)
    ESM_ENTROPY_TSV = Path(args.esm_entropy_tsv)
    COMPLETE_FRAGMENT_FASTA = Path(args.complete_fragment_fasta)
    PFAM_ROOT = Path(args.pfam_root)
    OUT_DIR = Path(args.out_dir)
    PRECOMPUTED_MSA_TSV = Path(args.precomputed_msa_tsv)

    USE_PRECOMPUTED_MSA_IF_AVAILABLE = bool(args.use_precomputed_msa_if_available)
    OVERWRITE_OUTPUT = bool(args.overwrite_output)
    SAVE_MERGED_ANALYSIS_TABLE = bool(args.save_merged_analysis_table)
    SAVE_FRAGMENT_MAPPING_DETAIL = bool(args.save_fragment_mapping_detail)
    MIN_COLUMN_AA_COUNT = int(args.min_column_aa_count)
    MAX_GAP_FRACTION = float(args.max_gap_fraction)
    MSA_LOW_ENTROPY_FRACTIONS = parse_float_list(args.msa_low_entropy_fractions)
    GLOBAL_MSA_LOW_ENTROPY_FRACTIONS = parse_float_list(args.global_msa_low_entropy_fractions)
    PERCENTILE_TIE_BREAK_SEED = int(args.percentile_tie_break_seed)
    PERCENTILE_REMOVAL_ROUNDING = str(args.percentile_removal_rounding)
    MAJORITY_FREQ_CUTOFF = float(args.majority_freq_cutoff)
    ENTROPY_ZERO_EPS = float(args.entropy_zero_eps)
    MIN_FRAGMENT_POSITIONS_PER_RECORD = int(args.min_fragment_positions_per_record)
    MIN_BACKGROUND_POSITIONS_PER_RECORD = int(args.min_background_positions_per_record)
    NUM_WORKERS = int(args.workers)
    MAX_INFLIGHT = int(args.max_inflight) if args.max_inflight is not None else NUM_WORKERS * 4
    POSITION_WRITE_CHUNK_SIZE = int(args.position_write_chunk_size)
    STATUS_WRITE_CHUNK_SIZE = int(args.status_write_chunk_size)
    MAKE_PLOTS = bool(args.make_plots)
    HIST_BINS = int(args.hist_bins)
    refresh_output_paths()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare fragment and background ESM2 entropy after removing MSA-conserved sites."
    )
    parser.add_argument("--representative-file", default=str(REPRESENTATIVE_FILE))
    parser.add_argument("--esm-entropy-tsv", default=str(ESM_ENTROPY_TSV))
    parser.add_argument("--complete-fragment-fasta", default=str(COMPLETE_FRAGMENT_FASTA))
    parser.add_argument("--pfam-root", default=str(PFAM_ROOT))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--precomputed-msa-tsv", default=str(PRECOMPUTED_MSA_TSV))
    parser.add_argument("--use-precomputed-msa-if-available", type=str_to_bool, default=USE_PRECOMPUTED_MSA_IF_AVAILABLE)
    parser.add_argument("--overwrite-output", type=str_to_bool, default=OVERWRITE_OUTPUT)
    parser.add_argument("--save-merged-analysis-table", type=str_to_bool, default=SAVE_MERGED_ANALYSIS_TABLE)
    parser.add_argument("--save-fragment-mapping-detail", type=str_to_bool, default=SAVE_FRAGMENT_MAPPING_DETAIL)
    parser.add_argument("--min-column-aa-count", type=int, default=MIN_COLUMN_AA_COUNT)
    parser.add_argument("--max-gap-fraction", type=float, default=MAX_GAP_FRACTION)
    parser.add_argument("--msa-low-entropy-fractions", default=','.join(str(x) for x in MSA_LOW_ENTROPY_FRACTIONS))
    parser.add_argument("--global-msa-low-entropy-fractions", default=','.join(str(x) for x in GLOBAL_MSA_LOW_ENTROPY_FRACTIONS))
    parser.add_argument("--percentile-tie-break-seed", type=int, default=PERCENTILE_TIE_BREAK_SEED)
    parser.add_argument("--percentile-removal-rounding", choices=["round", "floor", "ceil"], default=PERCENTILE_REMOVAL_ROUNDING)
    parser.add_argument("--majority-freq-cutoff", type=float, default=MAJORITY_FREQ_CUTOFF)
    parser.add_argument("--entropy-zero-eps", type=float, default=ENTROPY_ZERO_EPS)
    parser.add_argument("--min-fragment-positions-per-record", type=int, default=MIN_FRAGMENT_POSITIONS_PER_RECORD)
    parser.add_argument("--min-background-positions-per-record", type=int, default=MIN_BACKGROUND_POSITIONS_PER_RECORD)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max-inflight", type=int, default=None)
    parser.add_argument("--position-write-chunk-size", type=int, default=POSITION_WRITE_CHUNK_SIZE)
    parser.add_argument("--status-write-chunk-size", type=int, default=STATUS_WRITE_CHUNK_SIZE)
    parser.add_argument("--make-plots", type=str_to_bool, default=MAKE_PLOTS)
    parser.add_argument("--hist-bins", type=int, default=HIST_BINS)
    return parser


# ==============================================================================
# Output files and categorized subdirectories
# ==============================================================================
DIAGNOSTICS_DIR = OUT_DIR / "00_diagnostics"
MSA_DIR = OUT_DIR / "01_msa_conservation"
FRAGMENT_MAPPING_DIR = OUT_DIR / "02_complete_fragment_mapping"
MERGED_DIR = OUT_DIR / "03_merged_table"

ALL_FILTERS_TABLE_DIR = OUT_DIR / "04_all_filters" / "tables"
ALL_FILTERS_FIG_DIR = OUT_DIR / "04_all_filters" / "figures"
ALL_FILTERS_HIST_DIR = ALL_FILTERS_FIG_DIR / "histograms"

MAIN_TABLE_DIR = OUT_DIR / "05_main_per_record_percentile" / "tables"
MAIN_FIG_DIR = OUT_DIR / "05_main_per_record_percentile" / "figures"
MAIN_HIST_DIR = MAIN_FIG_DIR / "histograms"

GLOBAL_TABLE_DIR = OUT_DIR / "06_sensitivity_global_percentile" / "tables"
GLOBAL_FIG_DIR = OUT_DIR / "06_sensitivity_global_percentile" / "figures"
GLOBAL_HIST_DIR = GLOBAL_FIG_DIR / "histograms"

ABS_TABLE_DIR = OUT_DIR / "07_absolute_conservation_filters" / "tables"
ABS_FIG_DIR = OUT_DIR / "07_absolute_conservation_filters" / "figures"
ABS_HIST_DIR = ABS_FIG_DIR / "histograms"

OUTPUT_DIRS = [
    OUT_DIR,
    DIAGNOSTICS_DIR,
    MSA_DIR,
    FRAGMENT_MAPPING_DIR,
    MERGED_DIR,
    ALL_FILTERS_TABLE_DIR,
    ALL_FILTERS_FIG_DIR,
    ALL_FILTERS_HIST_DIR,
    MAIN_TABLE_DIR,
    MAIN_FIG_DIR,
    MAIN_HIST_DIR,
    GLOBAL_TABLE_DIR,
    GLOBAL_FIG_DIR,
    GLOBAL_HIST_DIR,
    ABS_TABLE_DIR,
    ABS_FIG_DIR,
    ABS_HIST_DIR,
]


def ensure_output_dirs() -> None:
    for d in OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def write_output_structure() -> None:
    lines = [
        f"OUT_DIR: {OUT_DIR}",
        "",
        "00_diagnostics/",
        "  merge_header_overlap_diagnostics.txt",
        "  output_structure.txt",
        "01_msa_conservation/",
        "  representative_position_msa_conservation.tsv",
        "  representative_fasta_msa_mapping_status.tsv",
        "02_complete_fragment_mapping/",
        "  complete_fragment_mapping_summary.tsv",
        "  complete_fragment_fasta_mapping_status.tsv (optional)",
        "  complete_fragment_intervals_assigned.tsv (optional)",
        "03_merged_table/",
        "  esm2_msa_complete_fragment_merged_analysis_table.tsv (optional)",
        "04_all_filters/",
        "  tables/",
        "    fragment_vs_background_esm2_distribution_shape_summary.by_msa_filter.tsv",
        "    fragment_vs_background_esm2_entropy_region_fraction_summary.by_msa_filter.tsv",
        "  figures/",
        "  figures/histograms/",
        "05_main_per_record_percentile/",
        "  tables/",
        "    fragment_vs_background_esm2_distribution_shape_summary.per_record_percentile_filters.tsv",
        "    fragment_vs_background_esm2_entropy_region_fraction_summary.per_record_percentile_filters.tsv",
        "  figures/",
        "  figures/histograms/",
        "06_sensitivity_global_percentile/",
        "  tables/",
        "    fragment_vs_background_esm2_distribution_shape_summary.global_percentile_filters.tsv",
        "    fragment_vs_background_esm2_entropy_region_fraction_summary.global_percentile_filters.tsv",
        "  figures/",
        "  figures/histograms/",
        "07_absolute_conservation_filters/",
        "  tables/",
        "    fragment_vs_background_esm2_distribution_shape_summary.absolute_conservation_filters.tsv",
        "    fragment_vs_background_esm2_entropy_region_fraction_summary.absolute_conservation_filters.tsv",
        "  figures/",
        "  figures/histograms/",
    ]
    OUTPUT_STRUCTURE_TXT.write_text("\n".join(lines) + "\n")


MERGE_DIAGNOSTICS_TXT = DIAGNOSTICS_DIR / "merge_header_overlap_diagnostics.txt"
OUTPUT_STRUCTURE_TXT = DIAGNOSTICS_DIR / "output_structure.txt"

PER_POSITION_MSA_TSV = MSA_DIR / "representative_position_msa_conservation.tsv"
RECORD_STATUS_TSV = MSA_DIR / "representative_fasta_msa_mapping_status.tsv"

FRAGMENT_MAPPING_SUMMARY_TSV = FRAGMENT_MAPPING_DIR / "complete_fragment_mapping_summary.tsv"
FRAGMENT_MAPPING_STATUS_TSV = FRAGMENT_MAPPING_DIR / "complete_fragment_fasta_mapping_status.tsv"
FRAGMENT_INTERVALS_TSV = FRAGMENT_MAPPING_DIR / "complete_fragment_intervals_assigned.tsv"

MERGED_ANALYSIS_TSV = MERGED_DIR / "esm2_msa_complete_fragment_merged_analysis_table.tsv"

FILTER_RETENTION_TSV = ALL_FILTERS_TABLE_DIR / "msa_conserved_site_filter_retention_summary.tsv"
GROUP_SUMMARY_TSV = ALL_FILTERS_TABLE_DIR / "fragment_vs_background_esm2_group_summary.by_msa_filter.tsv"
CONTRAST_SUMMARY_TSV = ALL_FILTERS_TABLE_DIR / "fragment_vs_background_esm2_contrast_summary.by_msa_filter.tsv"
PER_RECORD_DELTA_TSV = ALL_FILTERS_TABLE_DIR / "per_record_fragment_vs_background_esm2_delta.by_msa_filter.tsv"
PER_RECORD_DELTA_SUMMARY_TSV = ALL_FILTERS_TABLE_DIR / "per_record_fragment_vs_background_esm2_delta_summary.by_msa_filter.tsv"
HISTOGRAM_COUNTS_TSV = ALL_FILTERS_TABLE_DIR / "histogram_counts_fragment_vs_background_esm2_entropy.by_msa_filter.tsv"
DISTRIBUTION_SHAPE_SUMMARY_TSV = ALL_FILTERS_TABLE_DIR / "fragment_vs_background_esm2_distribution_shape_summary.by_msa_filter.tsv"
ENTROPY_REGION_FRACTION_TSV = ALL_FILTERS_TABLE_DIR / "fragment_vs_background_esm2_entropy_region_fraction_summary.by_msa_filter.tsv"

# Main result: per-record lowest 0%, 5%, 10%, 20%, and 50%.
FILTER_RETENTION_PER_RECORD_TSV = MAIN_TABLE_DIR / "msa_conserved_site_filter_retention_summary.per_record_percentile_filters.tsv"
CONTRAST_SUMMARY_PER_RECORD_TSV = MAIN_TABLE_DIR / "fragment_vs_background_esm2_contrast_summary.per_record_percentile_filters.tsv"
PER_RECORD_DELTA_SUMMARY_PER_RECORD_TSV = MAIN_TABLE_DIR / "per_record_fragment_vs_background_esm2_delta_summary.per_record_percentile_filters.tsv"
HISTOGRAM_COUNTS_PER_RECORD_TSV = MAIN_TABLE_DIR / "histogram_counts_fragment_vs_background_esm2_entropy.per_record_percentile_filters.tsv"
DISTRIBUTION_SHAPE_SUMMARY_PER_RECORD_TSV = MAIN_TABLE_DIR / "fragment_vs_background_esm2_distribution_shape_summary.per_record_percentile_filters.tsv"
ENTROPY_REGION_FRACTION_PER_RECORD_TSV = MAIN_TABLE_DIR / "fragment_vs_background_esm2_entropy_region_fraction_summary.per_record_percentile_filters.tsv"

# Sensitivity check: global lowest 0%, 5%, 10%, 20%, and 50%.
FILTER_RETENTION_GLOBAL_TSV = GLOBAL_TABLE_DIR / "msa_conserved_site_filter_retention_summary.global_percentile_filters.tsv"
CONTRAST_SUMMARY_GLOBAL_TSV = GLOBAL_TABLE_DIR / "fragment_vs_background_esm2_contrast_summary.global_percentile_filters.tsv"
PER_RECORD_DELTA_SUMMARY_GLOBAL_TSV = GLOBAL_TABLE_DIR / "per_record_fragment_vs_background_esm2_delta_summary.global_percentile_filters.tsv"
HISTOGRAM_COUNTS_GLOBAL_TSV = GLOBAL_TABLE_DIR / "histogram_counts_fragment_vs_background_esm2_entropy.global_percentile_filters.tsv"
DISTRIBUTION_SHAPE_SUMMARY_GLOBAL_TSV = GLOBAL_TABLE_DIR / "fragment_vs_background_esm2_distribution_shape_summary.global_percentile_filters.tsv"
ENTROPY_REGION_FRACTION_GLOBAL_TSV = GLOBAL_TABLE_DIR / "fragment_vs_background_esm2_entropy_region_fraction_summary.global_percentile_filters.tsv"

# Absolute filters are kept separately as auxiliary controls.
FILTER_RETENTION_ABSOLUTE_TSV = ABS_TABLE_DIR / "msa_conserved_site_filter_retention_summary.absolute_conservation_filters.tsv"
CONTRAST_SUMMARY_ABSOLUTE_TSV = ABS_TABLE_DIR / "fragment_vs_background_esm2_contrast_summary.absolute_conservation_filters.tsv"
PER_RECORD_DELTA_SUMMARY_ABSOLUTE_TSV = ABS_TABLE_DIR / "per_record_fragment_vs_background_esm2_delta_summary.absolute_conservation_filters.tsv"
HISTOGRAM_COUNTS_ABSOLUTE_TSV = ABS_TABLE_DIR / "histogram_counts_fragment_vs_background_esm2_entropy.absolute_conservation_filters.tsv"
DISTRIBUTION_SHAPE_SUMMARY_ABSOLUTE_TSV = ABS_TABLE_DIR / "fragment_vs_background_esm2_distribution_shape_summary.absolute_conservation_filters.tsv"
ENTROPY_REGION_FRACTION_ABSOLUTE_TSV = ABS_TABLE_DIR / "fragment_vs_background_esm2_entropy_region_fraction_summary.absolute_conservation_filters.tsv"

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)
RESIDUE_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
GAP_SET = {"-", ".", "_", "~"}

POSITION_COLUMNS = [
    "header", "pfam_id", "uid", "src_header", "selected_fasta_row_id", "alignment_path", "alignment_type",
    "domain_len", "position", "msa_column", "rep_aa", "msa_total_rows", "msa_aa_count", "msa_gap_count",
    "msa_nonstandard_count", "msa_invalid_count", "msa_missing_count", "gap_fraction", "nonstandard_fraction",
    "msa_entropy", "msa_entropy_norm", "msa_information_uniform", "majority_aa", "majority_freq",
    "effective_aa_number", "mapping_status", "row_match_status",
]

STATUS_COLUMNS = [
    "header", "pfam_id", "uid", "src_header", "alignment_path", "alignment_type", "selected_fasta_row_id",
    "row_match_status", "domain_len", "sequence_len", "msa_start", "msa_end", "mapped_positions",
    "mismatch_count", "mapping_status", "status", "error_message",
]

_FASTA_CACHE: Dict[str, Tuple[Optional[Dict[str, str]], Optional[Path], str]] = {}

# ==============================================================================
# Utilities
# ==============================================================================
def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore")
    return open(path, "r", errors="ignore")


def parse_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def parse_range(x: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"(\d+)-(\d+)", str(x))
    if not m:
        return None, None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b) if b >= a else (None, None)


def parse_pipe_header(h: str) -> Dict[str, str]:
    parts = str(h).strip().split("|")
    out = {"pfam_id": parts[0].strip() if parts else ""}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def clean_seq(seq: str) -> str:
    out: List[str] = []
    for c in str(seq).upper():
        if c in GAP_SET or c.isspace():
            continue
        if c in RESIDUE_SET:
            out.append(c)
    return "".join(out)


def acc_base(x: str) -> str:
    return str(x).strip().split(".")[0] if x else ""


def range_suffix(x: str) -> str:
    m = re.search(r"/\d+-\d+\s*$", str(x).strip())
    return m.group(0) if m else ""


def summary_num(s: pd.Series, prefix: str) -> Dict[str, object]:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_n": int(x.size),
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_median": float(x.median()),
        f"{prefix}_std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        f"{prefix}_q25": float(x.quantile(0.25)),
        f"{prefix}_q75": float(x.quantile(0.75)),
        f"{prefix}_min": float(x.min()),
        f"{prefix}_max": float(x.max()),
    }


def pooled_sd(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    denom = nx + ny - 2
    if denom <= 0:
        return np.nan
    return float(math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / denom))


def welch_ttest_pvalue(x: pd.Series, y: pd.Series) -> float:
    if scipy_stats is None:
        return np.nan
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    return float(scipy_stats.ttest_ind(x, y, equal_var=False, nan_policy="omit").pvalue)


def append_rows(rows: List[Dict[str, object]], path: Path, cols: Optional[List[str]] = None) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if cols is not None:
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols]
    mode = "a" if path.exists() else "w"
    df.to_csv(path, sep="\t", index=False, mode=mode, header=(mode == "w"))

# ==============================================================================
# Representative file
# ==============================================================================
def iter_representatives(path: Path) -> Iterable[Dict[str, object]]:
    header: Optional[str] = None
    first_seq_line: Optional[str] = None
    seq_parts: List[str] = []

    def flush() -> Optional[Dict[str, object]]:
        if header is None:
            return None
        info = parse_pipe_header(header)
        msa_start, msa_end = parse_range(info.get("msa_span", ""))
        seq = clean_seq("".join(seq_parts))
        try:
            n = int(info.get("len", len(seq)))
        except Exception:
            n = len(seq)
        return {
            "header": header,
            "pfam_id": info.get("pfam_id", ""),
            "uid": info.get("uid", ""),
            "src_header": info.get("src_header", ""),
            "msa_start": msa_start,
            "msa_end": msa_end,
            "len": n,
            "sequence": seq,
        }

    with open_text(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                rec = flush()
                if rec is not None:
                    yield rec
                header = line[1:].strip()
                first_seq_line = None
                seq_parts = []
            else:
                if header is None:
                    continue
                if first_seq_line is None:
                    first_seq_line = line
                    # Representative files may have a first metadata line such as frag_dom=...
                    if line.startswith("frag_dom="):
                        continue
                seq_parts.append(line)
    rec = flush()
    if rec is not None:
        yield rec


def load_representatives() -> Tuple[List[Dict[str, object]], Dict[str, str], Dict[Tuple[str, str], List[str]], Dict[str, int]]:
    if not REPRESENTATIVE_FILE.exists():
        raise FileNotFoundError(REPRESENTATIVE_FILE)
    records = list(iter_representatives(REPRESENTATIVE_FILE))
    header_to_seq: Dict[str, str] = {}
    header_to_len: Dict[str, int] = {}
    key_to_headers: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for r in records:
        h = str(r["header"])
        seq = str(r["sequence"])
        header_to_seq[h] = seq
        header_to_len[h] = len(seq)
        key_to_headers[(str(r["pfam_id"]), str(r["uid"]))].append(h)
    print(f"[INFO] representative records: {len(records):,}")
    print(f"[INFO] PF+uid keys: {len(key_to_headers):,}")
    return records, header_to_seq, key_to_headers, header_to_len

# ==============================================================================
# FASTA MSA and representative-to-MSA mapping
# ==============================================================================
def load_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    cur: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        if cur is not None:
            seqs[cur] = "".join(buf).replace(" ", "").replace("\t", "").upper()

    with open_text(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                cur = line[1:].strip()
                buf.clear()
            else:
                if cur is not None:
                    buf.append(line)
    flush()
    return seqs


def find_fasta_for_pfam(pfam_id: str) -> Tuple[Optional[Path], str]:
    fam_dir = PFAM_ROOT / pfam_id
    for ext in ["fas", "fasta", "fa", "fas.gz", "fasta.gz", "fa.gz"]:
        p = fam_dir / f"{pfam_id}.{ext}"
        if p.exists():
            return p, "FASTA"
    return None, "NOT_FOUND"


def split_src(src: str) -> Tuple[str, str]:
    m = re.match(r"^([A-Za-z0-9]+(?:\.\d+)?)_(.+/\d+-\d+)$", str(src).strip())
    return (m.group(1), m.group(2)) if m else ("", str(src).strip())


def fasta_variants(h: str) -> List[str]:
    h = str(h).strip()
    hs = re.sub(r"\s+", " ", h)
    hu = re.sub(r"\s+", "_", hs)
    out = {h, hs, hu}
    t = hs.split()
    if t:
        out.add(t[0])
    if len(t) >= 2:
        out.update([t[1], t[0] + "_" + t[1], t[0] + " " + t[1]])
    return [x for x in out if x]


def src_variants(src: str) -> List[str]:
    src = str(src).strip()
    out = {src}
    acc, rest = split_src(src)
    if acc and rest:
        out.update([acc, rest, acc + "_" + rest, acc + " " + rest])
    return [x for x in out if x]


def build_variant_index(seqs: Dict[str, str]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = defaultdict(list)
    for h in seqs:
        for v in fasta_variants(h):
            idx[v].append(h)
    return idx


def fasta_acc(header: str) -> str:
    return str(header).strip().split()[0] if str(header).strip() else ""


def find_selected_row(
    seqs: Dict[str, str],
    src_header: str,
    uid: str,
    rep_seq: str,
) -> Tuple[Optional[str], Optional[str], str]:
    rep_seq = clean_seq(rep_seq)
    idx = build_variant_index(seqs)

    for v in src_variants(src_header):
        hits = list(dict.fromkeys(idx.get(v, [])))
        if len(hits) == 1:
            h = hits[0]
            return h, seqs[h], "HEADER_VARIANT_UNIQUE_MATCH"
        if len(hits) > 1:
            content = [h for h in hits if clean_seq(seqs[h]) == rep_seq]
            if len(content) == 1:
                h = content[0]
                return h, seqs[h], "HEADER_VARIANT_AMBIGUOUS_RESOLVED_BY_SEQUENCE"
            if len(content) > 1:
                return None, None, "AMBIGUOUS_HEADER_VARIANT_AND_SEQUENCE_MATCH"
            return None, None, "AMBIGUOUS_HEADER_VARIANT_MATCH"

    acc, _ = split_src(src_header)
    accb = acc_base(acc)
    uidb = acc_base(uid)
    suf = range_suffix(src_header)

    cand = []
    for h, s in seqs.items():
        ha = fasta_acc(h)
        hab = acc_base(ha)
        ok = (acc and (acc == ha or accb == hab)) or (uid and (uid == ha or uidb == hab))
        if ok and suf and suf in h:
            cand.append((h, s))
    if len(cand) == 1:
        return cand[0][0], cand[0][1], "ACCESSION_AND_RANGE_MATCH"
    if len(cand) > 1:
        content = [(h, s) for h, s in cand if clean_seq(s) == rep_seq]
        if len(content) == 1:
            return content[0][0], content[0][1], "ACCESSION_AND_RANGE_AMBIGUOUS_RESOLVED_BY_SEQUENCE"
        if len(content) > 1:
            return None, None, "AMBIGUOUS_ACCESSION_AND_RANGE_AND_SEQUENCE_MATCH"
        return None, None, "AMBIGUOUS_ACCESSION_AND_RANGE_MATCH"

    if suf:
        cand = [(h, s) for h, s in seqs.items() if suf in h]
        if len(cand) == 1:
            return cand[0][0], cand[0][1], "RANGE_ONLY_UNIQUE_MATCH"
        if len(cand) > 1:
            content = [(h, s) for h, s in cand if clean_seq(s) == rep_seq]
            if len(content) == 1:
                return content[0][0], content[0][1], "RANGE_ONLY_AMBIGUOUS_RESOLVED_BY_SEQUENCE"
            if len(content) > 1:
                return None, None, "AMBIGUOUS_RANGE_ONLY_AND_SEQUENCE_MATCH"
            return None, None, "AMBIGUOUS_RANGE_ONLY_MATCH"

    cand = [(h, s) for h, s in seqs.items() if clean_seq(s) == rep_seq]
    if len(cand) == 1:
        return cand[0][0], cand[0][1], "SEQUENCE_CONTENT_UNIQUE_MATCH"
    if len(cand) > 1:
        return None, None, "AMBIGUOUS_SEQUENCE_CONTENT_MATCH"
    return None, None, "SELECTED_ROW_NOT_FOUND"


def col_metrics(seqs: Dict[str, str], col: int) -> Dict[str, object]:
    idx = col - 1
    counts = {aa: 0 for aa in AA_ORDER}
    gap = nonstd = invalid = missing = 0
    n = len(seqs)

    for s in seqs.values():
        if idx >= len(s):
            missing += 1
            gap += 1
            continue
        c = s[idx].upper()
        if c in GAP_SET:
            gap += 1
        elif c in AA_SET:
            counts[c] += 1
        elif c in RESIDUE_SET:
            nonstd += 1
        else:
            invalid += 1

    aa_count = sum(counts.values())
    if aa_count <= 0:
        return {
            "msa_total_rows": n,
            "msa_aa_count": 0,
            "msa_gap_count": gap,
            "msa_nonstandard_count": nonstd,
            "msa_invalid_count": invalid,
            "msa_missing_count": missing,
            "gap_fraction": gap / n if n else np.nan,
            "nonstandard_fraction": nonstd / n if n else np.nan,
            "msa_entropy": np.nan,
            "msa_entropy_norm": np.nan,
            "msa_information_uniform": np.nan,
            "majority_aa": "",
            "majority_freq": np.nan,
            "effective_aa_number": np.nan,
        }

    p = np.array([counts[aa] / aa_count for aa in AA_ORDER], dtype=float)
    nz = p[p > 0]
    ent = -float(np.sum(nz * np.log(nz)))
    maj = max(counts, key=lambda aa: counts[aa])
    return {
        "msa_total_rows": n,
        "msa_aa_count": aa_count,
        "msa_gap_count": gap,
        "msa_nonstandard_count": nonstd,
        "msa_invalid_count": invalid,
        "msa_missing_count": missing,
        "gap_fraction": gap / n if n else np.nan,
        "nonstandard_fraction": nonstd / n if n else np.nan,
        "msa_entropy": ent,
        "msa_entropy_norm": ent / math.log(20.0),
        "msa_information_uniform": math.log(20.0) - ent,
        "majority_aa": maj,
        "majority_freq": counts[maj] / aa_count,
        "effective_aa_number": math.exp(ent),
    }


def map_rep_to_msa(
    aln_seq: str,
    rep_seq: str,
    msa_start: Optional[int],
    msa_end: Optional[int],
) -> Tuple[Dict[int, int], int, int, str]:
    if msa_start is None or msa_end is None or msa_start < 1 or msa_end < msa_start:
        return {}, 0, 0, "BAD_MSA_SPAN"
    if len(aln_seq) < msa_end:
        return {}, 0, 0, f"ALIGNMENT_ROW_TOO_SHORT({len(aln_seq)}<{msa_end})"

    span = aln_seq[msa_start - 1:msa_end]
    pos_to_col: Dict[int, int] = {}
    obs: List[str] = []
    pos = 0
    for off, c in enumerate(span):
        c = c.upper()
        if c in GAP_SET or c not in RESIDUE_SET:
            continue
        pos += 1
        pos_to_col[pos] = msa_start + off
        obs.append(c)

    obs_s = "".join(obs)
    rep = clean_seq(rep_seq)
    mismatch_count = sum(a != b for a, b in zip(obs_s, rep)) + abs(len(obs_s) - len(rep))
    if len(obs_s) != len(rep):
        status = f"SEQ_LENGTH_MISMATCH(msa_nongap={len(obs_s)},rep_len={len(rep)})"
    elif mismatch_count > 0:
        status = f"SEQ_CONTENT_MISMATCH(n={mismatch_count})"
    else:
        status = "OK"
    return pos_to_col, len(obs_s), mismatch_count, status


def load_fasta_cached(pfam_id: str):
    if pfam_id in _FASTA_CACHE:
        return _FASTA_CACHE[pfam_id]
    p, typ = find_fasta_for_pfam(pfam_id)
    if p is None:
        res = (None, None, typ)
    else:
        try:
            res = (load_fasta(p), p, typ)
        except Exception:
            res = (None, p, "FASTA_LOAD_FAIL")
    _FASTA_CACHE[pfam_id] = res
    return res


def fail_status(
    rec: Dict[str, object],
    aln_path: object,
    aln_type: str,
    selected: str,
    row_status: str,
    mapped: int,
    mism: object,
    map_status: str,
    err: str,
) -> Dict[str, object]:
    return {
        "header": rec.get("header", ""),
        "pfam_id": rec.get("pfam_id", ""),
        "uid": rec.get("uid", ""),
        "src_header": rec.get("src_header", ""),
        "alignment_path": str(aln_path) if aln_path else "",
        "alignment_type": aln_type,
        "selected_fasta_row_id": selected,
        "row_match_status": row_status,
        "domain_len": rec.get("len", np.nan),
        "sequence_len": len(clean_seq(rec.get("sequence", ""))),
        "msa_start": rec.get("msa_start", np.nan),
        "msa_end": rec.get("msa_end", np.nan),
        "mapped_positions": mapped,
        "mismatch_count": mism,
        "mapping_status": map_status,
        "status": "FAIL",
        "error_message": err,
    }


def process_one_record(rec: Dict[str, object]):
    rep = clean_seq(rec.get("sequence", ""))
    if int(rec.get("len", len(rep))) != len(rep):
        return [], fail_status(
            rec, "", "", "", "", 0, np.nan,
            "REPRESENTATIVE_LEN_MISMATCH",
            f"header len={rec.get('len')}, sequence_len={len(rep)}",
        )

    seqs, fasta_path, typ = load_fasta_cached(str(rec["pfam_id"]))
    if seqs is None or not seqs:
        return [], fail_status(
            rec, fasta_path, typ, "", "", 0, np.nan,
            "FASTA_NOT_FOUND_OR_LOAD_FAIL",
            f"No valid FASTA for {rec['pfam_id']}",
        )

    sid, aln_seq, row_status = find_selected_row(seqs, str(rec["src_header"]), str(rec["uid"]), rep)
    if sid is None or aln_seq is None:
        return [], fail_status(
            rec, fasta_path, typ, "", row_status, 0, np.nan,
            "SELECTED_ROW_NOT_FOUND",
            row_status,
        )

    pos_to_col, mapped_n, mism, map_status = map_rep_to_msa(
        aln_seq, rep, rec.get("msa_start"), rec.get("msa_end")
    )
    if mapped_n != int(rec["len"]) or map_status != "OK":
        return [], fail_status(
            rec, fasta_path, typ, sid, row_status, mapped_n, mism,
            map_status,
            f"mapped_n={mapped_n}, domain_len={rec['len']}",
        )

    rows: List[Dict[str, object]] = []
    cache: Dict[int, Dict[str, object]] = {}
    for pos in range(1, int(rec["len"]) + 1):
        col = pos_to_col.get(pos)
        if col is None:
            continue
        if col not in cache:
            cache[col] = col_metrics(seqs, col)
        rows.append({
            "header": rec["header"],
            "pfam_id": rec["pfam_id"],
            "uid": rec["uid"],
            "src_header": rec["src_header"],
            "selected_fasta_row_id": sid,
            "alignment_path": str(fasta_path),
            "alignment_type": typ,
            "domain_len": rec["len"],
            "position": pos,
            "msa_column": col,
            "rep_aa": rep[pos - 1],
            **cache[col],
            "mapping_status": "OK",
            "row_match_status": row_status,
        })

    status = {
        "header": rec["header"],
        "pfam_id": rec["pfam_id"],
        "uid": rec["uid"],
        "src_header": rec["src_header"],
        "alignment_path": str(fasta_path),
        "alignment_type": typ,
        "selected_fasta_row_id": sid,
        "row_match_status": row_status,
        "domain_len": rec["len"],
        "sequence_len": len(rep),
        "msa_start": rec["msa_start"],
        "msa_end": rec["msa_end"],
        "mapped_positions": len(rows),
        "mismatch_count": mism,
        "mapping_status": "OK",
        "status": "OK",
        "error_message": "",
    }
    return rows, status


def run_msa(records: List[Dict[str, object]]) -> None:
    print(f"[INFO] Computing MSA conservation for {len(records):,} records, workers={NUM_WORKERS}")
    pos_buf: List[Dict[str, object]] = []
    status_buf: List[Dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        inflight = {}
        it = iter(records)

        def submit(r: Dict[str, object]) -> None:
            inflight[ex.submit(process_one_record, r)] = 1

        for _ in range(min(MAX_INFLIGHT, len(records))):
            try:
                submit(next(it))
            except StopIteration:
                break

        with tqdm(total=len(records), desc="FASTA MSA conservation", unit="record") as bar:
            while inflight:
                for fut in as_completed(list(inflight.keys())):
                    inflight.pop(fut, None)
                    try:
                        rows, st = fut.result()
                    except Exception as e:
                        rows = []
                        st = {c: "" for c in STATUS_COLUMNS}
                        st.update({
                            "mapping_status": "WORKER_EXCEPTION",
                            "status": "FAIL",
                            "error_message": str(e),
                            "mapped_positions": 0,
                        })
                    pos_buf.extend(rows)
                    status_buf.append(st)
                    bar.update(1)

                    try:
                        submit(next(it))
                    except StopIteration:
                        pass

                    if len(pos_buf) >= POSITION_WRITE_CHUNK_SIZE:
                        append_rows(pos_buf, PER_POSITION_MSA_TSV, POSITION_COLUMNS)
                        pos_buf = []
                    if len(status_buf) >= STATUS_WRITE_CHUNK_SIZE:
                        append_rows(status_buf, RECORD_STATUS_TSV, STATUS_COLUMNS)
                        status_buf = []
                    break

    append_rows(pos_buf, PER_POSITION_MSA_TSV, POSITION_COLUMNS)
    append_rows(status_buf, RECORD_STATUS_TSV, STATUS_COLUMNS)

# ==============================================================================
# Complete fragment assignment from conserved_fragments.pairs.fasta
# ==============================================================================
def iter_fasta_records(path: Path) -> Iterable[Tuple[str, str]]:
    h: Optional[str] = None
    seq: List[str] = []

    def flush() -> Optional[Tuple[str, str]]:
        if h is None:
            return None
        return h, clean_seq("".join(seq))

    with open_text(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                rec = flush()
                if rec is not None:
                    yield rec
                h = line[1:].strip()
                seq = []
            else:
                seq.append(line)
    rec = flush()
    if rec is not None:
        yield rec


def assign_complete_fragments(
    header_to_seq: Dict[str, str],
    key_to_headers: Dict[Tuple[str, str], List[str]],
    header_to_len: Dict[str, int],
) -> Dict[str, np.ndarray]:
    if not COMPLETE_FRAGMENT_FASTA.exists():
        raise FileNotFoundError(COMPLETE_FRAGMENT_FASTA)

    coverage = {h: np.zeros(header_to_len[h], dtype=np.int32) for h in header_to_seq}
    status_counter: Counter = Counter()
    match_counter: Counter = Counter()
    total = 0
    assigned = 0
    status_buf: List[Dict[str, object]] = []
    interval_buf: List[Dict[str, object]] = []

    for rid, (fh, fseq) in enumerate(tqdm(iter_fasta_records(COMPLETE_FRAGMENT_FASTA), desc="Assign complete fragments", unit="fragment")):
        total += 1
        info = parse_pipe_header(fh)
        pfam, uid = info.get("pfam_id", ""), info.get("uid", "")
        hs, he = parse_range(info.get("hmm", ""))
        ms, me = parse_range(info.get("msa", ""))
        candidates = key_to_headers.get((pfam, uid), [])

        status = "FAIL"
        match = ""
        assigned_header = ""
        err = ""

        if hs is None or he is None:
            match = "BAD_HMM_RANGE"
            err = f"Bad hmm={info.get('hmm', '')}"
        elif not candidates:
            match = "NO_CANDIDATE_HEADER"
            err = f"No representative for {(pfam, uid)}"
        else:
            exact = []
            for h in candidates:
                seq = header_to_seq[h]
                if he <= len(seq) and clean_seq(seq[hs - 1:he]) == fseq:
                    exact.append(h)
            if len(exact) == 1:
                assigned_header = exact[0]
                status = "ASSIGNED"
                match = "EXACT_SEQUENCE_AND_HMM_INTERVAL_MATCH"
            elif len(exact) > 1:
                match = "AMBIGUOUS_EXACT_SEQUENCE_MATCH"
                err = f"n={len(exact)}"
            else:
                example = ""
                if candidates and he <= len(header_to_seq[candidates[0]]):
                    example = clean_seq(header_to_seq[candidates[0]][hs - 1:he])[:80]
                match = "NO_EXACT_SEQUENCE_MATCH"
                err = (
                    f"n_candidates={len(candidates)}, frag_len={len(fseq)}, "
                    f"hmm_len={he - hs + 1 if hs and he else np.nan}, first_slice={example}"
                )

        if status == "ASSIGNED":
            arr = coverage[assigned_header]
            if he is not None and he <= len(arr):
                arr[hs - 1:he] += 1
                assigned += 1
                if SAVE_FRAGMENT_MAPPING_DETAIL:
                    interval_buf.append({
                        "fragment_record_id": rid,
                        "fragment_header": fh,
                        "assigned_header": assigned_header,
                        "pfam_id": pfam,
                        "uid": uid,
                        "side": info.get("side", ""),
                        "srcrow": info.get("srcrow", ""),
                        "hmm_start": hs,
                        "hmm_end": he,
                        "hmm_len": he - hs + 1,
                        "msa_start": ms,
                        "msa_end": me,
                        "mate": info.get("mate", ""),
                        "fragment_sequence_len": len(fseq),
                        "match_status": match,
                    })
            else:
                status = "FAIL"
                match = "ASSIGNED_BUT_RANGE_OUT_OF_BOUNDS"
                err = f"hmm_end={he}, len={len(arr)}"

        status_counter[status] += 1
        match_counter[match] += 1

        if SAVE_FRAGMENT_MAPPING_DETAIL:
            status_buf.append({
                "fragment_record_id": rid,
                "fragment_header": fh,
                "pfam_id": pfam,
                "uid": uid,
                "side": info.get("side", ""),
                "srcrow": info.get("srcrow", ""),
                "hmm": info.get("hmm", ""),
                "msa": info.get("msa", ""),
                "mate": info.get("mate", ""),
                "hmm_start": hs,
                "hmm_end": he,
                "hmm_len": he - hs + 1 if hs is not None and he is not None else np.nan,
                "msa_start": ms,
                "msa_end": me,
                "fragment_sequence_len": len(fseq),
                "n_candidate_headers": len(candidates),
                "assigned_header": assigned_header,
                "status": status,
                "match_status": match,
                "error_message": err,
            })
            if len(status_buf) >= 100_000:
                append_rows(status_buf, FRAGMENT_MAPPING_STATUS_TSV)
                status_buf = []
            if len(interval_buf) >= 100_000:
                append_rows(interval_buf, FRAGMENT_INTERVALS_TSV)
                interval_buf = []

    if SAVE_FRAGMENT_MAPPING_DETAIL:
        append_rows(status_buf, FRAGMENT_MAPPING_STATUS_TSV)
        append_rows(interval_buf, FRAGMENT_INTERVALS_TSV)

    summary_rows = []
    for key, value in status_counter.items():
        summary_rows.append({"summary_type": "status", "name": key, "count": int(value), "fraction": value / total if total else np.nan})
    for key, value in match_counter.items():
        summary_rows.append({"summary_type": "match_status", "name": key, "count": int(value), "fraction": value / total if total else np.nan})
    summary_rows.append({"summary_type": "overall", "name": "total_fragment_records", "count": int(total), "fraction": 1.0})
    summary_rows.append({"summary_type": "overall", "name": "assigned_fragment_records", "count": int(assigned), "fraction": assigned / total if total else np.nan})
    summary_rows.append({"summary_type": "overall", "name": "failed_fragment_records", "count": int(total - assigned), "fraction": (total - assigned) / total if total else np.nan})
    pd.DataFrame(summary_rows).to_csv(FRAGMENT_MAPPING_SUMMARY_TSV, sep="\t", index=False)

    print(f"[INFO] complete fragment records: {total:,}; assigned: {assigned:,}; failed: {total - assigned:,}")
    return coverage

# ==============================================================================
# Merge ESM2, MSA, and complete fragment coverage
# ==============================================================================
def select_msa_source(records: List[Dict[str, object]]) -> Path:
    if USE_PRECOMPUTED_MSA_IF_AVAILABLE and PRECOMPUTED_MSA_TSV.exists():
        print(f"[INFO] Reusing precomputed MSA table: {PRECOMPUTED_MSA_TSV}")
        return PRECOMPUTED_MSA_TSV

    if PER_POSITION_MSA_TSV.exists():
        print(f"[INFO] Reusing existing MSA table under OUT_DIR: {PER_POSITION_MSA_TSV}")
        return PER_POSITION_MSA_TSV

    run_msa(records)
    return PER_POSITION_MSA_TSV


def load_entropy_for_merge() -> pd.DataFrame:
    keep = {"header", "position", "wt_aa", "entropy"}
    df = pd.read_csv(ESM_ENTROPY_TSV, sep="\t", usecols=lambda c: c in keep, low_memory=False)
    req = {"header", "position", "wt_aa", "entropy"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"ESM entropy file missing columns: {sorted(miss)}")

    df["header"] = df["header"].astype(str)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    df["wt_aa"] = df["wt_aa"].astype(str).str.upper()
    df["esm_entropy"] = pd.to_numeric(df["entropy"], errors="coerce")
    return df[["header", "position", "wt_aa", "esm_entropy"]]


def load_msa_for_merge(msa_path: Path) -> pd.DataFrame:
    keep = {
        "header", "pfam_id", "uid", "position", "msa_column", "rep_aa",
        "msa_aa_count", "gap_fraction", "nonstandard_fraction",
        "msa_entropy", "msa_entropy_norm", "msa_information_uniform",
        "majority_aa", "majority_freq", "mapping_status",
    }
    df = pd.read_csv(msa_path, sep="\t", usecols=lambda c: c in keep, low_memory=False)
    df["header"] = df["header"].astype(str)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    for c in [
        "msa_aa_count", "gap_fraction", "nonstandard_fraction",
        "msa_entropy", "msa_entropy_norm", "msa_information_uniform", "majority_freq",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def write_merge_diagnostics(msa: pd.DataFrame, ent: pd.DataFrame) -> None:
    mh = set(msa["header"].astype(str).unique())
    eh = set(ent["header"].astype(str).unique())
    mp = set(zip(msa["header"].astype(str), msa["position"].astype(int)))
    ep = set(zip(ent["header"].astype(str), ent["position"].astype(int)))

    with open(MERGE_DIAGNOSTICS_TXT, "w") as f:
        f.write("[HEADER OVERLAP]\n")
        f.write(f"MSA mapped headers: {len(mh)}\n")
        f.write(f"ESM entropy headers: {len(eh)}\n")
        f.write(f"Header intersection: {len(mh & eh)}\n")
        f.write(f"Only in MSA: {len(mh - eh)}\n")
        f.write(f"Only in ESM: {len(eh - mh)}\n\n")
        f.write("[HEADER-POSITION PAIR OVERLAP]\n")
        f.write(f"MSA header-position pairs: {len(mp)}\n")
        f.write(f"ESM header-position pairs: {len(ep)}\n")
        f.write(f"Pair intersection: {len(mp & ep)}\n")

    print(
        f"[INFO] merge headers MSA={len(mh):,}, ESM={len(eh):,}, "
        f"intersection={len(mh & eh):,}, pair_intersection={len(mp & ep):,}"
    )


def annotate_complete_fragment_coverage(merged: pd.DataFrame, coverage: Dict[str, np.ndarray]) -> pd.DataFrame:
    pos = merged["position"].astype(int).to_numpy()
    counts = np.zeros(len(merged), dtype=np.int32)

    for h, idx in tqdm(merged.groupby("header", sort=False).indices.items(), desc="Annotate complete fragments", unit="header"):
        arr = coverage.get(h)
        if arr is None:
            continue
        p0 = pos[idx] - 1
        valid = (p0 >= 0) & (p0 < len(arr))
        tmp = np.zeros(len(idx), dtype=np.int32)
        tmp[valid] = arr[p0[valid]]
        counts[idx] = tmp

    merged["complete_fragment_coverage_count"] = counts
    merged["complete_is_fragment"] = counts > 0
    merged["analysis_is_fragment"] = merged["complete_is_fragment"]
    return merged


def merge_all(msa_path: Path, coverage: Dict[str, np.ndarray]) -> pd.DataFrame:
    ent = load_entropy_for_merge()
    msa = load_msa_for_merge(msa_path)
    write_merge_diagnostics(msa, ent)

    merged = ent.merge(msa, on=["header", "position"], how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged["analysis_ok"] = (
        merged["esm_entropy"].notna()
        & merged["msa_entropy"].notna()
        & (merged["mapping_status"] == "OK")
        & (merged["msa_aa_count"] >= MIN_COLUMN_AA_COUNT)
        & (merged["gap_fraction"] <= MAX_GAP_FRACTION)
    )
    merged = annotate_complete_fragment_coverage(merged, coverage)

    print(
        f"[INFO] merged rows={len(merged):,}; analysis_ok={int(merged['analysis_ok'].sum()):,}; "
        f"complete fragment positions among analysis_ok={int((merged['analysis_ok'] & merged['analysis_is_fragment']).sum()):,}"
    )

    if SAVE_MERGED_ANALYSIS_TABLE:
        merged.to_csv(MERGED_ANALYSIS_TSV, sep="\t", index=False)
        print(f"[INFO] wrote merged analysis table: {MERGED_ANALYSIS_TSV}")

    return merged

# ==============================================================================
# MSA-conserved-site filtering and fragment-vs-background ESM2 analysis
# ==============================================================================
def percent_label(frac: float) -> str:
    pct = int(round(frac * 100))
    return f"{pct}pct"


def per_record_filter_name_for_fraction(frac: float) -> str:
    return f"per_record_remove_msa_entropy_lowest_{percent_label(frac)}"


def global_filter_name_for_fraction(frac: float) -> str:
    return f"global_remove_msa_entropy_lowest_{percent_label(frac)}"


def filter_scope(filter_name: str) -> str:
    """Classify filters for easier downstream interpretation."""
    if filter_name == "raw_no_msa_conserved_site_removal":
        return "raw"
    if filter_name.startswith("per_record_"):
        return "per_record_percentile"
    if filter_name.startswith("global_"):
        return "global_percentile"
    if filter_name.startswith("remove_msa_entropy_eq_0") or filter_name.startswith("remove_majority_freq_ge_"):
        return "absolute_conservation"
    return "other"


def scope_figure_dir(scope: str) -> Path:
    if scope == "per_record_percentile":
        return MAIN_FIG_DIR
    if scope == "global_percentile":
        return GLOBAL_FIG_DIR
    if scope == "absolute_conservation":
        return ABS_FIG_DIR
    return ALL_FILTERS_FIG_DIR


def scope_hist_dir(scope: str) -> Path:
    if scope == "per_record_percentile":
        return MAIN_HIST_DIR
    if scope == "global_percentile":
        return GLOBAL_HIST_DIR
    if scope == "absolute_conservation":
        return ABS_HIST_DIR
    return ALL_FILTERS_HIST_DIR


def percentile_remove_count(n: int, frac: float) -> int:
    """Return the exact number of positions to remove for a percentile filter."""
    if n <= 0 or frac <= 0:
        return 0
    raw = float(n) * float(frac)
    mode = str(PERCENTILE_REMOVAL_ROUNDING).lower().strip()
    if mode == "floor":
        k = int(math.floor(raw))
    elif mode == "ceil":
        k = int(math.ceil(raw))
    elif mode == "round":
        k = int(round(raw))
    else:
        raise ValueError(f"Unknown PERCENTILE_REMOVAL_ROUNDING: {PERCENTILE_REMOVAL_ROUNDING}")
    return max(0, min(int(n), k))


def exact_low_entropy_keep_mask_global(df: pd.DataFrame, frac: float, seed: int) -> pd.Series:
    """Keep mask for global exact-count removal of the lowest MSA-entropy positions.

    Lower MSA entropy is removed first. If tied MSA-entropy values cross the
    boundary, the tied positions are broken by a reproducible random key so that
    exactly the requested number of positions is removed.
    """
    n = len(df)
    keep = np.ones(n, dtype=bool)
    k = percentile_remove_count(n, frac)
    if k <= 0:
        return pd.Series(keep, index=df.index)

    entropy = pd.to_numeric(df["msa_entropy"], errors="coerce").to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    random_key = rng.random(n)
    order = np.lexsort((random_key, entropy))
    keep[order[:k]] = False
    return pd.Series(keep, index=df.index)


def exact_low_entropy_keep_mask_per_record(df: pd.DataFrame, frac: float, seed: int) -> pd.Series:
    """Keep mask for per-record exact-count removal of lowest MSA-entropy positions.

    Each representative/header gets its own removal count, approximately frac * n
    after the configured rounding rule. Within each representative, lower MSA
    entropy is removed first; ties at the boundary are broken reproducibly.
    """
    n_total = len(df)
    keep = np.ones(n_total, dtype=bool)
    if frac <= 0 or n_total == 0:
        return pd.Series(keep, index=df.index)

    entropy = pd.to_numeric(df["msa_entropy"], errors="coerce").to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    random_key = rng.random(n_total)

    for _, locs in df.groupby("header", sort=False).indices.items():
        locs = np.asarray(locs, dtype=int)
        k = percentile_remove_count(len(locs), frac)
        if k <= 0:
            continue
        local_order = locs[np.lexsort((random_key[locs], entropy[locs]))]
        keep[local_order[:k]] = False

    return pd.Series(keep, index=df.index)


def build_filter_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Return keep masks. True means the position remains in the analysis.

    Percentile-based filters are produced in two ways:
    1. per-record filters:
       For each representative/header, remove its own lowest x% MSA-entropy positions.
       This controls for family-to-family differences in overall conservation level.
    2. global filters:
       Pool all analysis-ready positions, compute one global x% MSA-entropy threshold,
       then remove positions below or equal to that single threshold.
       This tests whether conclusions depend on using an absolute pooled cutoff.
    """
    masks: Dict[str, pd.Series] = {}
    masks["raw_no_msa_conserved_site_removal"] = pd.Series(True, index=df.index)

    masks["remove_msa_entropy_eq_0"] = ~(df["msa_entropy"] <= ENTROPY_ZERO_EPS)
    masks[f"remove_majority_freq_ge_{str(MAJORITY_FREQ_CUTOFF).replace('.', 'p')}"] = ~(
        df["majority_freq"] >= MAJORITY_FREQ_CUTOFF
    )

    # Per-record percentile filters with exact-count removal.
    # frac == 0.00 is an explicit no-removal baseline.
    for frac in MSA_LOW_ENTROPY_FRACTIONS:
        seed = PERCENTILE_TIE_BREAK_SEED + int(round(frac * 10000)) + 100000
        masks[per_record_filter_name_for_fraction(frac)] = exact_low_entropy_keep_mask_per_record(df, frac, seed)

    # Global percentile filters with exact-count removal.
    # frac == 0.00 is an explicit no-removal baseline.
    for frac in GLOBAL_MSA_LOW_ENTROPY_FRACTIONS:
        seed = PERCENTILE_TIE_BREAK_SEED + int(round(frac * 10000)) + 200000
        masks[global_filter_name_for_fraction(frac)] = exact_low_entropy_keep_mask_global(df, frac, seed)

    return masks


def group_summary_for_filter(sub: pd.DataFrame, filter_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for flag, group_name in [(True, "fragment"), (False, "background")]:
        g = sub[sub["analysis_is_fragment"] == flag]
        row: Dict[str, object] = {
            "filter": filter_name,
            "filter_scope": filter_scope(filter_name),
            "group": group_name,
            "n_positions": int(len(g)),
            "n_headers": int(g["header"].nunique()) if len(g) else 0,
        }
        for col, prefix in [
            ("esm_entropy", "esm_entropy"),
            ("msa_entropy", "msa_entropy"),
            ("msa_information_uniform", "msa_information_uniform"),
            ("majority_freq", "majority_freq"),
            ("gap_fraction", "gap_fraction"),
            ("complete_fragment_coverage_count", "complete_fragment_coverage_count"),
        ]:
            if col in g.columns:
                row.update(summary_num(g[col], prefix))
        rows.append(row)
    return rows


def contrast_summary_for_filter(sub: pd.DataFrame, filter_name: str) -> Dict[str, object]:
    frag = sub.loc[sub["analysis_is_fragment"], "esm_entropy"]
    bg = sub.loc[~sub["analysis_is_fragment"], "esm_entropy"]

    frag_clean = pd.to_numeric(frag, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    bg_clean = pd.to_numeric(bg, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    mean_frag = float(frag_clean.mean()) if len(frag_clean) else np.nan
    mean_bg = float(bg_clean.mean()) if len(bg_clean) else np.nan
    median_frag = float(frag_clean.median()) if len(frag_clean) else np.nan
    median_bg = float(bg_clean.median()) if len(bg_clean) else np.nan
    psd = pooled_sd(frag_clean, bg_clean)
    cohen_d = (mean_frag - mean_bg) / psd if psd and not np.isnan(psd) and psd > 0 else np.nan

    return {
        "filter": filter_name,
        "filter_scope": filter_scope(filter_name),
        "n_positions": int(len(sub)),
        "n_fragment_positions": int(len(frag_clean)),
        "n_background_positions": int(len(bg_clean)),
        "n_fragment_headers": int(sub.loc[sub["analysis_is_fragment"], "header"].nunique()),
        "n_background_headers": int(sub.loc[~sub["analysis_is_fragment"], "header"].nunique()),
        "fragment_esm_entropy_mean": mean_frag,
        "background_esm_entropy_mean": mean_bg,
        "delta_mean_fragment_minus_background": mean_frag - mean_bg,
        "fragment_esm_entropy_median": median_frag,
        "background_esm_entropy_median": median_bg,
        "delta_median_fragment_minus_background": median_frag - median_bg,
        "cohen_d_fragment_minus_background": cohen_d,
        "welch_ttest_pvalue_full_positions": welch_ttest_pvalue(frag_clean, bg_clean),
        "interpretation_if_delta_negative": "fragment has lower ESM2 entropy than background",
    }


def per_record_delta_for_filter(sub: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    stats = (
        sub.groupby(["header", "analysis_is_fragment"], sort=False)["esm_entropy"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    stats["group"] = np.where(stats["analysis_is_fragment"], "fragment", "background")
    pivot = stats.pivot(index="header", columns="group", values=["count", "mean", "median"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    for c in [
        "count_fragment", "count_background",
        "mean_fragment", "mean_background",
        "median_fragment", "median_background",
    ]:
        if c not in pivot.columns:
            pivot[c] = np.nan

    meta = sub.groupby("header", sort=False)[["pfam_id", "uid"]].first().reset_index()
    out = pivot.merge(meta, on="header", how="left")
    out = out[
        (out["count_fragment"] >= MIN_FRAGMENT_POSITIONS_PER_RECORD)
        & (out["count_background"] >= MIN_BACKGROUND_POSITIONS_PER_RECORD)
    ].copy()

    out["filter"] = filter_name
    out["filter_scope"] = filter_scope(filter_name)
    out["delta_mean_fragment_minus_background"] = out["mean_fragment"] - out["mean_background"]
    out["delta_median_fragment_minus_background"] = out["median_fragment"] - out["median_background"]

    cols = [
        "filter", "filter_scope", "header", "pfam_id", "uid",
        "count_fragment", "count_background",
        "mean_fragment", "mean_background", "delta_mean_fragment_minus_background",
        "median_fragment", "median_background", "delta_median_fragment_minus_background",
    ]
    return out[cols]


def summarize_per_record_delta(delta: pd.DataFrame, filter_name: str) -> Dict[str, object]:
    row: Dict[str, object] = {
        "filter": filter_name,
        "filter_scope": filter_scope(filter_name),
        "n_records_with_both_groups": int(len(delta)),
    }
    for col in ["delta_mean_fragment_minus_background", "delta_median_fragment_minus_background"]:
        x = pd.to_numeric(delta[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        prefix = col.replace("delta_", "per_record_delta_")
        row.update(summary_num(x, prefix))
        row[f"{prefix}_negative_fraction"] = float((x < 0).mean()) if len(x) else np.nan
        row[f"{prefix}_positive_fraction"] = float((x > 0).mean()) if len(x) else np.nan
    return row


def filter_retention_row(df: pd.DataFrame, mask: pd.Series, filter_name: str) -> Dict[str, object]:
    kept = df[mask]
    removed = df[~mask]
    return {
        "filter": filter_name,
        "filter_scope": filter_scope(filter_name),
        "input_positions": int(len(df)),
        "kept_positions": int(len(kept)),
        "removed_positions": int(len(removed)),
        "kept_fraction": float(len(kept) / len(df)) if len(df) else np.nan,
        "removed_fraction": float(len(removed) / len(df)) if len(df) else np.nan,
        "kept_fragment_positions": int(kept["analysis_is_fragment"].sum()),
        "kept_background_positions": int((~kept["analysis_is_fragment"]).sum()),
        "removed_fragment_positions": int(removed["analysis_is_fragment"].sum()),
        "removed_background_positions": int((~removed["analysis_is_fragment"]).sum()),
        "kept_headers": int(kept["header"].nunique()) if len(kept) else 0,
        "removed_headers": int(removed["header"].nunique()) if len(removed) else 0,
    }


# Region definitions are intentionally simple and interpretable for reporting.
# They are not exclusive: for example, low_tail_le_1p0 and very_low_le_0p5 overlap.
ENTROPY_REGION_DEFINITIONS = [
    ("very_low_le_0p5", None, 0.5, True, "ESM2 entropy <= 0.5"),
    ("low_tail_le_1p0", None, 1.0, True, "ESM2 entropy <= 1.0"),
    ("low_mid_1p0_1p5", 1.0, 1.5, True, "1.0 < ESM2 entropy <= 1.5"),
    ("middle_1p5_2p0", 1.5, 2.0, True, "1.5 < ESM2 entropy <= 2.0"),
    ("middle_2p0_2p5", 2.0, 2.5, True, "2.0 < ESM2 entropy <= 2.5"),
    ("intermediate_1p5_2p5", 1.5, 2.5, True, "1.5 < ESM2 entropy <= 2.5"),
    ("high_tail_gt_2p5", 2.5, None, False, "ESM2 entropy > 2.5"),
    ("very_high_gt_2p8", 2.8, None, False, "ESM2 entropy > 2.8"),
]


def numeric_clean_array(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return x.to_numpy(dtype=float, copy=False)


def quantile_or_nan(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))


def region_mask(values: np.ndarray, lower: Optional[float], upper: Optional[float], upper_closed: bool) -> np.ndarray:
    """Return boolean mask for a named entropy region.

    If lower is provided, the lower bound is always open: values > lower.
    If upper is provided, upper_closed controls whether values <= upper or values < upper.
    """
    mask = np.ones(values.shape[0], dtype=bool)
    if lower is not None:
        mask &= values > float(lower)
    if upper is not None:
        if upper_closed:
            mask &= values <= float(upper)
        else:
            mask &= values < float(upper)
    return mask


def cdf_difference_summary(fragment_values: np.ndarray, background_values: np.ndarray) -> Dict[str, object]:
    """Compute exact ECDF difference summary for two one-dimensional samples."""
    if fragment_values.size == 0 or background_values.size == 0:
        return {
            "cdf_max_abs_diff": np.nan,
            "cdf_max_abs_diff_entropy": np.nan,
            "cdf_signed_diff_at_max_abs_fragment_minus_background": np.nan,
            "cdf_d_plus_fragment_minus_background_max": np.nan,
            "cdf_d_plus_entropy": np.nan,
            "cdf_d_minus_background_minus_fragment_max": np.nan,
            "cdf_d_minus_entropy": np.nan,
            "cdf_direction_at_max_abs": "NA",
        }

    x = np.sort(fragment_values)
    y = np.sort(background_values)
    grid = np.sort(np.unique(np.concatenate([x, y])))
    fx = np.searchsorted(x, grid, side="right") / x.size
    fy = np.searchsorted(y, grid, side="right") / y.size
    diff = fx - fy

    idx_abs = int(np.argmax(np.abs(diff)))
    idx_plus = int(np.argmax(diff))
    idx_minus = int(np.argmax(-diff))
    signed = float(diff[idx_abs])
    if signed > 0:
        direction = "fragment_cdf_higher_at_threshold_fragment_more_left_up_to_this_entropy"
    elif signed < 0:
        direction = "background_cdf_higher_at_threshold_background_more_left_up_to_this_entropy"
    else:
        direction = "no_direction"

    return {
        "cdf_max_abs_diff": float(abs(diff[idx_abs])),
        "cdf_max_abs_diff_entropy": float(grid[idx_abs]),
        "cdf_signed_diff_at_max_abs_fragment_minus_background": signed,
        "cdf_d_plus_fragment_minus_background_max": float(diff[idx_plus]),
        "cdf_d_plus_entropy": float(grid[idx_plus]),
        "cdf_d_minus_background_minus_fragment_max": float(-diff[idx_minus]),
        "cdf_d_minus_entropy": float(grid[idx_minus]),
        "cdf_direction_at_max_abs": direction,
    }


def histogram_probability_distances(fragment_values: np.ndarray, background_values: np.ndarray) -> Dict[str, object]:
    """Compare histogram probability masses using a shared binning scheme."""
    if fragment_values.size == 0 or background_values.size == 0:
        return {
            "histogram_total_variation_distance": np.nan,
            "histogram_overlap_coefficient": np.nan,
            "histogram_l1_distance": np.nan,
            "histogram_js_divergence_natural_log": np.nan,
        }

    both = np.concatenate([fragment_values, background_values])
    vmin, vmax = float(np.min(both)), float(np.max(both))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return {
            "histogram_total_variation_distance": np.nan,
            "histogram_overlap_coefficient": np.nan,
            "histogram_l1_distance": np.nan,
            "histogram_js_divergence_natural_log": np.nan,
        }

    bins = np.linspace(vmin, vmax, HIST_BINS + 1)
    cf, _ = np.histogram(fragment_values, bins=bins)
    cb, _ = np.histogram(background_values, bins=bins)
    pf = cf.astype(float) / cf.sum() if cf.sum() > 0 else np.zeros_like(cf, dtype=float)
    pb = cb.astype(float) / cb.sum() if cb.sum() > 0 else np.zeros_like(cb, dtype=float)

    l1 = float(np.sum(np.abs(pf - pb)))
    tvd = 0.5 * l1
    overlap = float(np.sum(np.minimum(pf, pb)))

    m = 0.5 * (pf + pb)
    js = 0.0
    valid_f = pf > 0
    valid_b = pb > 0
    js += 0.5 * float(np.sum(pf[valid_f] * np.log(pf[valid_f] / m[valid_f])))
    js += 0.5 * float(np.sum(pb[valid_b] * np.log(pb[valid_b] / m[valid_b])))

    return {
        "histogram_total_variation_distance": float(tvd),
        "histogram_overlap_coefficient": float(overlap),
        "histogram_l1_distance": float(l1),
        "histogram_js_divergence_natural_log": float(js),
    }


def ks_test_summary(fragment_values: np.ndarray, background_values: np.ndarray) -> Dict[str, object]:
    if scipy_stats is None or fragment_values.size < 2 or background_values.size < 2:
        return {"ks_statistic": np.nan, "ks_pvalue": np.nan}
    try:
        res = scipy_stats.ks_2samp(fragment_values, background_values, alternative="two-sided", method="asymp")
    except TypeError:
        res = scipy_stats.ks_2samp(fragment_values, background_values, alternative="two-sided")
    return {"ks_statistic": float(res.statistic), "ks_pvalue": float(res.pvalue)}


def distribution_shape_summary_for_filter(sub: pd.DataFrame, filter_name: str) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Summarize full distribution-shape differences beyond mean/median.

    This table is designed to support observations from density plots, especially:
    - depletion/enrichment in low, intermediate, and high entropy regions;
    - high-entropy tail differences;
    - maximum ECDF/CDF difference between fragment and background;
    - KS statistic and histogram overlap/distance.
    """
    frag = numeric_clean_array(sub.loc[sub["analysis_is_fragment"], "esm_entropy"])
    bg = numeric_clean_array(sub.loc[~sub["analysis_is_fragment"], "esm_entropy"])
    scope = filter_scope(filter_name)

    row: Dict[str, object] = {
        "filter": filter_name,
        "filter_scope": scope,
        "n_fragment_positions": int(frag.size),
        "n_background_positions": int(bg.size),
    }

    for group_name, values in [("fragment", frag), ("background", bg)]:
        row[f"{group_name}_q01"] = quantile_or_nan(values, 0.01)
        row[f"{group_name}_q05"] = quantile_or_nan(values, 0.05)
        row[f"{group_name}_q10"] = quantile_or_nan(values, 0.10)
        row[f"{group_name}_q25"] = quantile_or_nan(values, 0.25)
        row[f"{group_name}_q50"] = quantile_or_nan(values, 0.50)
        row[f"{group_name}_q75"] = quantile_or_nan(values, 0.75)
        row[f"{group_name}_q90"] = quantile_or_nan(values, 0.90)
        row[f"{group_name}_q95"] = quantile_or_nan(values, 0.95)
        row[f"{group_name}_q99"] = quantile_or_nan(values, 0.99)

    for q in ["q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"]:
        row[f"delta_{q}_fragment_minus_background"] = row.get(f"fragment_{q}", np.nan) - row.get(f"background_{q}", np.nan)

    region_rows: List[Dict[str, object]] = []
    region_fraction_lookup: Dict[str, Tuple[float, float]] = {}
    for label, lower, upper, upper_closed, description in ENTROPY_REGION_DEFINITIONS:
        frag_count = int(region_mask(frag, lower, upper, upper_closed).sum()) if frag.size else 0
        bg_count = int(region_mask(bg, lower, upper, upper_closed).sum()) if bg.size else 0
        frag_frac = float(frag_count / frag.size) if frag.size else np.nan
        bg_frac = float(bg_count / bg.size) if bg.size else np.nan
        delta_frac = frag_frac - bg_frac if np.isfinite(frag_frac) and np.isfinite(bg_frac) else np.nan
        ratio_frac = frag_frac / bg_frac if np.isfinite(frag_frac) and np.isfinite(bg_frac) and bg_frac > 0 else np.nan
        region_fraction_lookup[label] = (frag_frac, bg_frac)
        region_rows.append({
            "filter": filter_name,
            "filter_scope": scope,
            "region": label,
            "region_description": description,
            "n_fragment_positions": int(frag.size),
            "n_background_positions": int(bg.size),
            "fragment_count_in_region": frag_count,
            "background_count_in_region": bg_count,
            "fragment_fraction_in_region": frag_frac,
            "background_fraction_in_region": bg_frac,
            "delta_fraction_fragment_minus_background": delta_frac,
            "ratio_fragment_fraction_over_background_fraction": ratio_frac,
        })

    # Put the most useful region fractions directly into the wide summary as well.
    for label, (frag_frac, bg_frac) in region_fraction_lookup.items():
        row[f"fragment_frac_{label}"] = frag_frac
        row[f"background_frac_{label}"] = bg_frac
        row[f"delta_frac_{label}_fragment_minus_background"] = frag_frac - bg_frac if np.isfinite(frag_frac) and np.isfinite(bg_frac) else np.nan

    row.update(cdf_difference_summary(frag, bg))
    row.update(ks_test_summary(frag, bg))
    row.update(histogram_probability_distances(frag, bg))

    return row, region_rows


def run_fragment_esm2_analysis(merged: pd.DataFrame) -> None:
    df = merged[merged["analysis_ok"]].copy()
    print(f"[INFO] analysis positions after MSA quality filter: {len(df):,}")
    print(f"[INFO] fragment positions before MSA-conserved removal: {int(df['analysis_is_fragment'].sum()):,}")
    print(f"[INFO] background positions before MSA-conserved removal: {int((~df['analysis_is_fragment']).sum()):,}")

    masks = build_filter_masks(df)

    retention_rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []
    contrast_rows: List[Dict[str, object]] = []
    distribution_shape_rows: List[Dict[str, object]] = []
    entropy_region_rows: List[Dict[str, object]] = []
    delta_tables: List[pd.DataFrame] = []
    delta_summary_rows: List[Dict[str, object]] = []

    for filter_name, mask in masks.items():
        print(f"[INFO] Running filter: {filter_name}")
        mask = mask.fillna(False)
        sub = df[mask].copy()

        retention_rows.append(filter_retention_row(df, mask, filter_name))
        group_rows.extend(group_summary_for_filter(sub, filter_name))
        contrast_rows.append(contrast_summary_for_filter(sub, filter_name))
        distribution_row, region_rows = distribution_shape_summary_for_filter(sub, filter_name)
        distribution_shape_rows.append(distribution_row)
        entropy_region_rows.extend(region_rows)

        delta = per_record_delta_for_filter(sub, filter_name)
        delta_tables.append(delta)
        delta_summary_rows.append(summarize_per_record_delta(delta, filter_name))

    retention = pd.DataFrame(retention_rows)
    group_summary = pd.DataFrame(group_rows)
    contrast_summary = pd.DataFrame(contrast_rows)
    distribution_shape_summary = pd.DataFrame(distribution_shape_rows)
    entropy_region_fraction_summary = pd.DataFrame(entropy_region_rows)
    per_record_delta = pd.concat(delta_tables, ignore_index=True) if delta_tables else pd.DataFrame()
    per_record_delta_summary = pd.DataFrame(delta_summary_rows)

    retention.to_csv(FILTER_RETENTION_TSV, sep="\t", index=False)
    group_summary.to_csv(GROUP_SUMMARY_TSV, sep="\t", index=False)
    contrast_summary.to_csv(CONTRAST_SUMMARY_TSV, sep="\t", index=False)
    distribution_shape_summary.to_csv(DISTRIBUTION_SHAPE_SUMMARY_TSV, sep="\t", index=False)
    entropy_region_fraction_summary.to_csv(ENTROPY_REGION_FRACTION_TSV, sep="\t", index=False)
    per_record_delta.to_csv(PER_RECORD_DELTA_TSV, sep="\t", index=False)
    per_record_delta_summary.to_csv(PER_RECORD_DELTA_SUMMARY_TSV, sep="\t", index=False)

    # Convenience split outputs: per-record percentile filters vs global percentile filters.
    retention[retention["filter_scope"].isin(["raw", "absolute_conservation", "per_record_percentile"])].to_csv(
        FILTER_RETENTION_PER_RECORD_TSV, sep="\t", index=False
    )
    retention[retention["filter_scope"].isin(["raw", "absolute_conservation", "global_percentile"])].to_csv(
        FILTER_RETENTION_GLOBAL_TSV, sep="\t", index=False
    )
    contrast_summary[contrast_summary["filter_scope"].isin(["raw", "absolute_conservation", "per_record_percentile"])].to_csv(
        CONTRAST_SUMMARY_PER_RECORD_TSV, sep="\t", index=False
    )
    contrast_summary[contrast_summary["filter_scope"].isin(["raw", "absolute_conservation", "global_percentile"])].to_csv(
        CONTRAST_SUMMARY_GLOBAL_TSV, sep="\t", index=False
    )
    per_record_delta_summary[per_record_delta_summary["filter_scope"].isin(["raw", "absolute_conservation", "per_record_percentile"])].to_csv(
        PER_RECORD_DELTA_SUMMARY_PER_RECORD_TSV, sep="\t", index=False
    )
    per_record_delta_summary[per_record_delta_summary["filter_scope"].isin(["raw", "absolute_conservation", "global_percentile"])].to_csv(
        PER_RECORD_DELTA_SUMMARY_GLOBAL_TSV, sep="\t", index=False
    )

    distribution_shape_summary[distribution_shape_summary["filter_scope"].isin(["raw", "absolute_conservation", "per_record_percentile"])].to_csv(
        DISTRIBUTION_SHAPE_SUMMARY_PER_RECORD_TSV, sep="\t", index=False
    )
    distribution_shape_summary[distribution_shape_summary["filter_scope"].isin(["raw", "absolute_conservation", "global_percentile"])].to_csv(
        DISTRIBUTION_SHAPE_SUMMARY_GLOBAL_TSV, sep="\t", index=False
    )
    distribution_shape_summary[distribution_shape_summary["filter_scope"] == "absolute_conservation"].to_csv(
        DISTRIBUTION_SHAPE_SUMMARY_ABSOLUTE_TSV, sep="\t", index=False
    )

    entropy_region_fraction_summary[entropy_region_fraction_summary["filter_scope"].isin(["raw", "absolute_conservation", "per_record_percentile"])].to_csv(
        ENTROPY_REGION_FRACTION_PER_RECORD_TSV, sep="\t", index=False
    )
    entropy_region_fraction_summary[entropy_region_fraction_summary["filter_scope"].isin(["raw", "absolute_conservation", "global_percentile"])].to_csv(
        ENTROPY_REGION_FRACTION_GLOBAL_TSV, sep="\t", index=False
    )
    entropy_region_fraction_summary[entropy_region_fraction_summary["filter_scope"] == "absolute_conservation"].to_csv(
        ENTROPY_REGION_FRACTION_ABSOLUTE_TSV, sep="\t", index=False
    )

    print("\n[CONTRAST SUMMARY: fragment - background ESM2 entropy]")
    keep_cols = [
        "filter",
        "filter_scope",
        "n_fragment_positions",
        "n_background_positions",
        "fragment_esm_entropy_mean",
        "background_esm_entropy_mean",
        "delta_mean_fragment_minus_background",
        "fragment_esm_entropy_median",
        "background_esm_entropy_median",
        "delta_median_fragment_minus_background",
        "cohen_d_fragment_minus_background",
    ]
    print(contrast_summary[keep_cols].to_string(index=False))

    print("\n[PER-RECORD DELTA SUMMARY]")
    print(per_record_delta_summary.to_string(index=False))

    if MAKE_PLOTS:
        save_plots(df, masks, contrast_summary, per_record_delta_summary)


def safe_filename_part(x: str) -> str:
    """Make filter names safe for file names."""
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "unnamed"


def save_position_histograms_for_filter(
    sub: pd.DataFrame,
    filter_name: str,
    hist_rows: List[Dict[str, object]],
) -> None:
    """Save position-level ESM2 entropy histograms for one filter into its categorized folder."""
    frag = pd.to_numeric(
        sub.loc[sub["analysis_is_fragment"], "esm_entropy"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()
    bg = pd.to_numeric(
        sub.loc[~sub["analysis_is_fragment"], "esm_entropy"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if frag.empty or bg.empty:
        print(
            f"[WARN] Skip histogram for {filter_name}: "
            f"fragment_n={len(frag):,}, background_n={len(bg):,}"
        )
        return

    both = pd.concat([frag, bg], ignore_index=True)
    vmin = float(both.min())
    vmax = float(both.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        print(f"[WARN] Skip histogram for {filter_name}: invalid entropy range {vmin}..{vmax}")
        return

    bins = np.linspace(vmin, vmax, HIST_BINS + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    fname = safe_filename_part(filter_name)
    scope = filter_scope(filter_name)

    # Raw filter is written only under the all-filter histogram folder.
    out_dir = scope_hist_dir(scope)
    out_dir.mkdir(parents=True, exist_ok=True)

    for group_name, values in [("fragment", frag), ("background", bg)]:
        counts, _ = np.histogram(values.to_numpy(), bins=bins)
        total = int(counts.sum())
        density = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
        for i in range(len(centers)):
            hist_rows.append({
                "filter": filter_name,
                "filter_scope": scope,
                "group": group_name,
                "bin_left": float(bins[i]),
                "bin_right": float(bins[i + 1]),
                "bin_center": float(centers[i]),
                "count": int(counts[i]),
                "density": float(density[i]),
                "total": total,
            })

    plt.figure(figsize=(12, 7))
    plt.hist(frag.to_numpy(), bins=bins, alpha=0.55, density=False, label=f"Fragment, n={len(frag):,}")
    plt.hist(bg.to_numpy(), bins=bins, alpha=0.55, density=False, label=f"Background, n={len(bg):,}")
    plt.xlabel("ESM2 entropy")
    plt.ylabel("Count")
    plt.title(f"ESM2 entropy: complete fragment positions vs background positions (count)\n{filter_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_esm2_entropy_complete_fragment_vs_background__{fname}__count.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.hist(frag.to_numpy(), bins=bins, alpha=0.55, density=True, label=f"Fragment, n={len(frag):,}")
    plt.hist(bg.to_numpy(), bins=bins, alpha=0.55, density=True, label=f"Background, n={len(bg):,}")
    plt.xlabel("ESM2 entropy")
    plt.ylabel("Density")
    plt.title(f"ESM2 entropy: complete fragment positions vs background positions (density)\n{filter_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_esm2_entropy_complete_fragment_vs_background__{fname}__density.png", dpi=200)
    plt.close()


def plot_subset_contrast(contrast_df: pd.DataFrame, out_dir: Path, stem_suffix: str, title_suffix: str) -> None:
    """Save mean-entropy and delta figures for a selected set of filters."""
    if contrast_df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(contrast_df))
    width = 0.38
    labels = contrast_df["filter"].astype(str).tolist()

    plt.figure(figsize=(max(9, 1.8 * len(labels) + 3), 6))
    plt.bar(x - width / 2, contrast_df["fragment_esm_entropy_mean"], width, label="Fragment")
    plt.bar(x + width / 2, contrast_df["background_esm_entropy_mean"], width, label="Background")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean ESM2 entropy")
    plt.title(f"Fragment vs background ESM2 entropy ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"mean_esm2_entropy_fragment_vs_background.{stem_suffix}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(max(9, 1.8 * len(labels) + 3), 6))
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.bar(x, contrast_df["delta_mean_fragment_minus_background"])
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean ESM2 entropy delta: fragment - background")
    plt.title(f"Position-level ESM2 entropy difference ({title_suffix})")
    plt.tight_layout()
    plt.savefig(out_dir / f"delta_mean_esm2_entropy_fragment_minus_background.{stem_suffix}.png", dpi=200)
    plt.close()


def plot_subset_per_record(delta_summary_df: pd.DataFrame, out_dir: Path, stem_suffix: str, title_suffix: str) -> None:
    """Save per-record median-delta figure for a selected set of filters."""
    if delta_summary_df.empty:
        return
    ycol = "per_record_delta_mean_fragment_minus_background_median"
    if ycol not in delta_summary_df.columns:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = delta_summary_df["filter"].astype(str).tolist()
    x = np.arange(len(delta_summary_df))
    plt.figure(figsize=(max(9, 1.8 * len(labels) + 3), 6))
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.bar(x, delta_summary_df[ycol])
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Median per-record mean delta: fragment - background")
    plt.title(f"Per-record ESM2 entropy difference ({title_suffix})")
    plt.tight_layout()
    plt.savefig(out_dir / f"per_record_median_delta_esm2_entropy.{stem_suffix}.png", dpi=200)
    plt.close()


def save_plots(
    df: pd.DataFrame,
    masks: Dict[str, pd.Series],
    contrast_summary: pd.DataFrame,
    per_record_delta_summary: pd.DataFrame,
) -> None:
    if contrast_summary.empty:
        return

    # Combined overview, useful for quick inspection but not the primary comparison.
    plot_subset_contrast(contrast_summary, ALL_FILTERS_FIG_DIR, "all_filters", "all filters")
    plot_subset_per_record(per_record_delta_summary, ALL_FILTERS_FIG_DIR, "all_filters", "all filters")

    # Main result: per-record lowest 0%, 5%, 10%, 20%, 50% only.
    main_contrast = contrast_summary[contrast_summary["filter_scope"] == "per_record_percentile"].copy()
    main_delta = per_record_delta_summary[per_record_delta_summary["filter_scope"] == "per_record_percentile"].copy()
    plot_subset_contrast(main_contrast, MAIN_FIG_DIR, "per_record_percentile_filters", "main result: per-record lowest 0/5/10/20/50%")
    plot_subset_per_record(main_delta, MAIN_FIG_DIR, "per_record_percentile_filters", "main result: per-record lowest 0/5/10/20/50%")

    # Sensitivity check: global lowest 0%, 5%, 10%, 20%, 50% only.
    global_contrast = contrast_summary[contrast_summary["filter_scope"] == "global_percentile"].copy()
    global_delta = per_record_delta_summary[per_record_delta_summary["filter_scope"] == "global_percentile"].copy()
    plot_subset_contrast(global_contrast, GLOBAL_FIG_DIR, "global_percentile_filters", "sensitivity check: global lowest 0/5/10/20/50%")
    plot_subset_per_record(global_delta, GLOBAL_FIG_DIR, "global_percentile_filters", "sensitivity check: global lowest 0/5/10/20/50%")

    # Auxiliary absolute conservation filters.
    abs_contrast = contrast_summary[contrast_summary["filter_scope"] == "absolute_conservation"].copy()
    abs_delta = per_record_delta_summary[per_record_delta_summary["filter_scope"] == "absolute_conservation"].copy()
    plot_subset_contrast(abs_contrast, ABS_FIG_DIR, "absolute_conservation_filters", "absolute conservation filters")
    plot_subset_per_record(abs_delta, ABS_FIG_DIR, "absolute_conservation_filters", "absolute conservation filters")

    # Filter-specific histograms.
    hist_rows: List[Dict[str, object]] = []
    for filter_name, mask in masks.items():
        mask = mask.fillna(False)
        sub = df[mask].copy()
        save_position_histograms_for_filter(sub, filter_name, hist_rows)

    if hist_rows:
        hist_df = pd.DataFrame(hist_rows)
        hist_df.to_csv(HISTOGRAM_COUNTS_TSV, sep="	", index=False)
        hist_df[hist_df["filter_scope"] == "per_record_percentile"].to_csv(HISTOGRAM_COUNTS_PER_RECORD_TSV, sep="	", index=False)
        hist_df[hist_df["filter_scope"] == "global_percentile"].to_csv(HISTOGRAM_COUNTS_GLOBAL_TSV, sep="	", index=False)
        hist_df[hist_df["filter_scope"] == "absolute_conservation"].to_csv(HISTOGRAM_COUNTS_ABSOLUTE_TSV, sep="	", index=False)
        print(f"[INFO] Wrote histogram count table: {HISTOGRAM_COUNTS_TSV}")


# ==============================================================================
# Cleanup and main
# ==============================================================================
def cleanup() -> None:
    if OVERWRITE_OUTPUT and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    ensure_output_dirs()
    write_output_structure()

def main() -> None:
    args = build_arg_parser().parse_args()
    configure_runtime(args)
    cleanup()

    print("=" * 100)
    print("Fragment vs background ESM2 entropy after removing highly conserved MSA positions")
    print(f"REPRESENTATIVE_FILE: {REPRESENTATIVE_FILE}")
    print(f"ESM_ENTROPY_TSV: {ESM_ENTROPY_TSV}")
    print(f"COMPLETE_FRAGMENT_FASTA: {COMPLETE_FRAGMENT_FASTA}")
    print(f"PFAM_ROOT: {PFAM_ROOT}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"USE_PRECOMPUTED_MSA_IF_AVAILABLE: {USE_PRECOMPUTED_MSA_IF_AVAILABLE}")
    print(f"PRECOMPUTED_MSA_TSV: {PRECOMPUTED_MSA_TSV}")
    print(f"MSA_LOW_ENTROPY_FRACTIONS (per-record): {MSA_LOW_ENTROPY_FRACTIONS}")
    print(f"GLOBAL_MSA_LOW_ENTROPY_FRACTIONS: {GLOBAL_MSA_LOW_ENTROPY_FRACTIONS}")
    print(f"PERCENTILE_TIE_BREAK_SEED: {PERCENTILE_TIE_BREAK_SEED}")
    print(f"PERCENTILE_REMOVAL_ROUNDING: {PERCENTILE_REMOVAL_ROUNDING}")
    print(f"MAJORITY_FREQ_CUTOFF: {MAJORITY_FREQ_CUTOFF}")
    print("=" * 100)

    records, header_to_seq, key_to_headers, header_to_len = load_representatives()
    msa_path = select_msa_source(records)
    coverage = assign_complete_fragments(header_to_seq, key_to_headers, header_to_len)
    merged = merge_all(msa_path, coverage)
    run_fragment_esm2_analysis(merged)

    print("\n[DONE]")
    print(f"Main outputs written under: {OUT_DIR}")
    print(f"- Main result tables: {MAIN_TABLE_DIR}")
    print(f"- Main result figures: {MAIN_FIG_DIR}")
    print(f"- Sensitivity-check tables: {GLOBAL_TABLE_DIR}")
    print(f"- Sensitivity-check figures: {GLOBAL_FIG_DIR}")
    print(f"- Absolute-filter tables: {ABS_TABLE_DIR}")
    print(f"- Absolute-filter figures: {ABS_FIG_DIR}")
    print(f"- All-filter tables: {ALL_FILTERS_TABLE_DIR}")
    print(f"- All-filter figures: {ALL_FILTERS_FIG_DIR}")
    print(f"- Distribution-shape summary: {DISTRIBUTION_SHAPE_SUMMARY_TSV}")
    print(f"- Entropy-region fraction summary: {ENTROPY_REGION_FRACTION_TSV}")
    print(f"- Output structure note: {OUTPUT_STRUCTURE_TXT}")



if __name__ == "__main__":
    main()
