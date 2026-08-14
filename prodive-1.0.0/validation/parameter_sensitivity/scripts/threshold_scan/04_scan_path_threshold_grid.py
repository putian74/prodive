#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import glob
import shutil
import re
from collections import defaultdict
from itertools import product
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# ==============================================================================
# 1. 
# ==============================================================================

#  NPZ 
# ：
#   F6p0_baseS8p0/
#   F5p5_baseS8p0/
#   ...
#   F1p0_baseS8p0/
PREPROCESSED_ROOT = None

# 
OUTPUT_ROOT = None

# 
PFAM_MAPPING_FILE = None

# coverage 
#  family ：
#   /path/to/PfamA_seed/PF00001/PF00001_coverage.csv
PFAM_SEED_DIR = None

#  HMM
USE_HMM_FILTER_LIST = False
ALLOWED_LIST_FILE = ""

# ，
FIRST_THRESHOLDS = [
    6.0, 5.5, 5.0, 4.5, 4.0, 3.5,
    3.0, 2.5, 2.0, 1.5, 1.0
]

# ， path  linear_vals
SECOND_THRESHOLDS = [
    8.0, 8.5, 9.0, 9.5, 10.0, 10.5,
    11.0, 11.5, 12.0, 12.5, 13.0,
    13.5, 14.0, 14.5, 15.0
]



#  NPZ 
NPZ_GLOB_PATTERN = "kl_*_sparse.npz"

# path 
MIN_SUB_SEGMENT_LENGTH = 5
TOP_N_RESULTS_TO_SAVE = None

FRAGMENT_SIZE = 6
MAX_JUMP_DISTANCE = FRAGMENT_SIZE - 1
LENGTH_EXTENSION_AMOUNT = FRAGMENT_SIZE - 1

# coverage rescoring 
DEFAULT_MISSING_COVERAGE = 0.0
COVERAGE_THRESHOLD = 0.85
SCORE_THRESHOLD = 1.2

#  rescored CSV  Score 
# Preserve production row order by default; enable only for presentation copies.
SORT_FINAL_OUTPUT = False

# Plotting every threshold combination is expensive, so it is disabled by default.
MAKE_SCORE_DISTRIBUTION_PLOT = False

DIST_BINS = 120
PLOT_DPI = 300
PLOT_FIGSIZE = (8.2, 6.0)
PLOT_XMIN = 0.5
PLOT_XMAX = 3.5

# 
NUM_PROCESSES = 20

#  temp csv
KEEP_TEMP_FILES = False

# ，
RESUME = True




def _parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(',') if x.strip()]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan first/second threshold combinations by path building and Sadj rescoring.")
    parser.add_argument("--preprocessed-root", required=True, help="Root containing F*_baseS8p0 preprocessed NPZ directories.")
    parser.add_argument("--output-root", required=True, help="Output root directory.")
    parser.add_argument("--mapping-file", required=True, help="Pfam mapping file.")
    parser.add_argument("--coverage-root", required=True, help="Root containing per-family coverage CSV files.")
    parser.add_argument("--allowed-list-file", default=ALLOWED_LIST_FILE, help="Optional allowed Pfam list file.")
    parser.add_argument("--use-hmm-filter-list", action="store_true", default=USE_HMM_FILTER_LIST, help="Restrict analysis to the allowed-list file.")
    parser.add_argument("--first-thresholds", default=','.join(map(str, FIRST_THRESHOLDS)), help="Comma-separated first thresholds.")
    parser.add_argument("--second-thresholds", default=','.join(map(str, SECOND_THRESHOLDS)), help="Comma-separated second thresholds.")
    parser.add_argument("--fragment-size", type=int, default=FRAGMENT_SIZE, help="Fragment size used for path extension settings.")
    parser.add_argument("--min-sub-segment-length", type=int, default=MIN_SUB_SEGMENT_LENGTH, help="Minimum sub-segment length.")
    parser.add_argument("--coverage-threshold", type=float, default=COVERAGE_THRESHOLD, help="Coverage threshold.")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD, help="Final Sadj score threshold.")
    parser.add_argument("--num-processes", type=int, default=NUM_PROCESSES, help="Worker process count.")
    parser.add_argument("--resume", action="store_true", default=RESUME, help="Skip already completed combinations.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not skip existing completed combinations.")
    return parser.parse_args()

def apply_runtime_args(args: argparse.Namespace) -> None:
    global PREPROCESSED_ROOT, OUTPUT_ROOT, PFAM_MAPPING_FILE, PFAM_SEED_DIR, ALLOWED_LIST_FILE, USE_HMM_FILTER_LIST
    global FIRST_THRESHOLDS, SECOND_THRESHOLDS, FRAGMENT_SIZE, MAX_JUMP_DISTANCE, LENGTH_EXTENSION_AMOUNT
    global MIN_SUB_SEGMENT_LENGTH, COVERAGE_THRESHOLD, SCORE_THRESHOLD, NUM_PROCESSES, RESUME
    PREPROCESSED_ROOT = args.preprocessed_root
    OUTPUT_ROOT = args.output_root
    PFAM_MAPPING_FILE = args.mapping_file
    PFAM_SEED_DIR = args.coverage_root
    ALLOWED_LIST_FILE = args.allowed_list_file
    USE_HMM_FILTER_LIST = bool(args.use_hmm_filter_list)
    FIRST_THRESHOLDS = _parse_float_list(args.first_thresholds)
    SECOND_THRESHOLDS = _parse_float_list(args.second_thresholds)
    FRAGMENT_SIZE = int(args.fragment_size)
    MAX_JUMP_DISTANCE = FRAGMENT_SIZE - 1
    LENGTH_EXTENSION_AMOUNT = FRAGMENT_SIZE - 1
    MIN_SUB_SEGMENT_LENGTH = int(args.min_sub_segment_length)
    COVERAGE_THRESHOLD = float(args.coverage_threshold)
    SCORE_THRESHOLD = float(args.score_threshold)
    NUM_PROCESSES = int(args.num_processes)
    RESUME = bool(args.resume)

# ==============================================================================
# 2. 
# ==============================================================================

global_allowed_set = None
global_temp_csv_dir = None
global_unclean_temp_csv_dir = None
global_gap_temp_csv_dir = None
global_id_to_name = None
global_first_threshold = None
global_second_threshold = None


# ==============================================================================
# 3. 
# ==============================================================================

def tag_float(x):
    return f"{float(x):.1f}".replace(".", "p")


def combo_tag(first_threshold, second_threshold):
    return f"F{tag_float(first_threshold)}_S{tag_float(second_threshold)}"


def first_dir_name(first_threshold):
    return f"F{tag_float(first_threshold)}_baseS8p0"


def normalize_pfam_id(x):
    s = str(x).strip()
    m = re.search(r"PF\d{5}", s)
    if m:
        return m.group(0)
    return s


def load_allowed_list(file_path):
    print(f" HMM : {file_path}")
    allowed_set = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            hmm_id = line.strip()
            if hmm_id:
                allowed_set.add(hmm_id)

    return allowed_set


def load_pfam_mapping(file_path):
    print(f" PF : {file_path}")
    id_to_name = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":", 1)
            if len(parts) != 2:
                continue

            pfam_name = normalize_pfam_id(parts[0])
            pfam_id = int(parts[1].strip())
            id_to_name[pfam_id] = pfam_name

    return id_to_name


def validate_generated_npz(z, filename):
    """
    Validate the generated NPZ schema.

    Required fields:
      i
      first_threshold
      meta
      rows
      cols
      linear_vals

    raw_vals and base_second_threshold are optional compatibility fields.
    """
    required = [
        "i",
        "first_threshold",
        "meta",
        "rows",
        "cols",
        "linear_vals",
    ]

    for key in required:
        if key not in z:
            raise KeyError(f"{filename}:  {key}")


def build_sparse_dict_from_block(rows_block, cols_block, vals_block):
    data = defaultdict(dict)

    for x, y, v in zip(rows_block, cols_block, vals_block):
        data[int(x)][int(y)] = float(v)

    return dict(data)


# ==============================================================================
# 4. path 
# ==============================================================================

def orient_data_for_analysis(data):
    if not data:
        return data, False

    num_main_keys = len(data)
    valid_sub_dicts = [v for v in data.values() if v]

    if not valid_sub_dicts:
        return data, False

    avg_sub_keys = np.mean([len(sub_dict) for sub_dict in valid_sub_dicts])

    if num_main_keys > avg_sub_keys * 2:
        inverted_data = defaultdict(dict)
        for main_key, sub_dict in data.items():
            for sub_key, value in sub_dict.items():
                inverted_data[sub_key][main_key] = value
        return dict(inverted_data), True

    return data, False


def invert_dictionary(data):
    inverted_data = defaultdict(dict)

    for main_key, sub_dict in data.items():
        for sub_key, value in sub_dict.items():
            inverted_data[sub_key][main_key] = value

    return dict(inverted_data)


def is_segment_clean(path, data):
    for x, y in path:
        sub_keys_at_x = data.get(x, {})
        if (y + 1) in sub_keys_at_x or (y - 1) in sub_keys_at_x:
            return False
    return True


def _get_path_key(path_list, hmm1, hmm2):
    xs_coords = tuple(sorted(list(set(p[0] for p in path_list))))
    ys_coords = tuple(sorted(list(set(p[1] for p in path_list))))

    part1 = (hmm1, xs_coords)
    part2 = (hmm2, ys_coords)

    if hmm1 <= hmm2:
        return (part1, part2)
    else:
        return (part2, part1)


def compress_segment_list_merged(segment_list):
    if not segment_list:
        return ""

    raw_intervals = []
    start = segment_list[0]
    end = segment_list[0]

    for val in segment_list[1:]:
        if val == end + 1:
            end = val
        else:
            raw_intervals.append((start, end + LENGTH_EXTENSION_AMOUNT))
            start = val
            end = val

    raw_intervals.append((start, end + LENGTH_EXTENSION_AMOUNT))

    if not raw_intervals:
        return ""

    merged_intervals = []
    current_start, current_end = raw_intervals[0]

    for next_start, next_end in raw_intervals[1:]:
        if next_start <= current_end + 1:
            current_end = max(current_end, next_end)
        else:
            merged_intervals.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged_intervals.append((current_start, current_end))

    return ",".join(f"{s}-{e}" for s, e in merged_intervals)


def compress_segment_list_detailed(segment_list):
    if not segment_list:
        return ""

    ranges = []
    start = segment_list[0]
    end = segment_list[0]

    for val in segment_list[1:]:
        if val == end + 1:
            end = val
        else:
            display_end = end + LENGTH_EXTENSION_AMOUNT
            ranges.append(f"{start}-{display_end}")
            start = val
            end = val

    display_end = end + LENGTH_EXTENSION_AMOUNT
    ranges.append(f"{start}-{display_end}")

    return ",".join(ranges)


def _process_path_as_one_to_one(path, data):
    if not path:
        return None

    x_coords = [p[0] for p in path]
    true_main_segment = sorted(list(set(x_coords)))

    min_x, max_x = min(x_coords), max(x_coords)
    physical_main_len = (max_x - min_x + 1) + LENGTH_EXTENSION_AMOUNT
    is_gapped = (max_x - min_x + 1) > len(x_coords)

    values = [data[p[0]][p[1]] for p in path]
    avg_value = float(np.mean(values)) if values else 0.0

    y_coords = [p[1] for p in path]
    min_y, max_y = min(y_coords), max(y_coords)
    physical_sub_len = (max_y - min_y + 1) + LENGTH_EXTENSION_AMOUNT

    stats = {
        "main_segment_len": physical_main_len,
        "num_sub_segments": 1,
        "avg_sub_segment_len": float(physical_sub_len),
        "structural_fit_iou": 1.0,
        "avg_similarity": f"{avg_value}",
    }

    sub_segment_details = [
        {
            "path": path,
            "avg_value": avg_value,
        }
    ]

    # Coverage is not applied at this stage, so the provisional score is the path mean.
    return {
        "score": avg_value,
        "main_segment": true_main_segment,
        "stats": stats,
        "sub_segments": sub_segment_details,
        "is_gapped": is_gapped,
    }


def find_patterns(data, min_len_threshold):
    if not data:
        return [], []

    points = set()
    successors = defaultdict(list)
    sorted_x = sorted(data.keys())

    for x in sorted_x:
        for y in data[x]:
            points.add((x, y))

    sorted_points = sorted(list(points))

    for point in sorted_points:
        x, y = point
        for delta in range(1, FRAGMENT_SIZE):
            predecessor = (x - delta, y - delta)
            if predecessor in points:
                successors[predecessor].append(point)
                break

    memo = {}

    def trace_longest_path_from(node):
        if node in memo:
            return memo[node]
        if not successors.get(node):
            return [node]
        path = [node] + trace_longest_path_from(successors[node][0])
        memo[node] = path
        return path

    all_raw_paths = []
    has_incoming = set()

    for src, dests in successors.items():
        for dst in dests:
            has_incoming.add(dst)

    start_nodes = sorted([p for p in points if p not in has_incoming])

    for start_node in start_nodes:
        all_raw_paths.append(trace_longest_path_from(start_node))

    structurally_clean_paths = [
        path for path in all_raw_paths
        if is_segment_clean(path, data)
    ]

    path_sets = [
        (path, set(p[0] for p in path))
        for path in structurally_clean_paths
    ]

    path_clusters = []

    while path_sets:
        current_cluster_paths = [path_sets[0][0]]
        main_union = path_sets[0][1]
        path_sets.pop(0)

        i = 0
        while i < len(path_sets):
            path_to_check, set_to_check = path_sets[i]

            intersection = len(main_union.intersection(set_to_check))
            union = len(main_union.union(set_to_check))
            iou = intersection / union if union > 0 else 0

            if iou > 0.1:
                current_cluster_paths.append(path_to_check)
                main_union.update(set_to_check)
                path_sets.pop(i)
            else:
                i += 1

        path_clusters.append(current_cluster_paths)

    clean_results = []
    unclean_results = []

    for path_group in path_clusters:
        if len(path_group) > 1:
            has_y_overlap = False

            for i in range(len(path_group)):
                for j in range(i + 1, len(path_group)):
                    if not {p[1] for p in path_group[i]}.isdisjoint(
                        {p[1] for p in path_group[j]}
                    ):
                        has_y_overlap = True
                        break
                if has_y_overlap:
                    break

            if has_y_overlap:
                relevant_paths = [
                    p for p in path_group
                    if len(p) >= min_len_threshold
                ]

                if not relevant_paths:
                    continue

                main_axis_sets = [
                    set(p[0] for p in path)
                    for path in relevant_paths
                ]

                main_coords_union = set().union(*main_axis_sets)

                if not main_coords_union:
                    continue

                min_x, max_x = min(main_coords_union), max(main_coords_union)
                physical_len = (max_x - min_x + 1) + LENGTH_EXTENSION_AMOUNT

                sub_segment_details = []
                for path in relevant_paths:
                    sub_segment_details.append(
                        {
                            "path": path,
                            "avg_value": "N/A (Overlap)",
                        }
                    )

                stats = {
                    "main_segment_len": physical_len,
                    "num_sub_segments": len(relevant_paths),
                    "avg_sub_segment_len": "N/A",
                    "structural_fit_iou": "N/A",
                    "avg_similarity": "N/A (Overlap)",
                }

                unclean_results.append(
                    {
                        "score": -1,
                        "main_segment": sorted(list(main_coords_union)),
                        "stats": stats,
                        "sub_segments": sub_segment_details,
                    }
                )
            else:
                for path in path_group:
                    if len(path) >= min_len_threshold:
                        result_1to1 = _process_path_as_one_to_one(path, data)
                        if result_1to1:
                            clean_results.append(result_1to1)

        elif len(path_group) == 1:
            path = path_group[0]
            if len(path) >= min_len_threshold:
                result_1to1 = _process_path_as_one_to_one(path, data)
                if result_1to1:
                    clean_results.append(result_1to1)

    return (
        sorted(clean_results, key=lambda x: x["score"], reverse=True),
        unclean_results,
    )


def process_analysis_results(results, file_name, hmm_main, hmm_sub, top_n_to_save,
                             first_threshold, second_threshold):
    formatted_results = []

    for res in results[:top_n_to_save if top_n_to_save is not None else None]:
        is_gapped = res.get("is_gapped", False)
        stats = res["stats"]

        merged_details_list = []

        if is_gapped:
            detailed_details_list = []

            for sub in res["sub_segments"]:
                path = sub["path"]
                main_part_list = [p[0] for p in path]
                sub_part_list = [p[1] for p in path]

                main_str_merged = compress_segment_list_merged(main_part_list)
                sub_str_merged = compress_segment_list_merged(sub_part_list)
                merged_details_list.append(f"{main_str_merged} -> {sub_str_merged}")

                main_str_detailed = compress_segment_list_detailed(main_part_list)
                sub_str_detailed = compress_segment_list_detailed(sub_part_list)

                avg_val = sub["avg_value"]
                suffix = " [OVERLAP]" if isinstance(avg_val, str) and "N/A" in avg_val else ""

                detailed_details_list.append(
                    f"{main_str_detailed} -> {sub_str_detailed}{suffix}"
                )

            merged_sub_str = "; ".join(merged_details_list)
            merged_main_str = compress_segment_list_merged(res["main_segment"])

            detailed_sub_str = "; ".join(detailed_details_list)
            detailed_main_str = compress_segment_list_detailed(res["main_segment"])

        else:
            for sub in res["sub_segments"]:
                path = sub["path"]
                main_part_list = [p[0] for p in path]
                sub_part_list = [p[1] for p in path]

                main_str_merged = compress_segment_list_merged(main_part_list)
                sub_str_merged = compress_segment_list_merged(sub_part_list)
                merged_details_list.append(f"{main_str_merged} -> {sub_str_merged}")

            merged_sub_str = "; ".join(merged_details_list)
            merged_main_str = compress_segment_list_merged(res["main_segment"])
            detailed_sub_str = ""
            detailed_main_str = ""

        record = {
            "First_Threshold": first_threshold,
            "Second_Threshold": second_threshold,
            "File": file_name,
            "Main_HMM": hmm_main,
            "Sub_HMM": hmm_sub,
            "Main_Segment": merged_main_str,
            "Main_Segment_Len": stats["main_segment_len"],
            "Num_Sub_Segments": stats["num_sub_segments"],
            "Avg_Sub_Segment_Len": stats["avg_sub_segment_len"],
            "Structural_Fit_IoU": stats["structural_fit_iou"],
            "Avg_Similarity": stats["avg_similarity"],
            "Sub_Segments_Details": merged_sub_str,
            "_Main_Segment_Detailed": detailed_main_str,
            "_Sub_Segments_Details_Detailed": detailed_sub_str,
        }

        if is_gapped:
            record["_is_gapped"] = True

        formatted_results.append(record)

    return formatted_results


# ==============================================================================
# 5. worker  NPZ 
# ==============================================================================

def init_worker(allowed_set_from_main, temp_dir_from_main, unclean_temp_dir_from_main,
                gap_temp_dir_from_main, id_to_name_from_main,
                first_threshold_from_main, second_threshold_from_main):
    global global_allowed_set
    global global_temp_csv_dir
    global global_unclean_temp_csv_dir
    global global_gap_temp_csv_dir
    global global_id_to_name
    global global_first_threshold
    global global_second_threshold

    global_allowed_set = allowed_set_from_main
    global_temp_csv_dir = temp_dir_from_main
    global_unclean_temp_csv_dir = unclean_temp_dir_from_main
    global_gap_temp_csv_dir = gap_temp_dir_from_main
    global_id_to_name = id_to_name_from_main
    global_first_threshold = first_threshold_from_main
    global_second_threshold = second_threshold_from_main


def analyze_npz_file(file_path):
    global global_allowed_set
    global global_temp_csv_dir
    global global_unclean_temp_csv_dir
    global global_gap_temp_csv_dir
    global global_id_to_name
    global global_first_threshold
    global global_second_threshold

    file_name = os.path.basename(file_path)

    final_clean_results = []
    final_unclean_results = []
    final_gap_results = []

    total_pairs_after_second = 0
    total_points_after_second = 0

    try:
        with np.load(file_path, allow_pickle=False) as z:
            validate_generated_npz(z, file_name)

            query_id_arr = z["i"]
            meta = z["meta"]
            rows = z["rows"]
            cols = z["cols"]
            linear_vals = z["linear_vals"]

        if query_id_arr.size != 1:
            return (file_name, 0, 0, 0, 0, 0)

        query_id = int(query_id_arr[0])
        query_hmm_id = global_id_to_name.get(query_id, f"ID_{query_id}")

        if meta.ndim != 2 or meta.shape[1] != 5:
            return (file_name, 0, 0, 0, 0, 0)

        for rec_idx in range(meta.shape[0]):
            target_id, nrows, ncols, offset, nnz = meta[rec_idx]

            target_id = int(target_id)
            offset = int(offset)
            nnz = int(nnz)

            if nnz <= 0:
                continue

            target_hmm_id = global_id_to_name.get(target_id, f"ID_{target_id}")

            if global_allowed_set is not None:
                if query_hmm_id not in global_allowed_set and target_hmm_id not in global_allowed_set:
                    continue

            if offset < 0 or offset + nnz > len(linear_vals):
                continue

            sl = slice(offset, offset + nnz)

            rows_block = rows[sl]
            cols_block = cols[sl]
            vals_block = linear_vals[sl]

            #  path 
            mask = (
                np.isfinite(vals_block)
                & (vals_block >= global_second_threshold)
            )

            if not np.any(mask):
                continue

            rows_block = rows_block[mask]
            cols_block = cols_block[mask]
            vals_block = vals_block[mask]

            if len(rows_block) == 0:
                continue

            total_pairs_after_second += 1
            total_points_after_second += len(rows_block)

            comparison_data = build_sparse_dict_from_block(
                rows_block,
                cols_block,
                vals_block
            )

            file_name_for_report = f"{query_hmm_id}_{target_hmm_id}.data"

            try:
                oriented_data, was_flipped = orient_data_for_analysis(comparison_data)
                hmm_x, hmm_y = (
                    (target_hmm_id, query_hmm_id)
                    if was_flipped
                    else (query_hmm_id, target_hmm_id)
                )

                if not oriented_data:
                    continue

                clean_x, unclean_x = find_patterns(
                    oriented_data,
                    MIN_SUB_SEGMENT_LENGTH
                )

                inverted_data = invert_dictionary(oriented_data)

                clean_y, unclean_y = find_patterns(
                    inverted_data,
                    MIN_SUB_SEGMENT_LENGTH
                )

                unclean_key_set = set()

                for res in unclean_x:
                    for sub in res["sub_segments"]:
                        unclean_key_set.add(_get_path_key(sub["path"], hmm_x, hmm_y))

                for res in unclean_y:
                    for sub in res["sub_segments"]:
                        inverted_path = [(p[1], p[0]) for p in sub["path"]]
                        unclean_key_set.add(_get_path_key(inverted_path, hmm_x, hmm_y))

                temp_clean_x_validated = []

                for res in clean_x:
                    is_dirty = False
                    for sub in res["sub_segments"]:
                        if _get_path_key(sub["path"], hmm_x, hmm_y) in unclean_key_set:
                            is_dirty = True
                            break

                    if is_dirty:
                        unclean_x.append(res)
                    else:
                        temp_clean_x_validated.append(res)

                temp_clean_y_validated = []

                for res in clean_y:
                    is_dirty = False
                    for sub in res["sub_segments"]:
                        inverted_path = [(p[1], p[0]) for p in sub["path"]]

                        if _get_path_key(inverted_path, hmm_x, hmm_y) in unclean_key_set:
                            is_dirty = True
                            break

                    if is_dirty:
                        unclean_y.append(res)
                    else:
                        temp_clean_y_validated.append(res)

                if hmm_x <= hmm_y:
                    results_to_add = process_analysis_results(
                        temp_clean_x_validated,
                        file_name_for_report,
                        hmm_x,
                        hmm_y,
                        TOP_N_RESULTS_TO_SAVE,
                        global_first_threshold,
                        global_second_threshold,
                    )
                else:
                    results_to_add = process_analysis_results(
                        temp_clean_y_validated,
                        file_name_for_report,
                        hmm_y,
                        hmm_x,
                        TOP_N_RESULTS_TO_SAVE,
                        global_first_threshold,
                        global_second_threshold,
                    )

                for rec in results_to_add:
                    is_gapped = rec.pop("_is_gapped", False)
                    main_detailed = rec.pop("_Main_Segment_Detailed", "")
                    sub_detailed = rec.pop("_Sub_Segments_Details_Detailed", "")

                    final_clean_results.append(rec.copy())

                    if is_gapped:
                        gap_rec = rec.copy()
                        gap_rec["Main_Segment"] = main_detailed
                        gap_rec["Sub_Segments_Details"] = sub_detailed
                        final_gap_results.append(gap_rec)

                final_unclean_results.extend(
                    process_analysis_results(
                        unclean_x,
                        file_name_for_report,
                        hmm_x,
                        hmm_y,
                        None,
                        global_first_threshold,
                        global_second_threshold,
                    )
                )

                final_unclean_results.extend(
                    process_analysis_results(
                        unclean_y,
                        file_name_for_report,
                        hmm_y,
                        hmm_x,
                        None,
                        global_first_threshold,
                        global_second_threshold,
                    )
                )

            except Exception:
                continue

    except Exception:
        return (file_name, 0, 0, 0, 0, 0)

    num_clean = len(final_clean_results)
    num_unclean = len(final_unclean_results)
    num_gap = len(final_gap_results)

    base_name = file_name.replace(".npz", ".csv")

    if num_clean > 0:
        try:
            df = pd.DataFrame(final_clean_results)
            cols = [c for c in df.columns if not c.startswith("_")]
            df = df[cols]
            df.to_csv(
                os.path.join(global_temp_csv_dir, base_name),
                index=False,
                encoding="utf-8-sig",
            )
        except Exception:
            num_clean = 0

    if num_unclean > 0:
        try:
            df = pd.DataFrame(final_unclean_results).drop_duplicates()
            cols = [c for c in df.columns if not c.startswith("_")]
            df = df[cols]

            if not df.empty:
                df.to_csv(
                    os.path.join(global_unclean_temp_csv_dir, base_name),
                    index=False,
                    encoding="utf-8-sig",
                )
        except Exception:
            num_unclean = 0

    if num_gap > 0:
        try:
            df = pd.DataFrame(final_gap_results)
            cols = [c for c in df.columns if not c.startswith("_")]
            df = df[cols]
            df.to_csv(
                os.path.join(global_gap_temp_csv_dir, base_name),
                index=False,
                encoding="utf-8-sig",
            )
        except Exception:
            num_gap = 0

    return (
        file_name,
        num_clean,
        num_unclean,
        num_gap,
        total_pairs_after_second,
        total_points_after_second,
    )


# ==============================================================================
# 6. coverage rescoring：Sadj 
# ==============================================================================

def process_details_string(details_str):
    segments = []

    if pd.isna(details_str) or str(details_str).strip() == "":
        return segments

    parts = str(details_str).split(";")

    for part in parts:
        match = re.search(
            r"(\d+)\s*-\s*(\d+)\s*->\s*(\d+)\s*-\s*(\d+)",
            part,
        )

        if match:
            segments.append(
                (
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                )
            )

    return segments


def load_coverage_file(pfam_id, cache):
    pfam_id = normalize_pfam_id(pfam_id)

    if pfam_id in cache:
        return cache[pfam_id]

    csv_path = os.path.join(PFAM_SEED_DIR, pfam_id, f"{pfam_id}_coverage.csv")

    if not os.path.exists(csv_path):
        cache[pfam_id] = None
        return None

    try:
        df = pd.read_csv(
            csv_path,
            usecols=["Match_State", "Completeness_Percent"],
        )

        data = dict(
            zip(
                df["Match_State"],
                df["Completeness_Percent"] / 100.0,
            )
        )

        cache[pfam_id] = data
        return data

    except Exception:
        cache[pfam_id] = None
        return None


def get_mean_coverage_from_dict(cov_data, start, end):
    if cov_data is None:
        return DEFAULT_MISSING_COVERAGE

    start, end = int(start), int(end)

    if start > end:
        start, end = end, start

    values = [
        cov_data[i]
        for i in range(start, end + 1)
        if i in cov_data
    ]

    if not values:
        return 0.0

    return sum(values) / len(values)


def process_dataframe_for_coverage_rescore(df):
    """
    ：

        Score = log10(Avg_Similarity * ((Coverage_Main + Coverage_Sub) / 2)^2)

    ：
        Avg_Similarity = path  linear_vals 
    """
    local_coverage_cache = {}

    scores = []
    main_covs = []
    sub_covs = []

    for row in df.itertuples(index=False):
        try:
            main_hmm = str(getattr(row, "Main_HMM"))
            sub_hmm = str(getattr(row, "Sub_HMM"))
            avg_similarity = float(getattr(row, "Avg_Similarity"))
            details = getattr(row, "Sub_Segments_Details")
        except Exception:
            scores.append(0.0)
            main_covs.append(0.0)
            sub_covs.append(0.0)
            continue

        segments = process_details_string(details)

        if not segments:
            scores.append(0.0)
            main_covs.append(0.0)
            sub_covs.append(0.0)
            continue

        main_data = load_coverage_file(main_hmm, local_coverage_cache)
        sub_data = load_coverage_file(sub_hmm, local_coverage_cache)

        seg_main_means = []
        seg_sub_means = []

        for m_s, m_e, s_s, s_e in segments:
            m_mean = get_mean_coverage_from_dict(main_data, m_s, m_e)
            s_mean = get_mean_coverage_from_dict(sub_data, s_s, s_e)

            seg_main_means.append(m_mean)
            seg_sub_means.append(s_mean)

        final_main_cov = (
            sum(seg_main_means) / len(seg_main_means)
            if seg_main_means
            else 0.0
        )

        final_sub_cov = (
            sum(seg_sub_means) / len(seg_sub_means)
            if seg_sub_means
            else 0.0
        )

        weight_factor = (final_main_cov + final_sub_cov) / 2.0

        new_raw_val = avg_similarity * (weight_factor ** 2)

        if new_raw_val > 0 and np.isfinite(new_raw_val):
            score = np.log10(new_raw_val)
        else:
            score = 0.0

        scores.append(score)
        main_covs.append(final_main_cov)
        sub_covs.append(final_sub_cov)

    result_df = df.copy()
    result_df["Coverage_Main"] = main_covs
    result_df["Coverage_Sub"] = sub_covs
    result_df["Score"] = scores

    return result_df


def _clean_scores_for_plot(scores):
    arr = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr >= PLOT_XMIN) & (arr <= PLOT_XMAX)]
    return arr


def _prob_weights(arr):
    if len(arr) == 0:
        return None
    return np.ones(len(arr), dtype=float) / len(arr)


def _draw_hist(ax, data, bins, color, label, alpha=0.32, lw=1.8):
    if len(data) == 0:
        return

    weights = _prob_weights(data)

    ax.hist(
        data,
        bins=bins,
        weights=weights,
        alpha=alpha,
        color=color,
        edgecolor="none",
        label=label,
    )

    ax.hist(
        data,
        bins=bins,
        weights=weights,
        histtype="step",
        linewidth=lw,
        color=color,
    )


def plot_score_distribution(final_scores, out_path):
    data = _clean_scores_for_plot(final_scores)

    if len(data) == 0:
        print("[WARN] ：Score ")
        return

    bins = np.linspace(PLOT_XMIN, PLOT_XMAX, DIST_BINS + 1)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    _draw_hist(
        ax,
        data,
        bins,
        "seagreen",
        "Final Score = log10(Avg_Similarity × w²)",
        alpha=0.28,
        lw=1.8,
    )

    ax.axvline(
        SCORE_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Score threshold = {SCORE_THRESHOLD}",
    )

    ax.set_xlim(PLOT_XMIN, PLOT_XMAX)
    ax.set_xlabel("Score")
    ax.set_ylabel("Proportion")
    ax.set_title("Final rescored score distribution")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)

    print(f"[DONE] : {out_path}")


def run_coverage_rescore_pipeline(input_csv, output_csv, output_plot):
    print("\n[INFO]  coverage rescoring")
    print(f"[INFO] : {input_csv}")
    print(f"[INFO] : {output_csv}")
    print(f"[INFO] : Coverage >= {COVERAGE_THRESHOLD}, Score > {SCORE_THRESHOLD}")

    if not os.path.exists(input_csv):
        print(f"[WARN]  CSV， coverage rescoring: {input_csv}")
        return {
            "input_rows": 0,
            "coverage_pass_rows": 0,
            "final_rows": 0,
            "covered_families_final": 0,
        }

    try:
        df = pd.read_csv(input_csv, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding="gbk")

    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    print(f"[INFO]  clean path : {len(df)}")

    if df.empty:
        pd.DataFrame().to_csv(output_csv, index=False, encoding="utf-8-sig")
        return {
            "input_rows": 0,
            "coverage_pass_rows": 0,
            "final_rows": 0,
            "covered_families_final": 0,
        }

    result_df = process_dataframe_for_coverage_rescore(df)

    coverage_filtered_df = result_df[
        (result_df["Coverage_Main"] >= COVERAGE_THRESHOLD) &
        (result_df["Coverage_Sub"] >= COVERAGE_THRESHOLD)
    ].copy()

    print(
        f"[INFO]  coverage  "
        f"{len(coverage_filtered_df)} / {len(result_df)} "
    )

    final_df = coverage_filtered_df[
        coverage_filtered_df["Score"] > SCORE_THRESHOLD
    ].copy()

    if SORT_FINAL_OUTPUT and not final_df.empty:
        final_df = final_df.sort_values(
            by="Score",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    print(f"[RESULT] : {len(df)} -> {len(final_df)} ")

    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] coverage rescoring : {output_csv}")

    if MAKE_SCORE_DISTRIBUTION_PLOT:
        plot_score_distribution(final_df["Score"], output_plot)

    covered_families = set()

    if not final_df.empty:
        covered_families = (
            set(final_df["Main_HMM"].astype(str)) |
            set(final_df["Sub_HMM"].astype(str))
        )

    return {
        "input_rows": int(len(df)),
        "coverage_pass_rows": int(len(coverage_filtered_df)),
        "final_rows": int(len(final_df)),
        "covered_families_final": int(len(covered_families)),
    }


# ==============================================================================
# 7.  CSV
# ==============================================================================

def merge_csv_files(temp_dir, global_path, description):
    print(f"\n {description}...")

    temp_csv_files = glob.glob(os.path.join(temp_dir, "*.csv"))

    if not temp_csv_files:
        return False

    header_written = False

    try:
        with open(global_path, "w", encoding="utf-8-sig") as f_global_out:
            for temp_file in tqdm(temp_csv_files, desc=f" {description}"):
                with open(temp_file, "r", encoding="utf-8") as f_in:
                    if not header_written:
                        shutil.copyfileobj(f_in, f_global_out)
                        header_written = True
                    else:
                        try:
                            next(f_in)
                        except StopIteration:
                            continue
                        shutil.copyfileobj(f_in, f_global_out)

        return True

    except Exception as e:
        print(f"[ERROR] : {e}")
        return False


# ==============================================================================
# 8. 
# ==============================================================================

def run_one_combo(first_threshold, second_threshold):
    ctag = combo_tag(first_threshold, second_threshold)

    base_input_dir = os.path.join(
        PREPROCESSED_ROOT,
        first_dir_name(first_threshold),
    )

    output_dir = os.path.join(OUTPUT_ROOT, ctag)

    csv_output_dir = os.path.join(output_dir, "temp_csv_files")
    unclean_csv_output_dir = os.path.join(output_dir, "temp_unclean_csv_files")
    gap_csv_output_dir = os.path.join(output_dir, "temp_gap_csv_files")

    global_csv_path = os.path.join(output_dir, f"{ctag}_global_clean_paths_before_rescore.csv")
    global_unclean_csv_path = os.path.join(output_dir, f"{ctag}_global_unclean_overlap_report.csv")
    global_gap_csv_path = os.path.join(output_dir, f"{ctag}_global_jump_gap_report.csv")

    output_sadj_csv = os.path.join(output_dir, f"{ctag}_global_clean_paths_Sadj.csv")
    output_score_dist_png = os.path.join(output_dir, f"{ctag}_score_distribution_Sadj.png")

    summary_csv = os.path.join(output_dir, f"{ctag}_summary.csv")

    if RESUME and os.path.exists(summary_csv) and os.path.exists(output_sadj_csv):
        print(f"[SKIP] ，: {ctag}")
        try:
            old_summary = pd.read_csv(summary_csv, encoding="utf-8-sig")
            if not old_summary.empty:
                return old_summary.iloc[0].to_dict()
        except Exception:
            return None

    if not os.path.isdir(base_input_dir):
        print(f"[WARN] ，: {base_input_dir}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    for dir_path in [csv_output_dir, unclean_csv_output_dir, gap_csv_output_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    if USE_HMM_FILTER_LIST:
        if not ALLOWED_LIST_FILE:
            raise ValueError("--allowed-list-file is required with --use-hmm-filter-list.")
        allowed_hmm_set = load_allowed_list(ALLOWED_LIST_FILE)
    else:
        allowed_hmm_set = None

    id_to_name = load_pfam_mapping(PFAM_MAPPING_FILE)

    all_npz_files = glob.glob(os.path.join(base_input_dir, NPZ_GLOB_PATTERN))

    print("=" * 100)
    print(f"[RUN] {ctag}")
    print(f"First threshold: {first_threshold}")
    print(f"Second threshold: {second_threshold}")
    print(f"Input dir: {base_input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Found {len(all_npz_files)} NPZ files")
    print("=" * 100)

    total_clean = 0
    total_unclean = 0
    total_gap = 0
    total_pairs_after_second = 0
    total_points_after_second = 0

    if not all_npz_files:
        return None

    with multiprocessing.Pool(
        processes=NUM_PROCESSES,
        maxtasksperchild=1,
        initializer=init_worker,
        initargs=(
            allowed_hmm_set,
            csv_output_dir,
            unclean_csv_output_dir,
            gap_csv_output_dir,
            id_to_name,
            float(first_threshold),
            float(second_threshold),
        ),
    ) as pool:
        iterator = pool.imap_unordered(analyze_npz_file, all_npz_files)
        pbar = tqdm(iterator, total=len(all_npz_files), desc=f"Path {ctag}")

        for (_, n_clean, n_unclean, n_gap, n_pair_second, n_point_second) in pbar:
            total_clean += n_clean
            total_unclean += n_unclean
            total_gap += n_gap
            total_pairs_after_second += n_pair_second
            total_points_after_second += n_point_second

            pbar.set_postfix_str(
                f"Cln:{total_clean}|Uncl:{total_unclean}|Gap:{total_gap}"
            )

    print("\nMerging results...")

    clean_merged = False
    unclean_merged = False
    gap_merged = False

    if total_clean > 0:
        clean_merged = merge_csv_files(
            csv_output_dir,
            global_csv_path,
            "Clean CSV",
        )

    if total_unclean > 0:
        unclean_merged = merge_csv_files(
            unclean_csv_output_dir,
            global_unclean_csv_path,
            "Unclean CSV",
        )

    if total_gap > 0:
        gap_merged = merge_csv_files(
            gap_csv_output_dir,
            global_gap_csv_path,
            "Gap/Jump CSV",
        )

    if clean_merged and os.path.exists(global_csv_path):
        rescore_summary = run_coverage_rescore_pipeline(
            input_csv=global_csv_path,
            output_csv=output_sadj_csv,
            output_plot=output_score_dist_png,
        )
    else:
        print("[WARN]  clean ， coverage rescoring")
        rescore_summary = {
            "input_rows": 0,
            "coverage_pass_rows": 0,
            "final_rows": 0,
            "covered_families_final": 0,
        }

    summary = {
        "first_threshold": float(first_threshold),
        "second_threshold": float(second_threshold),
        "combo_tag": ctag,
        "input_dir": base_input_dir,
        "output_dir": output_dir,
        "npz_files": int(len(all_npz_files)),
        "pairs_after_second_filter": int(total_pairs_after_second),
        "points_after_second_filter": int(total_points_after_second),
        "clean_paths_before_rescore": int(total_clean),
        "unclean_paths": int(total_unclean),
        "gap_paths": int(total_gap),
        "coverage_pass_rows": int(rescore_summary["coverage_pass_rows"]),
        "final_sadj_rows": int(rescore_summary["final_rows"]),
        "covered_families_final": int(rescore_summary["covered_families_final"]),
        "score_definition": "Score = log10(Avg_Similarity * ((Coverage_Main + Coverage_Sub)/2)^2)",
        "coverage_threshold": COVERAGE_THRESHOLD,
        "score_threshold": SCORE_THRESHOLD,
        "clean_before_rescore_csv": global_csv_path if clean_merged else "",
        "sadj_output_csv": output_sadj_csv if os.path.exists(output_sadj_csv) else "",
        "unclean_csv": global_unclean_csv_path if unclean_merged else "",
        "gap_csv": global_gap_csv_path if gap_merged else "",
    }

    pd.DataFrame([summary]).to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("-" * 100)
    print(f"[DONE] {ctag}")
    print(f"Pairs after second filter: {total_pairs_after_second:,}")
    print(f"Points after second filter: {total_points_after_second:,}")
    print(f"Clean before rescore: {total_clean:,}")
    print(f"Unclean: {total_unclean:,}")
    print(f"Gap: {total_gap:,}")
    print(f"Coverage pass rows: {summary['coverage_pass_rows']:,}")
    print(f"Final Sadj rows: {summary['final_sadj_rows']:,}")
    print(f"Covered families final: {summary['covered_families_final']:,}")
    print(f"Sadj output: {output_sadj_csv}")
    print("-" * 100)

    if not KEEP_TEMP_FILES:
        shutil.rmtree(csv_output_dir, ignore_errors=True)
        shutil.rmtree(unclean_csv_output_dir, ignore_errors=True)
        shutil.rmtree(gap_csv_output_dir, ignore_errors=True)

    return summary


# ==============================================================================
# 9. 
# ==============================================================================

def get_combinations():
    return [
        (float(a), float(b))
        for a, b in product(FIRST_THRESHOLDS, SECOND_THRESHOLDS)
    ]


def main():
    args = parse_args()
    apply_runtime_args(args)
    multiprocessing.freeze_support()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    combinations = get_combinations()

    print("=" * 100)
    print("Path building + Sadj rescoring parameter scan")
    print("Grid: FIRST_THRESHOLDS × SECOND_THRESHOLDS.")
    print("Coverage counts refer to Pfam families.")
    print("Final score definition:")
    print("  Score = log10(Avg_Similarity * ((Coverage_Main + Coverage_Sub)/2)^2)")
    print(f"PREPROCESSED_ROOT: {PREPROCESSED_ROOT}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"PFAM_SEED_DIR: {PFAM_SEED_DIR}")
    print(f"First thresholds: {FIRST_THRESHOLDS}")
    print(f"Second thresholds: {SECOND_THRESHOLDS}")
    print(f"Total combinations: {len(combinations)}")
    print("=" * 100)

    all_summaries = []

    global_summary_csv = os.path.join(
        OUTPUT_ROOT,
        "path_parameter_scan_sadj_summary.csv",
    )

    for first_threshold, second_threshold in combinations:
        summary = run_one_combo(first_threshold, second_threshold)

        if summary is not None:
            all_summaries.append(summary)

            pd.DataFrame(all_summaries).to_csv(
                global_summary_csv,
                index=False,
                encoding="utf-8-sig",
            )

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)

        summary_df = summary_df.sort_values(
            by=[
                "covered_families_final",
                "final_sadj_rows",
                "points_after_second_filter",
            ],
            ascending=[False, True, True],
        ).reset_index(drop=True)

        summary_df.to_csv(
            global_summary_csv,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nTop parameter combinations:")
        display_cols = [
            "first_threshold",
            "second_threshold",
            "covered_families_final",
            "final_sadj_rows",
            "coverage_pass_rows",
            "clean_paths_before_rescore",
            "unclean_paths",
            "gap_paths",
            "points_after_second_filter",
        ]

        print(summary_df[display_cols].head(30).to_string(index=False))

    print("=" * 100)
    print("[DONE] ")
    print(f"Global summary: {global_summary_csv}")
    print("=" * 100)


if __name__ == "__main__":
    main()
