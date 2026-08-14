#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build diagonal paths from background-filtered ProDive pickle files.

Input
-----
This script reads filtered pickle files generated after C++ KL calculation and
background-based threshold filtering:

    kl_<query_record_name>_filtered.pk

Each pickle is expected to contain:

    {
        target_record_name: {
            query_window_start_1based: {
                target_window_start_1based: linear_score
            }
        }
    }

Output
------
The script extracts diagonal chains of high-scoring window pairs and writes
three global CSV files:

    global_high_score_summary.csv       clean one-to-one paths
    global_UNCL_overlap_report.csv      overlap/ambiguous paths
    global_jump_gap_report.csv          clean paths with skipped window starts

The path-building logic preserves the original ProDive downstream convention:
- window-start coordinates are 1-based;
- adjacent points can be connected when both axes advance by 1..fragment-1;
- physical segment end coordinates are extended by fragment-1;
- clean path score is log10(number_of_points * mean_linear_score).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ==============================================================================
# Runtime globals for multiprocessing workers
# ==============================================================================

G_ALLOWED_SET: Optional[Set[str]] = None
G_TEMP_CLEAN_DIR: Optional[str] = None
G_TEMP_UNCLEAN_DIR: Optional[str] = None
G_TEMP_GAP_DIR: Optional[str] = None
G_ARGS: Optional[argparse.Namespace] = None
G_SAFE_TO_RECORD: Dict[str, str] = {}


# ==============================================================================
# Basic utilities
# ==============================================================================


def safe_filename_component(value: str) -> str:
    """Return the same filesystem-safe component used by the filtering stage."""
    value = str(value).strip()
    value = value.replace(os.sep, "_")
    if os.altsep:
        value = value.replace(os.altsep, "_")
    value = re.sub(r"[^A-Za-z0-9._+-]+", "_", value)
    value = value.strip("._")
    return value or "record"


def load_mapping_names(mapping_file: Optional[str]) -> List[str]:
    """Load record names from a `record_name: integer_id` mapping file."""
    if not mapping_file:
        return []

    names: List[Tuple[int, str]] = []
    with open(mapping_file, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or ":" not in line:
                continue
            left, right = line.split(":", 1)
            try:
                idx = int(right.strip())
            except ValueError:
                print(f"[WARN] Bad mapping value at line {line_no}: {line}", file=sys.stderr)
                continue
            names.append((idx, left.strip()))

    names.sort(key=lambda x: x[0])
    return [name for _, name in names]


def load_allowed_list(file_path: str) -> Set[str]:
    """Load an optional list of record names used to restrict path extraction."""
    allowed: Set[str] = set()
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if item:
                allowed.add(item)
    return allowed


def resolve_query_name_from_pk(path: Path) -> Optional[str]:
    """Resolve the original query record name from a filtered PK filename."""
    name = path.name
    match = re.match(r"^kl_(.+?)_filtered\.pk$", name)
    if not match:
        return None
    token = match.group(1)

    # If mapping was provided, token may be a sanitized filename component.
    if token in G_SAFE_TO_RECORD:
        return G_SAFE_TO_RECORD[token]

    # Fallback for old files whose filename already contains the true query ID.
    return token


def orient_data_for_analysis(data: Dict[int, Dict[int, float]]) -> Tuple[Dict[int, Dict[int, float]], bool]:
    """
    Orient sparse points so that the denser axis is treated consistently.

    The heuristic is preserved from the original downstream script. It can flip
    x/y when the number of main-axis keys is much larger than the average number
    of sub-axis keys.
    """
    if not data:
        return data, False

    valid_sub_dicts = [v for v in data.values() if isinstance(v, dict) and v]
    if not valid_sub_dicts:
        return data, False

    num_main_keys = len(data)
    avg_sub_keys = float(np.mean([len(sub_dict) for sub_dict in valid_sub_dicts]))

    if num_main_keys > avg_sub_keys * 2:
        inverted_data: Dict[int, Dict[int, float]] = defaultdict(dict)
        for main_key, sub_dict in data.items():
            if not isinstance(sub_dict, dict):
                continue
            for sub_key, value in sub_dict.items():
                inverted_data[int(sub_key)][int(main_key)] = float(value)
        return dict(inverted_data), True

    # Normalize keys/values to int/float for downstream processing.
    out: Dict[int, Dict[int, float]] = defaultdict(dict)
    for main_key, sub_dict in data.items():
        if not isinstance(sub_dict, dict):
            continue
        try:
            x = int(main_key)
        except Exception:
            continue
        for sub_key, value in sub_dict.items():
            try:
                out[x][int(sub_key)] = float(value)
            except Exception:
                continue
    return dict(out), False


def invert_dictionary(data: Dict[int, Dict[int, float]]) -> Dict[int, Dict[int, float]]:
    """Invert a sparse `{x: {y: value}}` dictionary."""
    inverted_data: Dict[int, Dict[int, float]] = defaultdict(dict)
    for main_key, sub_dict in data.items():
        for sub_key, value in sub_dict.items():
            inverted_data[int(sub_key)][int(main_key)] = float(value)
    return dict(inverted_data)


def is_segment_clean(path: Sequence[Tuple[int, int]], data: Dict[int, Dict[int, float]]) -> bool:
    """Return False if neighboring y positions exist at any x in the path."""
    for x, y in path:
        sub_keys_at_x = data.get(x, {})
        if (y + 1) in sub_keys_at_x or (y - 1) in sub_keys_at_x:
            return False
    return True


def get_path_key(path_list: Sequence[Tuple[int, int]], hmm1: str, hmm2: str) -> Tuple[Tuple[str, Tuple[int, ...]], Tuple[str, Tuple[int, ...]]]:
    """Build an orientation-independent key for detecting clean/unclean overlap."""
    xs_coords = tuple(sorted(set(p[0] for p in path_list)))
    ys_coords = tuple(sorted(set(p[1] for p in path_list)))
    part1 = (hmm1, xs_coords)
    part2 = (hmm2, ys_coords)
    return (part1, part2) if hmm1 <= hmm2 else (part2, part1)


# ==============================================================================
# Segment compression
# ==============================================================================


def compress_segment_list_merged(segment_list: Sequence[int], length_extension: int) -> str:
    """Compress window starts into merged physical segment intervals."""
    if not segment_list:
        return ""

    values = sorted(int(x) for x in segment_list)
    raw_intervals: List[Tuple[int, int]] = []

    start = values[0]
    end = values[0]
    for val in values[1:]:
        if val == end + 1:
            end = val
        else:
            raw_intervals.append((start, end + length_extension))
            start = val
            end = val
    raw_intervals.append((start, end + length_extension))

    merged_intervals: List[Tuple[int, int]] = []
    current_start, current_end = raw_intervals[0]
    for next_start, next_end in raw_intervals[1:]:
        if next_start <= current_end + 1:
            current_end = max(current_end, next_end)
        else:
            merged_intervals.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged_intervals.append((current_start, current_end))

    return ",".join(f"{s}-{e}" for s, e in merged_intervals)


def compress_segment_list_detailed(segment_list: Sequence[int], length_extension: int) -> str:
    """Compress window starts without merging intervals created by gaps."""
    if not segment_list:
        return ""

    values = sorted(int(x) for x in segment_list)
    ranges: List[str] = []

    start = values[0]
    end = values[0]
    for val in values[1:]:
        if val == end + 1:
            end = val
        else:
            ranges.append(f"{start}-{end + length_extension}")
            start = val
            end = val
    ranges.append(f"{start}-{end + length_extension}")

    return ",".join(ranges)


# ==============================================================================
# Path extraction
# ==============================================================================


def process_path_as_one_to_one(
    path: Sequence[Tuple[int, int]],
    data: Dict[int, Dict[int, float]],
    length_extension: int,
) -> Optional[Dict[str, Any]]:
    """Convert one clean path into a result record before CSV formatting."""
    if not path:
        return None

    x_coords = [p[0] for p in path]
    y_coords = [p[1] for p in path]

    true_main_segment = sorted(set(x_coords))
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    physical_main_len = (max_x - min_x + 1) + length_extension
    physical_sub_len = (max_y - min_y + 1) + length_extension
    is_gapped = (max_x - min_x + 1) > len(set(x_coords))

    values = [float(data[p[0]][p[1]]) for p in path]
    avg_value = float(np.mean(values)) if values else 0.0
    evidence_count = len(path)
    final_score = evidence_count * avg_value

    stats = {
        "main_segment_length": physical_main_len,
        "sub_segment_count": 1,
        "avg_sub_segment_length": f"{physical_sub_len:.2f}",
        "structural_fit_iou": "1.0000",
        "avg_similarity": f"{avg_value}",
    }
    sub_segment_details = [{"path": list(path), "avg_value": f"{avg_value:.4f}"}]

    return {
        "score": final_score,
        "main_segment": true_main_segment,
        "stats": stats,
        "sub_segments": sub_segment_details,
        "is_gapped": is_gapped,
    }


def find_patterns(
    data: Dict[int, Dict[int, float]],
    min_path_points: int,
    fragment_size: int,
    length_extension: int,
    overlap_iou_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Find clean one-to-one paths and overlap-related unclean paths."""
    if not data:
        return [], []

    points: Set[Tuple[int, int]] = set()
    successors: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)

    for x in sorted(data.keys()):
        sub = data.get(x, {})
        for y in sub:
            points.add((int(x), int(y)))

    sorted_points = sorted(points)

    # Connect each point to the nearest valid predecessor on the same diagonal
    # with a step size up to fragment_size - 1.
    for point in sorted_points:
        x, y = point
        for delta in range(1, fragment_size):
            predecessor = (x - delta, y - delta)
            if predecessor in points:
                successors[predecessor].append(point)
                break

    memo: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def trace_longest_path_from(node: Tuple[int, int]) -> List[Tuple[int, int]]:
        # The original implementation follows the first successor after sorting.
        # This behavior is preserved for reproducibility.
        if node in memo:
            return memo[node]
        if not successors.get(node):
            memo[node] = [node]
            return memo[node]
        path = [node] + trace_longest_path_from(sorted(successors[node])[0])
        memo[node] = path
        return path

    has_incoming: Set[Tuple[int, int]] = set()
    for dests in successors.values():
        for dst in dests:
            has_incoming.add(dst)

    start_nodes = sorted([p for p in points if p not in has_incoming])
    all_raw_paths = [trace_longest_path_from(start_node) for start_node in start_nodes]
    structurally_clean_paths = [path for path in all_raw_paths if is_segment_clean(path, data)]

    path_sets: List[Tuple[List[Tuple[int, int]], Set[int]]] = [
        (list(path), set(p[0] for p in path)) for path in structurally_clean_paths
    ]

    path_clusters: List[List[List[Tuple[int, int]]]] = []
    while path_sets:
        current_cluster_paths = [path_sets[0][0]]
        main_union = set(path_sets[0][1])
        path_sets.pop(0)
        idx = 0
        while idx < len(path_sets):
            path_to_check, set_to_check = path_sets[idx]
            intersection = len(main_union.intersection(set_to_check))
            union = len(main_union.union(set_to_check))
            iou = intersection / union if union > 0 else 0.0
            if iou > overlap_iou_threshold:
                current_cluster_paths.append(path_to_check)
                main_union.update(set_to_check)
                path_sets.pop(idx)
            else:
                idx += 1
        path_clusters.append(current_cluster_paths)

    clean_results: List[Dict[str, Any]] = []
    unclean_results: List[Dict[str, Any]] = []

    for path_group in path_clusters:
        if len(path_group) > 1:
            has_y_overlap = False
            for i in range(len(path_group)):
                y_i = {p[1] for p in path_group[i]}
                for j in range(i + 1, len(path_group)):
                    if not y_i.isdisjoint({p[1] for p in path_group[j]}):
                        has_y_overlap = True
                        break
                if has_y_overlap:
                    break

            if has_y_overlap:
                relevant_paths = [p for p in path_group if len(p) >= min_path_points]
                if not relevant_paths:
                    continue
                main_axis_sets = [set(p[0] for p in path) for path in relevant_paths]
                main_coords_union = set().union(*main_axis_sets)
                if not main_coords_union:
                    continue
                min_x, max_x = min(main_coords_union), max(main_coords_union)
                physical_len = (max_x - min_x + 1) + length_extension
                sub_segment_details = [
                    {"path": path, "avg_value": "N/A (Overlap)"} for path in relevant_paths
                ]
                stats = {
                    "main_segment_length": physical_len,
                    "sub_segment_count": len(relevant_paths),
                    "avg_sub_segment_length": "N/A",
                    "structural_fit_iou": "N/A",
                    "avg_similarity": "N/A (Overlap)",
                }
                unclean_results.append({
                    "score": -1,
                    "main_segment": sorted(main_coords_union),
                    "stats": stats,
                    "sub_segments": sub_segment_details,
                })
            else:
                for path in path_group:
                    if len(path) >= min_path_points:
                        result = process_path_as_one_to_one(path, data, length_extension)
                        if result:
                            clean_results.append(result)
        elif len(path_group) == 1:
            path = path_group[0]
            if len(path) >= min_path_points:
                result = process_path_as_one_to_one(path, data, length_extension)
                if result:
                    clean_results.append(result)

    return sorted(clean_results, key=lambda x: x["score"], reverse=True), unclean_results


# ==============================================================================
# CSV formatting
# ==============================================================================


def format_analysis_results(
    results: Sequence[Dict[str, Any]],
    file_name: str,
    hmm_main: str,
    hmm_sub: str,
    top_n_to_save: Optional[int],
    length_extension: int,
) -> List[Dict[str, Any]]:
    """Format extracted paths as CSV-ready records."""
    formatted_results: List[Dict[str, Any]] = []
    selected = results[:top_n_to_save] if top_n_to_save is not None else results

    for res in selected:
        is_gapped = bool(res.get("is_gapped", False))
        raw_score = float(res.get("score", 0.0))
        transformed_score = np.log10(raw_score + 1e-9) if raw_score > 0 else raw_score
        formatted_score = f"{transformed_score:.4f}"
        stats = res["stats"]

        merged_details_list: List[str] = []

        if is_gapped:
            detailed_details_list: List[str] = []
            for sub in res["sub_segments"]:
                path = sub["path"]
                main_part_list = [p[0] for p in path]
                sub_part_list = [p[1] for p in path]

                main_str_merged = compress_segment_list_merged(main_part_list, length_extension)
                sub_str_merged = compress_segment_list_merged(sub_part_list, length_extension)
                merged_details_list.append(f"{main_str_merged} -> {sub_str_merged}")

                main_str_detailed = compress_segment_list_detailed(main_part_list, length_extension)
                sub_str_detailed = compress_segment_list_detailed(sub_part_list, length_extension)
                avg_val_str = str(sub.get("avg_value", ""))
                suffix = " [OVERLAP]" if "N/A" in avg_val_str else ""
                detailed_details_list.append(f"{main_str_detailed} -> {sub_str_detailed}{suffix}")

            merged_sub_str = "; ".join(merged_details_list)
            merged_main_str = compress_segment_list_merged(res["main_segment"], length_extension)
            detailed_sub_str = "; ".join(detailed_details_list)
            detailed_main_str = compress_segment_list_detailed(res["main_segment"], length_extension)
        else:
            for sub in res["sub_segments"]:
                path = sub["path"]
                main_part_list = [p[0] for p in path]
                sub_part_list = [p[1] for p in path]
                main_str_merged = compress_segment_list_merged(main_part_list, length_extension)
                sub_str_merged = compress_segment_list_merged(sub_part_list, length_extension)
                merged_details_list.append(f"{main_str_merged} -> {sub_str_merged}")

            merged_sub_str = "; ".join(merged_details_list)
            merged_main_str = compress_segment_list_merged(res["main_segment"], length_extension)
            detailed_sub_str = ""
            detailed_main_str = ""

        record = {
            "File": file_name,
            "Score": formatted_score,
            "Main_HMM": hmm_main,
            "Sub_HMM": hmm_sub,
            "Main_Segment": merged_main_str,
            "Main_Segment_Len": stats["main_segment_length"],
            "Num_Sub_Segments": stats["sub_segment_count"],
            "Avg_Sub_Segment_Len": stats["avg_sub_segment_length"],
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


def write_records_csv(records: List[Dict[str, Any]], output_path: str, drop_internal: bool = True, drop_duplicates: bool = False) -> int:
    """Write records to CSV and return the number of written rows."""
    if not records:
        return 0
    df = pd.DataFrame(records)
    if drop_duplicates:
        df = df.drop_duplicates()
    if drop_internal:
        df = df[[c for c in df.columns if not c.startswith("_")]]
    if df.empty:
        return 0
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return int(len(df))


# ==============================================================================
# Worker
# ==============================================================================


def init_worker(args: argparse.Namespace, allowed_set: Optional[Set[str]], safe_to_record: Dict[str, str]) -> None:
    global G_ALLOWED_SET, G_TEMP_CLEAN_DIR, G_TEMP_UNCLEAN_DIR, G_TEMP_GAP_DIR, G_ARGS, G_SAFE_TO_RECORD
    G_ARGS = args
    G_ALLOWED_SET = allowed_set
    G_TEMP_CLEAN_DIR = str(Path(args.output_dir) / "temp_clean_csv")
    G_TEMP_UNCLEAN_DIR = str(Path(args.output_dir) / "temp_unclean_csv")
    G_TEMP_GAP_DIR = str(Path(args.output_dir) / "temp_gap_csv")
    G_SAFE_TO_RECORD = safe_to_record


def analyze_pk_file(file_path_str: str) -> Dict[str, Any]:
    """Analyze one filtered PK file and write temporary CSV outputs."""
    if G_ARGS is None:
        raise RuntimeError("Worker arguments were not initialized")

    file_path = Path(file_path_str)
    file_name = file_path.name
    query_hmm_id = resolve_query_name_from_pk(file_path)
    if query_hmm_id is None:
        return {"file": file_name, "ok": False, "reason": "bad_filtered_pk_filename", "clean": 0, "unclean": 0, "gap": 0}

    final_clean_results: List[Dict[str, Any]] = []
    final_unclean_results: List[Dict[str, Any]] = []
    final_gap_results: List[Dict[str, Any]] = []
    errors = 0
    error_examples: List[str] = []

    try:
        with file_path.open("rb") as handle:
            data_in_file = pickle.load(handle)
    except Exception as exc:
        return {"file": file_name, "ok": False, "reason": f"pickle_load_failed:{exc!r}", "clean": 0, "unclean": 0, "gap": 0}

    if not isinstance(data_in_file, dict):
        return {"file": file_name, "ok": False, "reason": "input_pk_not_dict", "clean": 0, "unclean": 0, "gap": 0}

    for target_hmm_id_raw, comparison_data in data_in_file.items():
        target_hmm_id = str(target_hmm_id_raw)

        if G_ALLOWED_SET is not None:
            if query_hmm_id not in G_ALLOWED_SET and target_hmm_id not in G_ALLOWED_SET:
                continue

        file_name_for_report = f"{query_hmm_id}_{target_hmm_id}.data"

        try:
            oriented_data, was_flipped = orient_data_for_analysis(comparison_data)
            hmm_x, hmm_y = (target_hmm_id, query_hmm_id) if was_flipped else (query_hmm_id, target_hmm_id)
            if not oriented_data:
                continue

            clean_x, unclean_x = find_patterns(
                oriented_data,
                min_path_points=G_ARGS.min_path_points,
                fragment_size=G_ARGS.fragment,
                length_extension=G_ARGS.fragment - 1,
                overlap_iou_threshold=G_ARGS.overlap_iou_threshold,
            )
            inverted_data = invert_dictionary(oriented_data)
            clean_y, unclean_y = find_patterns(
                inverted_data,
                min_path_points=G_ARGS.min_path_points,
                fragment_size=G_ARGS.fragment,
                length_extension=G_ARGS.fragment - 1,
                overlap_iou_threshold=G_ARGS.overlap_iou_threshold,
            )

            unclean_key_set = set()
            for res in unclean_x:
                for sub in res["sub_segments"]:
                    unclean_key_set.add(get_path_key(sub["path"], hmm_x, hmm_y))
            for res in unclean_y:
                for sub in res["sub_segments"]:
                    inverted_path = [(p[1], p[0]) for p in sub["path"]]
                    unclean_key_set.add(get_path_key(inverted_path, hmm_x, hmm_y))

            clean_x_validated: List[Dict[str, Any]] = []
            for res in clean_x:
                is_dirty = any(get_path_key(sub["path"], hmm_x, hmm_y) in unclean_key_set for sub in res["sub_segments"])
                if is_dirty:
                    unclean_x.append(res)
                else:
                    clean_x_validated.append(res)

            clean_y_validated: List[Dict[str, Any]] = []
            for res in clean_y:
                is_dirty = False
                for sub in res["sub_segments"]:
                    inverted_path = [(p[1], p[0]) for p in sub["path"]]
                    if get_path_key(inverted_path, hmm_x, hmm_y) in unclean_key_set:
                        is_dirty = True
                        break
                if is_dirty:
                    unclean_y.append(res)
                else:
                    clean_y_validated.append(res)

            if hmm_x <= hmm_y:
                results_to_add = format_analysis_results(
                    clean_x_validated,
                    file_name_for_report,
                    hmm_x,
                    hmm_y,
                    G_ARGS.top_n,
                    G_ARGS.fragment - 1,
                )
            else:
                results_to_add = format_analysis_results(
                    clean_y_validated,
                    file_name_for_report,
                    hmm_y,
                    hmm_x,
                    G_ARGS.top_n,
                    G_ARGS.fragment - 1,
                )

            for rec in results_to_add:
                is_gapped = bool(rec.pop("_is_gapped", False))
                main_detailed = rec.pop("_Main_Segment_Detailed", "")
                sub_detailed = rec.pop("_Sub_Segments_Details_Detailed", "")

                final_clean_results.append(rec.copy())

                if is_gapped:
                    gap_rec = rec.copy()
                    gap_rec["Main_Segment"] = main_detailed
                    gap_rec["Sub_Segments_Details"] = sub_detailed
                    final_gap_results.append(gap_rec)

            final_unclean_results.extend(format_analysis_results(unclean_x, file_name_for_report, hmm_x, hmm_y, None, G_ARGS.fragment - 1))
            final_unclean_results.extend(format_analysis_results(unclean_y, file_name_for_report, hmm_y, hmm_x, None, G_ARGS.fragment - 1))

        except Exception as exc:
            errors += 1
            if len(error_examples) < 5:
                error_examples.append(f"{target_hmm_id}: {exc!r}")
            if G_ARGS.strict:
                raise
            continue

    base_name = file_name.replace(".pk", ".csv")
    clean_count = write_records_csv(final_clean_results, os.path.join(G_TEMP_CLEAN_DIR, base_name))
    unclean_count = write_records_csv(final_unclean_results, os.path.join(G_TEMP_UNCLEAN_DIR, base_name), drop_duplicates=True)
    gap_count = write_records_csv(final_gap_results, os.path.join(G_TEMP_GAP_DIR, base_name))

    return {
        "file": file_name,
        "ok": True,
        "reason": "",
        "clean": int(clean_count),
        "unclean": int(unclean_count),
        "gap": int(gap_count),
        "target_errors": int(errors),
        "target_error_examples": error_examples,
    }


# ==============================================================================
# Merge and driver
# ==============================================================================


def merge_csv_files(temp_dir: Path, global_path: Path, description: str) -> bool:
    """Merge worker-level temporary CSV files into one global CSV."""
    temp_csv_files = sorted(glob.glob(str(temp_dir / "*.csv")))
    if not temp_csv_files:
        return False

    header_written = False
    with global_path.open("w", encoding="utf-8-sig", newline="") as global_out:
        iterator: Iterable[str] = temp_csv_files
        if HAS_TQDM:
            iterator = tqdm(temp_csv_files, desc=f"Merging {description}", unit="file")
        for temp_file in iterator:
            with open(temp_file, "r", encoding="utf-8-sig", newline="") as handle:
                if not header_written:
                    shutil.copyfileobj(handle, global_out)
                    header_written = True
                else:
                    try:
                        next(handle)
                    except StopIteration:
                        continue
                    shutil.copyfileobj(handle, global_out)
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build diagonal ProDive paths from background-filtered PK files."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing kl_*_filtered.pk files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for path summary CSV files.")
    parser.add_argument("--mapping-file", default="", help="Optional record mapping file used to recover sanitized query names.")
    parser.add_argument("--fragment", type=int, default=6, help="Fragment/window length. Default: 6.")
    parser.add_argument("--min-path-points", type=int, default=5, help="Minimum number of linked window pairs required for a path. Default: 5.")
    parser.add_argument("--overlap-iou-threshold", type=float, default=0.1, help="Main-axis IoU threshold used to cluster overlapping paths. Default: 0.1.")
    parser.add_argument("--top-n", type=int, default=None, help="Optional maximum number of clean paths retained per comparison.")
    parser.add_argument("--workers", type=int, default=20, help="Number of worker processes. Default: 20.")
    parser.add_argument("--allowed-list", default="", help="Optional file containing record IDs. A pair is processed if either side is listed.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary per-query CSV files.")
    parser.add_argument("--strict", action="store_true", help="Raise target-level exceptions instead of recording them in the summary.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.fragment <= 1:
        raise ValueError("--fragment must be greater than 1")
    if args.min_path_points <= 0:
        raise ValueError("--min-path-points must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_clean_dir = output_dir / "temp_clean_csv"
    temp_unclean_dir = output_dir / "temp_unclean_csv"
    temp_gap_dir = output_dir / "temp_gap_csv"
    for path in [temp_clean_dir, temp_unclean_dir, temp_gap_dir]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    mapping_names = load_mapping_names(args.mapping_file) if args.mapping_file else []
    safe_to_record = {safe_filename_component(name): name for name in mapping_names}

    allowed_set = load_allowed_list(args.allowed_list) if args.allowed_list else None

    all_pk_files = sorted(glob.glob(str(input_dir / "kl_*_filtered.pk")))
    if not all_pk_files:
        raise FileNotFoundError(f"No kl_*_filtered.pk files found under: {input_dir}")

    print("=" * 80)
    print("Build ProDive diagonal paths")
    print(f"Input PK directory: {input_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Filtered PK files:  {len(all_pk_files):,}")
    print(f"Fragment length:    {args.fragment}")
    print(f"Min path points:    {args.min_path_points}")
    print(f"Workers:            {args.workers}")
    print("=" * 80)

    t0 = time.time()
    results: List[Dict[str, Any]] = []

    if args.workers == 1:
        init_worker(args, allowed_set, safe_to_record)
        iterator: Iterable[str] = all_pk_files
        if HAS_TQDM:
            iterator = tqdm(all_pk_files, desc="Path building", unit="file")
        for fp in iterator:
            results.append(analyze_pk_file(fp))
    else:
        with mp.Pool(
            processes=args.workers,
            maxtasksperchild=1,
            initializer=init_worker,
            initargs=(args, allowed_set, safe_to_record),
        ) as pool:
            iterator = pool.imap_unordered(analyze_pk_file, all_pk_files)
            if HAS_TQDM:
                iterator = tqdm(iterator, total=len(all_pk_files), desc="Path building", unit="file")
            for result in iterator:
                results.append(result)

    total_clean = sum(int(r.get("clean", 0)) for r in results)
    total_unclean = sum(int(r.get("unclean", 0)) for r in results)
    total_gap = sum(int(r.get("gap", 0)) for r in results)
    failed_files = [r for r in results if not r.get("ok")]
    target_error_count = sum(int(r.get("target_errors", 0)) for r in results)

    clean_csv = output_dir / "global_high_score_summary.csv"
    unclean_csv = output_dir / "global_UNCL_overlap_report.csv"
    gap_csv = output_dir / "global_jump_gap_report.csv"

    if total_clean > 0:
        merge_csv_files(temp_clean_dir, clean_csv, "clean path CSV")
    if total_unclean > 0:
        merge_csv_files(temp_unclean_dir, unclean_csv, "unclean path CSV")
    if total_gap > 0:
        merge_csv_files(temp_gap_dir, gap_csv, "gap/jump path CSV")

    if not args.keep_temp:
        for path in [temp_clean_dir, temp_unclean_dir, temp_gap_dir]:
            shutil.rmtree(path, ignore_errors=True)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "filtered_pk_files": len(all_pk_files),
        "fragment": int(args.fragment),
        "min_path_points": int(args.min_path_points),
        "overlap_iou_threshold": float(args.overlap_iou_threshold),
        "workers": int(args.workers),
        "clean_paths": int(total_clean),
        "unclean_paths": int(total_unclean),
        "gap_paths": int(total_gap),
        "failed_files": len(failed_files),
        "failed_file_examples_first_50": failed_files[:50],
        "target_error_count": int(target_error_count),
        "target_error_examples_first_50": [
            {"file": r.get("file"), "examples": r.get("target_error_examples", [])}
            for r in results
            if r.get("target_error_examples")
        ][:50],
        "outputs": {
            "clean_csv": str(clean_csv) if total_clean > 0 else "",
            "unclean_csv": str(unclean_csv) if total_unclean > 0 else "",
            "gap_csv": str(gap_csv) if total_gap > 0 else "",
        },
        "duration_seconds": round(time.time() - t0, 3),
    }

    summary_path = output_dir / "path_building_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Path building completed")
    print(f"Clean paths:   {total_clean:,}")
    print(f"Unclean paths: {total_unclean:,}")
    print(f"Gap paths:     {total_gap:,}")
    print(f"Summary:       {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    mp.freeze_support()
    main()
