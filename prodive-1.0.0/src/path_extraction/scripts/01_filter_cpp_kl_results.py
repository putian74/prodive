#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply background-based threshold filtering to C++/CUDA ProDive KL matrices.

Input
-----
The C++/CUDA stage writes one dense NumPy matrix per HHM-record pair:

    kl_0000_0001.npy

Each matrix element is the raw normalized divergence for one fixed-length
window pair. This script applies the integrated first/second filtering rule:

    raw4 = round(raw_score, 4)
    keep raw4 < raw_threshold
    linear = (query_background[row] + target_background[col]) / (raw4 + epsilon) * (3 * fragment + 2)
    keep linear >= filter_threshold

Output
------
For each query record ID, the output is a filtered pickle file:

    kl_<query_record_id>_filtered.pk

The pickle structure is compatible with the downstream path-building stage:

    {
        target_record_name: {
            query_window_start_1based: {
                target_window_start_1based: round(linear, 4)
            }
        }
    }

Coordinate convention
---------------------
C++ KL matrices are 0-based by array index. The output pickle stores 1-based
window-start coordinates for compatibility with the original downstream code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


DEFAULT_FRAGMENT = 6
DEFAULT_RAW_THRESHOLD = 2.0
DEFAULT_FILTER_THRESHOLD = 13.0
DEFAULT_EPSILON = 0.0001
DEFAULT_ROW_BLOCK_SIZE = 2048
DEFAULT_BACKGROUND_DTYPE = "float32"


# ==============================================================================
# Mapping and background loading
# ==============================================================================


def canonical_name(value: Any, use_pfam_regex: bool = False) -> str:
    """Return the record name used for matching mapping and background keys."""
    text = str(value).strip()
    if use_pfam_regex:
        match = re.search(r"PF\d{5}", text)
        if match:
            return match.group(0)
    return text


def safe_filename_component(value: str) -> str:
    """Create a filesystem-safe output-name component without changing record IDs inside files."""
    value = str(value).strip()
    value = value.replace(os.sep, "_")
    if os.altsep:
        value = value.replace(os.altsep, "_")
    value = re.sub(r"[^A-Za-z0-9._+-]+", "_", value)
    value = value.strip("._")
    return value or "record"


def load_mapping(mapping_file: str, use_pfam_regex: bool = False) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Load a mapping file with lines such as `record_name: 0`."""
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}

    with open(mapping_file, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or ":" not in line:
                continue
            left, right = line.split(":", 1)
            name = canonical_name(left, use_pfam_regex=use_pfam_regex)
            try:
                idx = int(right.strip())
            except ValueError:
                print(f"[WARN] Bad mapping value at line {line_no}: {line}", file=sys.stderr)
                continue
            if idx in id_to_name:
                raise ValueError(
                    f"Duplicate numeric ID {idx} in mapping file: {id_to_name[idx]!r} and {name!r}"
                )
            if name in name_to_id:
                raise ValueError(
                    f"Duplicate record name {name!r} in mapping file: IDs {name_to_id[name]} and {idx}"
                )
            id_to_name[idx] = name
            name_to_id[name] = idx

    if not id_to_name:
        raise RuntimeError(f"No valid mapping entries were loaded from: {mapping_file}")

    return id_to_name, name_to_id


def dtype_from_name(dtype_name: str) -> np.dtype:
    if dtype_name == "float32":
        return np.dtype(np.float32)
    if dtype_name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_background_json(
    background_json: str,
    id_to_name: Dict[int, str],
    dtype: np.dtype,
    use_pfam_regex: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load per-record background/window-information arrays.

    Supported JSON keys:
      - exact record names from the mapping file
      - numeric record IDs as strings
      - Pfam IDs embedded in keys when --pfam-id-regex is used
    """
    with open(background_json, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    background: Dict[str, np.ndarray] = {}

    for key, values in raw.items():
        key_text = str(key).strip()

        if key_text.isdigit() and int(key_text) in id_to_name:
            name = id_to_name[int(key_text)]
        else:
            name = canonical_name(key_text, use_pfam_regex=use_pfam_regex)

        background[name] = np.asarray(values, dtype=dtype)

    if not background:
        raise RuntimeError(f"No background arrays were loaded from: {background_json}")

    return background


# ==============================================================================
# C++ output discovery
# ==============================================================================


CPP_NPY_RE = re.compile(r"^kl_(\d+)_(\d+)\.npy$")


def parse_cpp_npy_filename(path: Path) -> Optional[Tuple[int, int]]:
    match = CPP_NPY_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def find_cpp_npy_files(input_dir: str) -> List[Path]:
    paths: List[Path] = []
    root = Path(input_dir)
    for path in root.rglob("kl_*_*.npy"):
        if path.is_file() and parse_cpp_npy_filename(path) is not None:
            paths.append(path)
    return sorted(paths)


def cpp_output_filename(i: int, j: int) -> str:
    """Return the exact filename pattern used by the C++/CUDA program."""
    return f"kl_{int(i):04d}_{int(j):04d}.npy"


def split_csv_or_whitespace(line: str) -> List[str]:
    """Split a pair-list row using comma, tab, or whitespace separators."""
    return [token for token in re.split(r"[,\t\s]+", line.strip()) if token]


def is_int_token(token: str) -> bool:
    try:
        int(token)
        return True
    except Exception:
        return False


def load_pair_list(pair_list_file: str) -> List[Tuple[int, int]]:
    """
    Load the same style of pair-list file accepted by the C++ program.

    Supported formats:
      - two-column file without a header: `0,1` or `0 1`
      - CSV/TSV/text file with columns named `i` and `j`
      - wider CSV with columns such as `family_i`/`family_j` or `start_i`/`start_j`

    Duplicate pairs are removed while preserving the first occurrence.
    """
    pairs: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    header_processed = False
    i_col = 0
    j_col = 1

    with open(pair_list_file, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = split_csv_or_whitespace(line)
            if not tokens:
                continue

            if not header_processed:
                header_processed = True
                first_two_integer = len(tokens) >= 2 and is_int_token(tokens[0]) and is_int_token(tokens[1])
                if not first_two_integer:
                    lower = [x.lower() for x in tokens]
                    i_candidates = ["i", "family_i", "idx_i", "start_i"]
                    j_candidates = ["j", "family_j", "idx_j", "start_j"]
                    i_col = next((lower.index(x) for x in i_candidates if x in lower), -1)
                    j_col = next((lower.index(x) for x in j_candidates if x in lower), -1)
                    if i_col < 0 or j_col < 0:
                        raise ValueError(
                            f"Pair-list header must contain i and j columns, or use a two-column no-header file: {pair_list_file}"
                        )
                    continue

            if len(tokens) <= max(i_col, j_col):
                raise ValueError(f"Malformed pair-list line {line_no}: not enough columns")
            if not is_int_token(tokens[i_col]) or not is_int_token(tokens[j_col]):
                raise ValueError(f"Malformed pair-list line {line_no}: i/j are not integers")

            pair = (int(tokens[i_col]), int(tokens[j_col]))
            if pair[0] < 0 or pair[1] < 0:
                raise ValueError(f"Invalid pair-list line {line_no}: pair indices must be non-negative: {pair}")
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)

    if not pairs:
        raise RuntimeError(f"Pair list contains no valid pairs: {pair_list_file}")
    return pairs


def resolve_cpp_files_from_expected_pairs(
    input_dir: str,
    expected_pairs: List[Tuple[int, int]],
) -> Tuple[List[Path], List[Tuple[int, int, str]]]:
    """Resolve expected C++ result files and report missing outputs."""
    root = Path(input_dir)
    existing_files: List[Path] = []
    missing: List[Tuple[int, int, str]] = []

    for i, j in expected_pairs:
        path = root / cpp_output_filename(i, j)
        if path.is_file():
            existing_files.append(path)
        else:
            missing.append((int(i), int(j), str(path)))

    return existing_files, missing



def resolve_cpp_run_manifest(input_dir: str, manifest_path: Optional[str]) -> Optional[Path]:
    """Return the C++ run manifest path if available."""
    if manifest_path:
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"C++ run manifest not found: {path}")
        return path

    default_path = Path(input_dir) / "prodive_run_manifest.json"
    if default_path.is_file():
        return default_path
    return None


def load_cpp_run_manifest(manifest_path: Path) -> Dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise RuntimeError(f"C++ run manifest is not a JSON object: {manifest_path}")
    return obj


def expand_range_pairs(start_i: int, end_i: int, start_j: int, end_j: int, reconstruct_symmetric: bool) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for i in range(int(start_i), int(end_i)):
        j_start = max(i, int(start_j)) if reconstruct_symmetric else int(start_j)
        for j in range(j_start, int(end_j)):
            pairs.append((int(i), int(j)))
    return pairs


def load_pairs_from_cpp_manifest(input_dir: str, manifest: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Load the expected pair set declared by the C++ stage.

    Pair-list mode reads prodive_computed_pairs.csv.  Range mode reads
    prodive_computed_ranges.csv when present, otherwise falls back to the range
    fields embedded in prodive_run_manifest.json.
    """
    root = Path(input_dir)
    mode = str(manifest.get("mode", "")).strip()

    if mode == "pair_list":
        pair_file = str(manifest.get("computed_pairs_file") or "prodive_computed_pairs.csv")
        pair_path = root / pair_file
        return load_pair_list(str(pair_path))

    if mode == "range":
        range_file = str(manifest.get("computed_ranges_file") or "prodive_computed_ranges.csv")
        range_path = root / range_file
        if range_path.is_file():
            pairs: List[Tuple[int, int]] = []
            with range_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if not row:
                        continue
                    reconstruct_text = str(row.get("reconstruct_symmetric", manifest.get("reconstruct_symmetric", True))).strip().lower()
                    reconstruct = reconstruct_text in {"1", "true", "yes", "y"}
                    pairs.extend(expand_range_pairs(
                        int(row["start_i"]),
                        int(row["end_i"]),
                        int(row["start_j"]),
                        int(row["end_j"]),
                        reconstruct,
                    ))
            if not pairs:
                raise RuntimeError(f"C++ range manifest contains no expected pairs: {range_path}")
            return pairs

        return expand_range_pairs(
            int(manifest["start_i"]),
            int(manifest["end_i"]),
            int(manifest["start_j"]),
            int(manifest["end_j"]),
            bool(manifest.get("reconstruct_symmetric", True)),
        )

    raise RuntimeError(f"Unsupported C++ manifest mode: {mode!r}")


def validate_cpp_manifest_completion(input_dir: str, manifest_path: Path, manifest: Dict[str, Any], allow_incomplete: bool) -> None:
    """Validate that the C++ program declared the run complete before filtering."""
    status = str(manifest.get("status", "")).strip().lower()
    marker_name = str(manifest.get("completion_marker") or "RUN_COMPLETE")
    marker_path = Path(input_dir) / marker_name

    if status != "completed" or not marker_path.is_file():
        message = (
            f"C++ run is not marked complete. manifest={manifest_path}, "
            f"status={status!r}, completion_marker_exists={marker_path.is_file()}"
        )
        if allow_incomplete:
            print(f"[WARN] {message}")
        else:
            raise RuntimeError(message)


def resolve_selected_pairs(args: argparse.Namespace) -> Optional[set[Tuple[int, int]]]:
    if args.pair is None and args.query_id is None:
        return None

    selected: set[Tuple[int, int]] = set()

    if args.pair:
        for item in args.pair:
            parts = re.split(r"[,\s]+", item.strip())
            parts = [p for p in parts if p]
            if len(parts) != 2:
                raise ValueError(f"Bad --pair value: {item!r}; expected i,j")
            selected.add((int(parts[0]), int(parts[1])))

    # query_id restricts by the first numeric ID in the output filename.
    if args.query_id is not None:
        q = int(args.query_id)
        if not selected:
            selected.add((q, -1))
        else:
            selected = {p for p in selected if p[0] == q}

    return selected


def pair_is_selected(pair: Tuple[int, int], selected: Optional[set[Tuple[int, int]]]) -> bool:
    if selected is None:
        return True
    i, j = pair
    return (i, j) in selected or (i, -1) in selected


# ==============================================================================
# Filtering logic
# ==============================================================================


def threshold_mask(raw4: np.ndarray, threshold: float, inclusive: bool) -> np.ndarray:
    if inclusive:
        return np.isfinite(raw4) & (raw4 <= threshold)
    return np.isfinite(raw4) & (raw4 < threshold)


def filter_one_cpp_matrix(
    matrix_path: Path,
    query_id: int,
    target_id: int,
    query_name: str,
    target_name: str,
    query_bg: np.ndarray,
    target_bg: np.ndarray,
    fragment: int,
    raw_threshold: float,
    raw_threshold_inclusive: bool,
    filter_threshold: float,
    epsilon: float,
    row_block_size: int,
) -> Tuple[Dict[str, Dict[int, Dict[int, float]]], Dict[str, Any]]:
    """Filter one dense C++ KL matrix and return target-keyed nested output."""
    arr = np.load(matrix_path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D KL matrix, got shape {arr.shape}: {matrix_path}")

    n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])
    usable_rows = min(n_rows, len(query_bg))
    usable_cols = min(n_cols, len(target_bg))

    const_factor = float(3 * fragment + 2)

    target_output: Dict[int, Dict[int, float]] = {}

    stats = {
        "matrix_file": str(matrix_path),
        "query_id": int(query_id),
        "target_id": int(target_id),
        "query_name": query_name,
        "target_name": target_name,
        "matrix_shape": [n_rows, n_cols],
        "query_background_length": int(len(query_bg)),
        "target_background_length": int(len(target_bg)),
        "usable_shape": [int(usable_rows), int(usable_cols)],
        "input_points_in_usable_shape": int(usable_rows * usable_cols),
        "raw_pass_points": 0,
        "linear_positive_points": 0,
        "second_pass_points": 0,
        "skipped_due_to_shape_mismatch": bool(n_rows != len(query_bg) or n_cols != len(target_bg)),
        "min_log10_linear": None,
        "max_log10_linear": None,
    }

    if usable_rows <= 0 or usable_cols <= 0:
        return {}, stats

    for row_start in range(0, usable_rows, row_block_size):
        row_end = min(row_start + row_block_size, usable_rows)
        raw_block = np.asarray(arr[row_start:row_end, :usable_cols], dtype=np.float64)
        raw4_block = np.round(raw_block, 4)

        raw_mask = threshold_mask(raw4_block, raw_threshold, raw_threshold_inclusive)
        if not np.any(raw_mask):
            continue

        local_rows, cols_0 = np.nonzero(raw_mask)
        rows_0 = local_rows + row_start
        raw4_values = raw4_block[local_rows, cols_0]
        stats["raw_pass_points"] += int(raw4_values.size)

        q_vals = query_bg[rows_0]
        t_vals = target_bg[cols_0]
        linear_values = (q_vals + t_vals) / (raw4_values + epsilon) * const_factor

        positive_mask = np.isfinite(linear_values) & (linear_values > 0)
        if not np.any(positive_mask):
            continue

        rows_0 = rows_0[positive_mask]
        cols_0 = cols_0[positive_mask]
        linear_values = linear_values[positive_mask]
        stats["linear_positive_points"] += int(linear_values.size)

        log_values = np.log10(linear_values + 1e-9)
        local_min = float(log_values.min())
        local_max = float(log_values.max())
        if stats["min_log10_linear"] is None or local_min < stats["min_log10_linear"]:
            stats["min_log10_linear"] = local_min
        if stats["max_log10_linear"] is None or local_max > stats["max_log10_linear"]:
            stats["max_log10_linear"] = local_max

        second_mask = linear_values >= filter_threshold
        if not np.any(second_mask):
            continue

        rows_keep = rows_0[second_mask]
        cols_keep = cols_0[second_mask]
        linear_keep = linear_values[second_mask]
        stats["second_pass_points"] += int(linear_keep.size)

        for r0, c0, value in zip(rows_keep, cols_keep, linear_keep):
            # Downstream path-building code expects 1-based window-start coordinates.
            r1 = int(r0) + 1
            c1 = int(c0) + 1
            target_output.setdefault(r1, {})[c1] = round(float(value), 4)

    if stats["min_log10_linear"] is None:
        stats["min_log10_linear"] = 0.0
    if stats["max_log10_linear"] is None:
        stats["max_log10_linear"] = 0.0

    if target_output:
        return {target_name: target_output}, stats
    return {}, stats


def merge_target_output(
    query_output: Dict[str, Dict[int, Dict[int, float]]],
    pair_output: Dict[str, Dict[int, Dict[int, float]]],
) -> None:
    """Merge one pair result into one query-level output object."""
    for target_name, rows in pair_output.items():
        target_store = query_output.setdefault(target_name, {})
        for row, cols in rows.items():
            row_store = target_store.setdefault(int(row), {})
            for col, value in cols.items():
                row_store[int(col)] = float(value)


def write_query_outputs(
    output_dir: Path,
    query_outputs: Dict[int, Dict[str, Dict[int, Dict[int, float]]]],
    id_to_name: Dict[int, str],
) -> Dict[int, str]:
    paths: Dict[int, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for query_id, obj in sorted(query_outputs.items()):
        query_name = id_to_name.get(query_id, str(query_id))
        safe_query = safe_filename_component(query_name)
        path = output_dir / f"kl_{safe_query}_filtered.pk"
        with open(path, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        paths[query_id] = str(path)

    return paths


# ==============================================================================
# Main driver
# ==============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter C++/CUDA ProDive KL .npy matrices using background-normalized scores."
    )

    parser.add_argument("--input-dir", required=True,
                        help="Directory containing C++/CUDA KL output files named kl_<i>_<j>.npy.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for filtered PK outputs and summary files.")
    parser.add_argument("--mapping-file", required=True,
                        help="Mapping file produced by 01_generate_pfam_hhm_mapping.py.")
    parser.add_argument("--background-json", required=True,
                        help="JSON file containing per-record background/window-information arrays.")

    parser.add_argument("--fragment", type=int, default=DEFAULT_FRAGMENT,
                        help=f"Fragment length. Default: {DEFAULT_FRAGMENT}.")
    parser.add_argument("--raw-threshold", type=float, default=DEFAULT_RAW_THRESHOLD,
                        help=f"First threshold applied after round(raw, 4). Default: {DEFAULT_RAW_THRESHOLD}.")
    parser.add_argument("--raw-threshold-inclusive", action="store_true",
                        help="Use <= raw-threshold instead of the default strict < raw-threshold.")
    parser.add_argument("--filter-threshold", type=float, default=DEFAULT_FILTER_THRESHOLD,
                        help=f"Second threshold applied to the linear background-normalized score. Default: {DEFAULT_FILTER_THRESHOLD}.")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                        help=f"Small denominator stabilizer. Default: {DEFAULT_EPSILON}.")

    parser.add_argument("--background-dtype", choices=["float32", "float64"], default=DEFAULT_BACKGROUND_DTYPE,
                        help=f"dtype used when loading background arrays. Default: {DEFAULT_BACKGROUND_DTYPE}.")
    parser.add_argument("--row-block-size", type=int, default=DEFAULT_ROW_BLOCK_SIZE,
                        help=f"Number of matrix rows processed per block. Default: {DEFAULT_ROW_BLOCK_SIZE}.")

    parser.add_argument("--skip-self-comparison", dest="skip_self_comparison", action="store_true", default=True,
                        help="Skip files where query_id == target_id. Enabled by default.")
    parser.add_argument("--include-self-comparison", dest="skip_self_comparison", action="store_false",
                        help="Include self-comparison matrices. Mainly useful for debugging.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip a query if its query-level filtered PK already exists.")

    parser.add_argument("--query-id", type=int, default=None,
                        help="Optional: process only files whose first numeric ID equals this query ID.")
    parser.add_argument("--pair", action="append", default=None,
                        help="Optional explicit pair to process, written as i,j. Can be supplied multiple times.")
    parser.add_argument("--cpp-run-manifest", default=None,
                        help="Optional path to prodive_run_manifest.json. If omitted, input-dir/prodive_run_manifest.json is used when present.")
    parser.add_argument("--allow-incomplete-cpp", action="store_true",
                        help="Allow filtering when the C++ manifest is missing RUN_COMPLETE or has status other than completed. Not recommended for reproducible runs.")
    parser.add_argument("--ignore-cpp-manifest", action="store_true",
                        help="Ignore prodive_run_manifest.json and scan existing kl_*.npy files directly. Mainly for legacy output directories.")
    parser.add_argument("--expected-pair-list", default=None,
                        help="Legacy option: pair-list file used to validate C++ completion before filtering. Prefer the C++ prodive_run_manifest.json.")
    parser.add_argument("--require-complete-cpp", action="store_true",
                        help="Legacy option: fail if any pair from --expected-pair-list is missing from the C++ output directory.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Optional explicit test mode: process at most this many KL .npy files.")
    parser.add_argument("--pfam-id-regex", action="store_true",
                        help="Use embedded PFxxxxx IDs when matching mapping/background names. Disabled by default to support arbitrary HHM names.")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.fragment <= 0:
        raise ValueError("--fragment must be positive")
    if args.row_block_size <= 0:
        raise ValueError("--row-block-size must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_to_name, name_to_id = load_mapping(args.mapping_file, use_pfam_regex=args.pfam_id_regex)
    background = load_background_json(
        args.background_json,
        id_to_name=id_to_name,
        dtype=dtype_from_name(args.background_dtype),
        use_pfam_regex=args.pfam_id_regex,
    )

    selected = resolve_selected_pairs(args)
    expected_pairs: Optional[List[Tuple[int, int]]] = None
    missing_expected_outputs: List[Tuple[int, int, str]] = []
    cpp_manifest_path: Optional[Path] = None
    cpp_manifest: Optional[Dict[str, Any]] = None

    if not args.ignore_cpp_manifest:
        cpp_manifest_path = resolve_cpp_run_manifest(args.input_dir, args.cpp_run_manifest)

    if cpp_manifest_path is not None:
        cpp_manifest = load_cpp_run_manifest(cpp_manifest_path)
        validate_cpp_manifest_completion(
            args.input_dir,
            cpp_manifest_path,
            cpp_manifest,
            allow_incomplete=args.allow_incomplete_cpp,
        )
        expected_pairs = load_pairs_from_cpp_manifest(args.input_dir, cpp_manifest)
        files, missing_expected_outputs = resolve_cpp_files_from_expected_pairs(args.input_dir, expected_pairs)
        if selected is not None:
            files = [p for p in files if pair_is_selected(parse_cpp_npy_filename(p), selected)]
            missing_expected_outputs = [
                item for item in missing_expected_outputs
                if pair_is_selected((item[0], item[1]), selected)
            ]
        if missing_expected_outputs:
            print(
                f"[WARN] Missing {len(missing_expected_outputs)} expected C++ output file(s) declared by the C++ manifest."
            )
            for i, item in enumerate(missing_expected_outputs[:20], start=1):
                print(f"  missing[{i}]: pair=({item[0]},{item[1]}) path={item[2]}")
            if len(missing_expected_outputs) > 20:
                print(f"  ... {len(missing_expected_outputs) - 20} additional missing file(s) omitted")
            if not args.allow_incomplete_cpp:
                raise RuntimeError(
                    "C++ output validation failed: at least one KL matrix declared by the C++ manifest is missing. "
                    "Do not run filtering until the C++ stage is complete."
                )

    elif args.expected_pair_list:
        expected_pairs = load_pair_list(args.expected_pair_list)
        files, missing_expected_outputs = resolve_cpp_files_from_expected_pairs(args.input_dir, expected_pairs)
        if selected is not None:
            files = [p for p in files if pair_is_selected(parse_cpp_npy_filename(p), selected)]
            missing_expected_outputs = [
                item for item in missing_expected_outputs
                if pair_is_selected((item[0], item[1]), selected)
            ]
        if missing_expected_outputs:
            print(
                f"[WARN] Missing {len(missing_expected_outputs)} expected C++ output file(s) from --expected-pair-list."
            )
            for i, item in enumerate(missing_expected_outputs[:20], start=1):
                print(f"  missing[{i}]: pair=({item[0]},{item[1]}) path={item[2]}")
            if len(missing_expected_outputs) > 20:
                print(f"  ... {len(missing_expected_outputs) - 20} additional missing file(s) omitted")
            if args.require_complete_cpp:
                raise RuntimeError(
                    "C++ output validation failed: at least one expected KL matrix is missing. "
                    "Do not run filtering until the C++ stage is complete, or omit --require-complete-cpp "
                    "to process only existing outputs."
                )
    else:
        print(
            "[WARN] No C++ run manifest was found. Falling back to direct kl_*.npy scanning. "
            "For reproducible runs, use a C++ output directory containing prodive_run_manifest.json and RUN_COMPLETE."
        )
        files = []
        for path in find_cpp_npy_files(args.input_dir):
            pair = parse_cpp_npy_filename(path)
            if pair is None:
                continue
            if pair_is_selected(pair, selected):
                files.append(path)

    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("--max-files must be positive when provided")
        files = files[:args.max_files]
        print(f"[TEST MODE] Processing only the first {len(files)} matched KL files.")

    if not files:
        print("[WARN] No C++ KL .npy files matched the requested criteria.")
        return

    print("=" * 100)
    print("ProDive C++ KL result filtering")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Mapping records: {len(id_to_name)}")
    print(f"Background records: {len(background)}")
    if cpp_manifest_path is not None:
        print(f"C++ run manifest: {cpp_manifest_path}")
        print(f"C++ manifest status: {cpp_manifest.get('status') if cpp_manifest else None}")
        print(f"Expected pairs from C++ manifest: {len(expected_pairs) if expected_pairs is not None else 0}")
        print(f"Missing expected C++ outputs: {len(missing_expected_outputs)}")
    elif args.expected_pair_list:
        print(f"Expected pair-list: {args.expected_pair_list}")
        print(f"Expected pairs: {len(expected_pairs) if expected_pairs is not None else 0}")
        print(f"Missing expected C++ outputs: {len(missing_expected_outputs)}")
    print(f"KL files selected: {len(files)}")
    print(f"Raw score rule: round(raw, 4) {'<=' if args.raw_threshold_inclusive else '<'} {args.raw_threshold}")
    print(f"Linear score rule: linear >= {args.filter_threshold}")
    print(f"Formula: (query_bg[row] + target_bg[col]) / (raw4 + {args.epsilon}) * (3 * {args.fragment} + 2)")
    print("Saved value: round(linear, 4)")
    print("Output coordinates: 1-based window starts")
    print("=" * 100)

    query_outputs: Dict[int, Dict[str, Dict[int, Dict[int, float]]]] = defaultdict(dict)

    # Keep only aggregate run-level statistics.  A previous version wrote one
    # JSONL row per C++ matrix file, which becomes unnecessarily large for
    # all-vs-all or large pair-list runs.
    total_pair_records = 0
    ok_pairs = 0
    failed_pairs = 0
    skipped_pairs = 0
    total_input = 0
    total_raw_pass = 0
    total_second_pass = 0
    failure_reason_counts: Dict[str, int] = defaultdict(int)
    skipped_reason_counts: Dict[str, int] = defaultdict(int)
    failed_pair_examples: List[Dict[str, Any]] = []
    max_failed_examples = 100

    def record_pair_result(row: Dict[str, Any]) -> None:
        nonlocal total_pair_records, ok_pairs, failed_pairs, skipped_pairs
        nonlocal total_input, total_raw_pass, total_second_pass

        total_pair_records += 1

        if row.get("ok"):
            ok_pairs += 1
        else:
            failed_pairs += 1
            reason = str(row.get("reason", "unknown"))
            failure_reason_counts[reason] += 1
            if len(failed_pair_examples) < max_failed_examples:
                failed_pair_examples.append(dict(row))

        if row.get("skipped"):
            skipped_pairs += 1
            reason = str(row.get("reason", "unknown"))
            skipped_reason_counts[reason] += 1

        total_input += int(row.get("input_points_in_usable_shape", 0))
        total_raw_pass += int(row.get("raw_pass_points", 0))
        total_second_pass += int(row.get("second_pass_points", 0))

    iterator: Iterable[Path] = files
    if HAS_TQDM:
        iterator = tqdm(files, desc="Filter KL matrices", unit="file")

    skipped_existing_queries: set[int] = set()

    for matrix_path in iterator:
        parsed = parse_cpp_npy_filename(matrix_path)
        if parsed is None:
            continue
        query_id, target_id = parsed

        query_name = id_to_name.get(query_id)
        target_name = id_to_name.get(target_id)

        summary_base: Dict[str, Any] = {
            "matrix_file": str(matrix_path),
            "query_id": int(query_id),
            "target_id": int(target_id),
            "ok": False,
        }

        if query_name is None or target_name is None:
            summary_base["reason"] = "query_or_target_id_missing_from_mapping"
            record_pair_result(summary_base)
            continue

        if args.skip_self_comparison and query_id == target_id:
            summary_base["ok"] = True
            summary_base["skipped"] = True
            summary_base["reason"] = "self_comparison"
            record_pair_result(summary_base)
            continue

        if args.resume:
            safe_query = safe_filename_component(query_name)
            expected_output = output_dir / f"kl_{safe_query}_filtered.pk"
            if expected_output.exists():
                skipped_existing_queries.add(query_id)
                summary_base["ok"] = True
                summary_base["skipped"] = True
                summary_base["reason"] = "query_output_exists"
                summary_base["output_file"] = str(expected_output)
                record_pair_result(summary_base)
                continue

        query_bg = background.get(query_name)
        target_bg = background.get(target_name)
        if query_bg is None or target_bg is None:
            summary_base["reason"] = "query_or_target_background_missing"
            summary_base["query_name"] = query_name
            summary_base["target_name"] = target_name
            record_pair_result(summary_base)
            continue

        try:
            pair_output, stats = filter_one_cpp_matrix(
                matrix_path=matrix_path,
                query_id=query_id,
                target_id=target_id,
                query_name=query_name,
                target_name=target_name,
                query_bg=query_bg,
                target_bg=target_bg,
                fragment=args.fragment,
                raw_threshold=args.raw_threshold,
                raw_threshold_inclusive=args.raw_threshold_inclusive,
                filter_threshold=args.filter_threshold,
                epsilon=args.epsilon,
                row_block_size=args.row_block_size,
            )
            merge_target_output(query_outputs[query_id], pair_output)
            stats["ok"] = True
            stats["skipped"] = False
            record_pair_result(stats)
        except Exception as exc:
            summary_base["reason"] = repr(exc)
            record_pair_result(summary_base)

    output_paths = write_query_outputs(output_dir, query_outputs, id_to_name)

    run_summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "mapping_file": args.mapping_file,
        "background_json": args.background_json,
        "fragment": int(args.fragment),
        "raw_threshold": float(args.raw_threshold),
        "raw_threshold_rule": "<=" if args.raw_threshold_inclusive else "<",
        "filter_threshold": float(args.filter_threshold),
        "epsilon": float(args.epsilon),
        "const_factor": float(3 * args.fragment + 2),
        "background_dtype": args.background_dtype,
        "row_block_size": int(args.row_block_size),
        "cpp_run_manifest": str(cpp_manifest_path) if cpp_manifest_path is not None else None,
        "cpp_manifest_status": cpp_manifest.get("status") if cpp_manifest else None,
        "cpp_manifest_mode": cpp_manifest.get("mode") if cpp_manifest else None,
        "allow_incomplete_cpp": bool(args.allow_incomplete_cpp),
        "ignore_cpp_manifest": bool(args.ignore_cpp_manifest),
        "expected_pair_list": args.expected_pair_list,
        "expected_pairs": int(len(expected_pairs)) if expected_pairs is not None else None,
        "require_complete_cpp": bool(args.require_complete_cpp),
        "missing_expected_cpp_outputs_count": int(len(missing_expected_outputs)),
        "missing_expected_cpp_outputs_first_100": [
            {"i": int(i), "j": int(j), "path": path}
            for i, j, path in missing_expected_outputs[:100]
        ],
        "selected_matrix_files": int(len(files)),
        "pair_records_seen": int(total_pair_records),
        "ok_pairs": int(ok_pairs),
        "failed_pairs": int(failed_pairs),
        "skipped_pairs": int(skipped_pairs),
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        "failed_pair_examples_first_100": failed_pair_examples,
        "query_outputs_written": {str(k): v for k, v in sorted(output_paths.items())},
        "skipped_existing_query_ids": sorted(int(x) for x in skipped_existing_queries),
        "total_input_points_in_usable_shape": int(total_input),
        "total_raw_pass_points": int(total_raw_pass),
        "total_second_pass_points": int(total_second_pass),
        "score_definition": "linear = (query_background[row] + target_background[col]) / (round(raw_score, 4) + epsilon) * (3 * fragment + 2)",
        "saved_value": "round(linear, 4)",
        "coordinate_system": "input NPY arrays use 0-based row/column indices; output PK files use 1-based window-start coordinates",
    }

    summary_path = output_dir / "filter_cpp_kl_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("DONE")
    print(f"Pairs processed or skipped successfully: {ok_pairs}")
    print(f"Failed pairs: {failed_pairs}")
    print(f"Query-level PK files written: {len(output_paths)}")
    print(f"Total raw-pass points: {total_raw_pass:,}")
    print(f"Total second-pass points: {total_second_pass:,}")
    print(f"Run summary: {summary_path}")
    print("Per-pair JSONL summary is disabled; aggregate counts are stored in the run summary.")
    print("=" * 100)


if __name__ == "__main__":
    main()
